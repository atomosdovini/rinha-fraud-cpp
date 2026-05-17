// build_index.cpp — Offline builder for the exact BVH index.
//
// Pipeline: gunzip references.json.gz → parse into int16[14] vectors + labels
// → bucket by partition_key (up to 256 partitions) → build a median-split
// KD-tree per bucket with min/max bounding boxes → write index.bin.
//
// Usage: build-index references.json.gz index.bin [leaf_size=128]

#include "index.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <zlib.h>

using Clock = std::chrono::steady_clock;
using rinha::Block;
using rinha::Dims;
using rinha::Node;
using rinha::Partition;

// ── Reference parsing ─────────────────────────────────────────────────────────

static std::vector<char> read_gzip(const std::string& path) {
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<char> out;
    out.reserve(300 * 1024 * 1024);
    std::array<char, 1 << 20> buf{};
    for (;;) {
        int n = gzread(f, buf.data(), static_cast<unsigned>(buf.size()));
        if (n < 0) {
            int err = 0;
            const char* msg = gzerror(f, &err);
            gzclose(f);
            throw std::runtime_error(msg ? msg : "gzread failed");
        }
        if (n == 0) break;
        out.insert(out.end(), buf.data(), buf.data() + n);
    }
    gzclose(f);
    out.push_back('\0');
    return out;
}

static const char* find_or_die(const char* p, const char* needle) {
    const char* q = std::strstr(p, needle);
    if (!q) throw std::runtime_error(std::string("missing token ") + needle);
    return q;
}

static void parse_refs(const std::string& path,
                       std::vector<int16_t>& vectors,
                       std::vector<uint8_t>& labels) {
    auto raw = read_gzip(path);
    vectors.reserve(size_t(3'000'000) * Dims);
    labels.reserve(3'000'000);

    const char* p = raw.data();
    while ((p = std::strstr(p, "\"vector\"")) != nullptr) {
        p = std::strchr(p, '[');
        if (!p) throw std::runtime_error("bad vector");
        ++p;
        for (int d = 0; d < Dims; ++d) {
            char* end = nullptr;
            float v = std::strtof(p, &end);
            if (end == p) throw std::runtime_error("bad float");
            vectors.push_back(rinha::qround(v));
            p = end;
            while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
        }
        const char* l     = find_or_die(p, "\"label\"");
        const char* colon = std::strchr(l, ':');
        const char* quote = std::strchr(colon, '"');
        if (!quote) throw std::runtime_error("bad label");
        labels.push_back(quote[1] == 'f' ? 1 : 0);
        p = quote + 1;
    }

    if (labels.empty() || vectors.size() != labels.size() * Dims)
        throw std::runtime_error("reference parse produced inconsistent data");
}

// ── KD-tree construction ──────────────────────────────────────────────────────

namespace {

const int16_t*       g_vectors  = nullptr;   // flat n*Dims
std::vector<Node>    g_nodes;
std::vector<uint32_t> g_leaf_order;          // reference ids in leaf-scan order
uint32_t             g_block_count = 0;
int                  g_leaf_size   = 128;

inline const int16_t* vec(uint32_t id) { return g_vectors + size_t(id) * Dims; }

// Compute the bounding box of ids[lo,hi) into a node.
void bbox(const uint32_t* ids, int lo, int hi, Node& n) {
    int16_t mn[Dims], mx[Dims];
    const int16_t* v0 = vec(ids[lo]);
    for (int d = 0; d < Dims; ++d) { mn[d] = v0[d]; mx[d] = v0[d]; }
    for (int i = lo + 1; i < hi; ++i) {
        const int16_t* v = vec(ids[i]);
        for (int d = 0; d < Dims; ++d) {
            if (v[d] < mn[d]) mn[d] = v[d];
            if (v[d] > mx[d]) mx[d] = v[d];
        }
    }
    std::memcpy(n.min, mn, sizeof(mn));
    std::memcpy(n.max, mx, sizeof(mx));
}

// Build a subtree over ids[lo,hi); returns its node index. The id array is
// reordered in place so each leaf owns a contiguous run.
int build(std::vector<uint32_t>& ids, int lo, int hi) {
    int self = int(g_nodes.size());
    g_nodes.emplace_back();
    {
        Node& n = g_nodes[self];
        bbox(ids.data(), lo, hi, n);
        n.left = n.right = -1;
        n.start = 0;
        n.len   = hi - lo;
    }

    const int count = hi - lo;
    if (count <= g_leaf_size) {
        Node& n = g_nodes[self];
        n.start = int(g_block_count);
        n.len   = count;
        g_block_count += uint32_t((count + Block - 1) / Block);
        for (int i = lo; i < hi; ++i) g_leaf_order.push_back(ids[i]);
        return self;
    }

    // Split on the dimension with the widest spread, at the median.
    int   split_dim = 0;
    int   best_span = -1;
    for (int d = 0; d < Dims; ++d) {
        const int16_t* mn = g_nodes[self].min;
        const int16_t* mx = g_nodes[self].max;
        int span = int(mx[d]) - int(mn[d]);
        if (span > best_span) { best_span = span; split_dim = d; }
    }
    int mid = lo + count / 2;
    std::nth_element(ids.begin() + lo, ids.begin() + mid, ids.begin() + hi,
                     [split_dim](uint32_t a, uint32_t b) {
                         return vec(a)[split_dim] < vec(b)[split_dim];
                     });

    int l = build(ids, lo, mid);
    int r = build(ids, mid, hi);
    g_nodes[self].left  = l;
    g_nodes[self].right = r;
    return self;
}

}  // namespace

// ── Index file writer ─────────────────────────────────────────────────────────

static void write_index(const std::string& out_path,
                         const std::vector<int16_t>& vectors,
                         const std::vector<uint8_t>& labels) {
    const uint32_t n = uint32_t(labels.size());
    g_vectors = vectors.data();

    // Bucket reference ids by partition key.
    std::vector<std::vector<uint32_t>> buckets(256);
    for (uint32_t i = 0; i < n; ++i)
        buckets[rinha::partition_key(vec(i)) & 255u].push_back(i);

    g_nodes.clear();
    g_leaf_order.clear();
    g_leaf_order.reserve(n);
    g_block_count = 0;

    std::vector<Partition> parts;
    for (uint32_t key = 0; key < 256; ++key) {
        auto& ids = buckets[key];
        if (ids.empty()) continue;
        int root = build(ids, 0, int(ids.size()));
        Partition pt{};
        pt.key    = key;
        pt.root   = root;
        pt.length = int32_t(ids.size());
        std::memcpy(pt.min, g_nodes[root].min, sizeof(pt.min));
        std::memcpy(pt.max, g_nodes[root].max, sizeof(pt.max));
        parts.push_back(pt);
    }

    const uint32_t part_count  = uint32_t(parts.size());
    const uint32_t node_count  = uint32_t(g_nodes.size());
    const uint32_t block_count = g_block_count;

    // Emit vectors (block-major, dimension-major within a block) and labels in
    // leaf-scan order, so each leaf's blocks are contiguous.
    std::vector<int16_t> blocks(size_t(block_count) * Dims * Block, 0);
    std::vector<uint8_t> out_labels(size_t(block_count) * Block, 0);

    // Leaf block ranges were assigned sequentially during build(); replay the
    // node list in the same order, consuming ids from g_leaf_order per leaf.
    {
        size_t cursor = 0;
        for (const Node& nd : g_nodes) {
            if (nd.left >= 0) continue;            // internal
            for (int j = 0; j < nd.len; ++j) {
                uint32_t id   = g_leaf_order[cursor++];
                size_t   blk  = size_t(nd.start) + size_t(j) / Block;
                int      lane = j % Block;
                const int16_t* src = vec(id);
                int16_t* dst = blocks.data() + blk * Dims * Block;
                for (int d = 0; d < Dims; ++d) dst[d * Block + lane] = src[d];
                out_labels[blk * Block + lane] = labels[id];
            }
        }
        if (cursor != g_leaf_order.size())
            throw std::runtime_error("leaf order mismatch");
    }

    rinha::FileHeader h{};
    h.magic       = rinha::kMagic;
    h.version     = rinha::kVer;
    h.n           = n;
    h.part_count  = part_count;
    h.node_count  = node_count;
    h.block_count = block_count;
    h.dims        = Dims;
    h.block_size  = Block;

    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + out_path);
    auto put = [&](const void* d, size_t len) {
        out.write(static_cast<const char*>(d), std::streamsize(len));
        if (!out) throw std::runtime_error("write failed");
    };
    put(&h, sizeof(h));
    put(parts.data(),      parts.size()      * sizeof(Partition));
    put(g_nodes.data(),    g_nodes.size()    * sizeof(Node));
    put(blocks.data(),     blocks.size()     * sizeof(int16_t));
    put(out_labels.data(), out_labels.size());

    size_t total = sizeof(h)
                 + parts.size() * sizeof(Partition)
                 + g_nodes.size() * sizeof(Node)
                 + blocks.size() * sizeof(int16_t)
                 + out_labels.size();
    std::cerr << "index written: " << out_path << " ("
              << (total / (1024 * 1024)) << " MB)  partitions=" << part_count
              << " nodes=" << node_count << " blocks=" << block_count << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: build-index references.json.gz index.bin [leaf_size=128]\n";
        return 2;
    }
    std::string refs = argv[1];
    std::string out  = argv[2];
    g_leaf_size = argc > 3 ? std::atoi(argv[3]) : 128;
    if (g_leaf_size < Block) g_leaf_size = Block;

    auto t0 = Clock::now();
    std::vector<int16_t> vectors;
    std::vector<uint8_t> labels;
    parse_refs(refs, vectors, labels);
    std::cerr << "parsed " << labels.size() << " refs in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     Clock::now() - t0).count()
              << "ms\n";

    write_index(out, vectors, labels);
    std::cerr << "done in "
              << std::chrono::duration_cast<std::chrono::seconds>(
                     Clock::now() - t0).count()
              << "s\n";
    return 0;
}
