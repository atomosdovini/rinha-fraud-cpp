#pragma once
// index.hpp — Exact k-NN index over the reference transactions.
//
// Structure: a forest of bounding-volume hierarchies (BVH / KD-tree). The
// reference set is split into up to 256 partitions keyed by a hand-crafted
// 8-bit signature of semantically meaningful features (partition_key). Each
// partition holds a binary tree whose nodes carry an axis-aligned min/max
// bounding box. Search is exact branch-and-bound: descend the tree, prune any
// subtree whose minimum possible squared distance already exceeds the current
// 5th-nearest neighbour. No NPROBE, no re-run, no recall tuning — the result
// is always the true 5 nearest neighbours.

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace rinha {

[[noreturn]] inline void die() { std::_Exit(1); }

// ── Constants ─────────────────────────────────────────────────────────────────

constexpr int      Dims   = 14;
constexpr int      Block  = 8;
constexpr int      K      = 5;             // nearest neighbours kept
constexpr int      Scale  = 10000;         // quantisation scale
constexpr uint64_t kMagic = 0x3248564232484E52ULL;  // "RNH2BVH2"
constexpr uint32_t kVer   = 2;

// ── File layout ───────────────────────────────────────────────────────────────

struct FileHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t n;             // total reference vectors
    uint32_t part_count;    // populated partitions
    uint32_t node_count;
    uint32_t block_count;   // blocks of 8 vectors
    uint32_t dims;
    uint32_t block_size;
    uint32_t reserved[7];
};
static_assert(sizeof(FileHeader) == 64);

// One partition (bucket) — POD, read directly from the mapped file.
struct Partition {
    uint32_t key;
    int32_t  root;          // index into the node array
    int32_t  length;        // reference vectors in this partition
    int16_t  min[Dims];
    int16_t  max[Dims];
};
static_assert(sizeof(Partition) == 68);

// One BVH node — POD. left < 0 marks a leaf.
struct Node {
    int32_t left;
    int32_t right;
    int32_t start;          // leaf: first block index
    int32_t len;            // leaf: reference vector count
    int16_t min[Dims];
    int16_t max[Dims];
};
static_assert(sizeof(Node) == 72);

// ── Quantisation helpers ──────────────────────────────────────────────────────

inline int16_t qround(double v) {
    v = v < -1.0 ? -1.0 : v > 1.0 ? 1.0 : v;
    return static_cast<int16_t>(__builtin_llround(v * Scale));
}

inline int16_t qclamp01(double v) {
    v = v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v;
    return static_cast<int16_t>(__builtin_llround(v * Scale));
}

// ── Partition key ─────────────────────────────────────────────────────────────
// 8-bit signature over semantically meaningful features. Two transactions with
// the same key land in the same partition; queries first probe their own key's
// partition, which is almost always where their neighbours live.

inline uint32_t partition_key(const int16_t* v) {
    uint32_t key = 0;
    if (v[5]  >= 0) key |= 1u << 0;   // has a previous transaction
    if (v[9]  >  0) key |= 1u << 1;   // is_online
    if (v[10] >  0) key |= 1u << 2;   // card_present
    if (v[11] >  0) key |= 1u << 3;   // unknown merchant
    if      (v[12] <= 2047) {}        // MCC risk bucket
    else if (v[12] <= 4095) key |= 1u << 4;
    else if (v[12] <= 6143) key |= 2u << 4;
    else                    key |= 3u << 4;
    if (v[2] > 4096) key |= 1u << 6;  // amount / customer-average ratio high
    if (v[8] > 2048) key |= 1u << 7;  // tx_count_24h high
    return key;
}

// ── Distance primitives ───────────────────────────────────────────────────────

// Minimum possible squared L2 distance from q to an axis-aligned box [lo,hi].
inline uint64_t bbox_lb(const int16_t* __restrict__ q,
                        const int16_t* __restrict__ lo,
                        const int16_t* __restrict__ hi) {
    uint64_t acc = 0;
    for (int d = 0; d < Dims; ++d) {
        int64_t e = 0;
        if      (q[d] < lo[d]) e = int64_t(lo[d]) - q[d];
        else if (q[d] > hi[d]) e = int64_t(q[d])  - hi[d];
        acc += uint64_t(e * e);
    }
    return acc;
}

// Squared L2 distances from q to 8 vectors of one block. 64-bit accumulation —
// an exact search visits distant leaves where int32 sums would overflow.
inline void distance_block8(const int16_t* __restrict__ block,
                            const int16_t* __restrict__ q,
                            int64_t out[Block]) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();
    for (int d = 0; d < Dims; ++d) {
        __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(block + d * Block));
        __m256i values = _mm256_cvtepi16_epi32(packed);
        __m256i qd     = _mm256_set1_epi32(int(q[d]));
        __m256i diff   = _mm256_sub_epi32(values, qd);
        __m256i sq     = _mm256_mullo_epi32(diff, diff);
        acc_lo = _mm256_add_epi64(acc_lo,
            _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq)));
        acc_hi = _mm256_add_epi64(acc_hi,
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq, 1)));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),     acc_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 4), acc_hi);
}

// ── BvhIndex ──────────────────────────────────────────────────────────────────

class BvhIndex {
public:
    explicit BvhIndex(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) die();
        struct stat st{};
        if (::fstat(fd, &st) != 0) { ::close(fd); die(); }
        size_ = static_cast<size_t>(st.st_size);
        ::posix_fadvise(fd, 0, static_cast<off_t>(size_), POSIX_FADV_WILLNEED);

        const char* ev = std::getenv("INDEX_MMAP");
        mmap_ = ev && (ev[0] == '1' || std::strcmp(ev, "true") == 0);
        if (mmap_) {
            raw_ = static_cast<uint8_t*>(
                ::mmap(nullptr, size_, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0));
            if (raw_ == MAP_FAILED) { ::close(fd); die(); }
            ::madvise(raw_, size_, MADV_WILLNEED);
            ::close(fd);
        } else {
            buf_.resize(size_);
            size_t off = 0;
            while (off < size_) {
                ssize_t r = ::read(fd, buf_.data() + off, size_ - off);
                if (r > 0)  { off += size_t(r); continue; }
                if (r < 0 && errno == EINTR) continue;
                ::close(fd); die();
            }
            ::close(fd);
            raw_ = buf_.data();
        }

        hdr_ = reinterpret_cast<const FileHeader*>(raw_);
        if (hdr_->magic != kMagic || hdr_->version != kVer ||
            hdr_->dims != Dims    || hdr_->block_size != Block) die();

        size_t p = sizeof(FileHeader);
        parts_  = reinterpret_cast<const Partition*>(raw_ + p);
        p += size_t(hdr_->part_count) * sizeof(Partition);
        nodes_  = reinterpret_cast<const Node*>(raw_ + p);
        p += size_t(hdr_->node_count) * sizeof(Node);
        vecs_   = reinterpret_cast<const int16_t*>(raw_ + p);
        p += size_t(hdr_->block_count) * Dims * Block * sizeof(int16_t);
        labels_ = raw_ + p;
        p += size_t(hdr_->block_count) * Block;
        if (p > size_) die();

        part_count_ = int(hdr_->part_count);
        node_count_ = int(hdr_->node_count);
        for (int i = 0; i < 256; ++i) part_by_key_[i] = -1;
        for (int i = 0; i < part_count_; ++i)
            part_by_key_[parts_[i].key & 255u] = i;

        const char* em = std::getenv("EARLY_DIST_MILLI");
        int milli = em ? std::atoi(em) : 0;
        if (milli > 0) {
            int64_t r = int64_t(Scale) * milli / 1000;
            early_limit_ = r * r;          // stop once the 5th NN is this close
        }

        prefault();
    }

    BvhIndex(const BvhIndex&)            = delete;
    BvhIndex& operator=(const BvhIndex&) = delete;

    ~BvhIndex() {
        if (mmap_ && raw_ && raw_ != MAP_FAILED) ::munmap(raw_, size_);
        raw_ = nullptr;
    }

    uint32_t num_vectors()    const { return hdr_->n; }
    int      num_partitions() const { return part_count_; }
    int      num_nodes()      const { return node_count_; }

    // Returns the number of fraud labels among the 5 nearest neighbours.
    uint8_t query(const int16_t q[Dims]) const {
        Knn knn;

        const uint32_t key   = partition_key(q);
        const int      match = part_by_key_[key & 255u];
        if (match >= 0) {
            if (search_node(parts_[match].root, 0, q, knn))
                return knn.fraud_count();
        }

        struct Cand { int idx; uint64_t bound; };
        Cand cand[256];
        int  cn = 0;
        for (int i = 0; i < part_count_; ++i) {
            if (i == match) continue;
            uint64_t b = bbox_lb(q, parts_[i].min, parts_[i].max);
            if (b >= knn.gate()) continue;
            cand[cn++] = {i, b};
        }
        for (int i = 1; i < cn; ++i) {           // insertion sort by lower bound
            Cand c = cand[i];
            int  j = i - 1;
            while (j >= 0 && cand[j].bound > c.bound) { cand[j + 1] = cand[j]; --j; }
            cand[j + 1] = c;
        }
        for (int i = 0; i < cn; ++i) {
            if (cand[i].bound >= knn.gate()) break;
            if (search_node(parts_[cand[i].idx].root, cand[i].bound, q, knn))
                break;
        }
        return knn.fraud_count();
    }

private:
    // Fixed-size top-5 of nearest neighbours, kept sorted ascending by distance.
    struct Knn {
        std::array<uint64_t, K> dist;
        std::array<uint8_t,  K> label;
        Knn() { dist.fill(UINT64_MAX); label.fill(0); }

        uint64_t gate() const { return dist[K - 1]; }   // current 5th-best

        void insert(uint64_t d, uint8_t l) {
            if (d >= dist[K - 1]) return;
            int pos = K - 1;
            while (pos > 0 && d < dist[pos - 1]) {
                dist[pos]  = dist[pos - 1];
                label[pos] = label[pos - 1];
                --pos;
            }
            dist[pos]  = d;
            label[pos] = l;
        }

        uint8_t fraud_count() const {
            return uint8_t(label[0] + label[1] + label[2] + label[3] + label[4]);
        }
    };

    bool early_done(const Knn& knn) const {
        return early_limit_ >= 0 && knn.gate() <= uint64_t(early_limit_);
    }

    // Scan all blocks of a leaf node into the top-5.
    void scan_leaf(const Node& n, const int16_t* q, Knn& knn) const {
        const int blocks = (n.len + Block - 1) / Block;
        for (int b = 0; b < n.len; b += Block) {
            const int blk   = n.start + b / Block;
            const int valid = std::min(Block, n.len - b);
            const int16_t* vblock = vecs_   + size_t(blk) * Dims * Block;
            const uint8_t* lbls   = labels_ + size_t(blk) * Block;
            if (b / Block + 1 < blocks) {
                __builtin_prefetch(vblock + Dims * Block, 0, 1);
                __builtin_prefetch(lbls   + Block,        0, 1);
            }
            int64_t d[Block];
            distance_block8(vblock, q, d);
            for (int lane = 0; lane < valid; ++lane)
                knn.insert(uint64_t(d[lane]), lbls[lane]);
        }
    }

    // Exact branch-and-bound descent of one partition's tree.
    bool search_node(int root, uint64_t root_bound,
                      const int16_t* q, Knn& knn) const {
        if (root < 0 || root >= node_count_) return false;

        int      stack_node[128];
        uint64_t stack_bound[128];
        int      sp = 0;
        int      cur        = root;
        uint64_t cur_bound  = root_bound;

        for (;;) {
            if (cur_bound < knn.gate()) {
                const Node& n = nodes_[cur];
                if (n.left < 0) {
                    scan_leaf(n, q, knn);
                    if (early_done(knn)) return true;
                } else {
                    const Node& L = nodes_[n.left];
                    const Node& R = nodes_[n.right];
                    uint64_t lb = bbox_lb(q, L.min, L.max);
                    uint64_t rb = bbox_lb(q, R.min, R.max);
                    int      near = n.left,  far = n.right;
                    uint64_t nearb = lb,     farb = rb;
                    if (rb < lb) { near = n.right; far = n.left; nearb = rb; farb = lb; }
                    if (farb < knn.gate() && sp < 128) {
                        stack_node[sp]  = far;
                        stack_bound[sp] = farb;
                        ++sp;
                    }
                    cur       = near;
                    cur_bound = nearb;
                    continue;
                }
            }
            if (sp == 0) break;
            --sp;
            cur       = stack_node[sp];
            cur_bound = stack_bound[sp];
        }
        return early_done(knn);
    }

    void prefault() {
        long page = ::sysconf(_SC_PAGESIZE);
        if (page <= 0) page = 4096;
        volatile uint8_t sink = 0;
        for (size_t off = 0; off < size_; off += size_t(page)) sink ^= raw_[off];
        if (size_ > 0) sink ^= raw_[size_ - 1];
        warmup_ = sink;
    }

    size_t               size_   = 0;
    std::vector<uint8_t> buf_;
    bool                 mmap_   = false;
    uint8_t*             raw_    = nullptr;
    uint8_t              warmup_ = 0;

    const FileHeader* hdr_     = nullptr;
    const Partition*  parts_   = nullptr;
    const Node*       nodes_   = nullptr;
    const int16_t*    vecs_    = nullptr;
    const uint8_t*    labels_  = nullptr;
    int               part_count_ = 0;
    int               node_count_ = 0;
    int               part_by_key_[256];
    int64_t           early_limit_ = -1;    // -1 = disabled (pure exact search)
};

} // namespace rinha
