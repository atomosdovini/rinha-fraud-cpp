#include "index.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <zlib.h>

using Clock = std::chrono::steady_clock;

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

static void parse_refs(const std::string& path, std::vector<int16_t>& vectors, std::vector<uint8_t>& labels) {
    auto raw = read_gzip(path);
    vectors.reserve(size_t(3'000'000) * rinha::Dims);
    labels.reserve(3'000'000);

    const char* p = raw.data();
    while ((p = std::strstr(p, "\"vector\"")) != nullptr) {
        p = std::strchr(p, '[');
        if (!p) throw std::runtime_error("bad vector");
        ++p;
        for (int d = 0; d < rinha::Dims; ++d) {
            char* end = nullptr;
            float v = std::strtof(p, &end);
            if (end == p) throw std::runtime_error("bad float");
            vectors.push_back(rinha::qround(v));
            p = end;
            while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
        }
        const char* l = find_or_die(p, "\"label\"");
        const char* colon = std::strchr(l, ':');
        const char* quote = std::strchr(colon, '"');
        if (!quote) throw std::runtime_error("bad label");
        labels.push_back(quote[1] == 'f' ? 1 : 0);
        p = quote + 1;
    }

    if (labels.empty() || vectors.size() != labels.size() * rinha::Dims) {
        throw std::runtime_error("reference parse produced inconsistent data");
    }
}

static inline float dist_point_centroid(const int16_t* p, const std::array<float, rinha::Dims>& c) {
    float s = 0;
    for (int d = 0; d < rinha::Dims; ++d) {
        float diff = float(p[d]) - c[d];
        s += diff * diff;
    }
    return s;
}

static std::array<float, rinha::Dims> point_as_centroid(const std::vector<int16_t>& vectors, uint32_t id) {
    std::array<float, rinha::Dims> c{};
    const int16_t* p = vectors.data() + size_t(id) * rinha::Dims;
    for (int d = 0; d < rinha::Dims; ++d) c[d] = float(p[d]);
    return c;
}

static std::vector<uint32_t> make_sample(uint32_t n, uint32_t sample_size, uint64_t seed) {
    sample_size = std::min(sample_size, n);
    std::vector<uint32_t> sample(sample_size);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint32_t> pick(0, n - 1);
    for (uint32_t& v : sample) v = pick(rng);
    return sample;
}

static std::vector<std::array<float, rinha::Dims>> init_kmeans_pp(
    const std::vector<int16_t>& vectors,
    const std::vector<uint32_t>& sample,
    uint32_t k,
    uint64_t seed
) {
    std::vector<std::array<float, rinha::Dims>> centroids(k);
    std::vector<float> dmin(sample.size(), std::numeric_limits<float>::infinity());
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> first_pick(0, sample.size() - 1);
    centroids[0] = point_as_centroid(vectors, sample[first_pick(rng)]);

    for (uint32_t c = 1; c < k; ++c) {
        const auto& prev = centroids[c - 1];
        double sum = 0;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (size_t i = 0; i < sample.size(); ++i) {
            const int16_t* p = vectors.data() + size_t(sample[i]) * rinha::Dims;
            float d = dist_point_centroid(p, prev);
            if (d < dmin[i]) dmin[i] = d;
            sum += dmin[i];
        }
        if (sum <= 0) {
            centroids[c] = point_as_centroid(vectors, sample[first_pick(rng)]);
            continue;
        }
        std::uniform_real_distribution<double> dist(0, sum);
        double target = dist(rng);
        double acc = 0;
        size_t chosen = sample.size() - 1;
        for (size_t i = 0; i < sample.size(); ++i) {
            acc += dmin[i];
            if (acc >= target) {
                chosen = i;
                break;
            }
        }
        centroids[c] = point_as_centroid(vectors, sample[chosen]);
        if ((c & 255U) == 0) std::cerr << "  init centroid " << c << "/" << k << "\n";
    }
    return centroids;
}

static uint16_t nearest_centroid(const int16_t* p, const std::vector<std::array<float, rinha::Dims>>& centroids) {
    uint16_t best = 0;
    float best_d = dist_point_centroid(p, centroids[0]);
    for (uint32_t c = 1; c < centroids.size(); ++c) {
        float d = dist_point_centroid(p, centroids[c]);
        if (d < best_d) {
            best_d = d;
            best = static_cast<uint16_t>(c);
        }
    }
    return best;
}

static void train_sample_kmeans(
    const std::vector<int16_t>& vectors,
    const std::vector<uint32_t>& sample,
    std::vector<std::array<float, rinha::Dims>>& centroids,
    int iters
) {
    const uint32_t k = static_cast<uint32_t>(centroids.size());
    const int threads = std::max(1, omp_get_max_threads());
    std::vector<uint16_t> assign(sample.size());

    for (int iter = 0; iter < iters; ++iter) {
        uint64_t changed = 0;
        #pragma omp parallel for reduction(+:changed) schedule(static)
        for (size_t i = 0; i < sample.size(); ++i) {
            const int16_t* p = vectors.data() + size_t(sample[i]) * rinha::Dims;
            uint16_t c = nearest_centroid(p, centroids);
            changed += (c != assign[i]);
            assign[i] = c;
        }

        std::vector<double> sums(size_t(threads) * k * rinha::Dims);
        std::vector<uint32_t> counts(size_t(threads) * k);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double* tsums = sums.data() + size_t(tid) * k * rinha::Dims;
            uint32_t* tcounts = counts.data() + size_t(tid) * k;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < sample.size(); ++i) {
                uint16_t c = assign[i];
                const int16_t* p = vectors.data() + size_t(sample[i]) * rinha::Dims;
                ++tcounts[c];
                double* row = tsums + size_t(c) * rinha::Dims;
                for (int d = 0; d < rinha::Dims; ++d) row[d] += p[d];
            }
        }

        std::mt19937_64 rng(0xC0FFEE + iter);
        std::uniform_int_distribution<size_t> pick(0, sample.size() - 1);
        for (uint32_t c = 0; c < k; ++c) {
            uint64_t count = 0;
            std::array<double, rinha::Dims> sum{};
            for (int t = 0; t < threads; ++t) {
                count += counts[size_t(t) * k + c];
                const double* row = sums.data() + (size_t(t) * k + c) * rinha::Dims;
                for (int d = 0; d < rinha::Dims; ++d) sum[d] += row[d];
            }
            if (count == 0) {
                centroids[c] = point_as_centroid(vectors, sample[pick(rng)]);
            } else {
                double inv = 1.0 / double(count);
                for (int d = 0; d < rinha::Dims; ++d) centroids[c][d] = float(sum[d] * inv);
            }
        }
        std::cerr << "  sample kmeans iter " << (iter + 1) << "/" << iters
                  << " changed=" << changed << "\n";
    }
}

static std::vector<uint16_t> assign_all(
    const std::vector<int16_t>& vectors,
    const std::vector<std::array<float, rinha::Dims>>& centroids,
    std::vector<uint32_t>& counts
) {
    uint32_t n = static_cast<uint32_t>(vectors.size() / rinha::Dims);
    uint32_t k = static_cast<uint32_t>(centroids.size());
    std::vector<uint16_t> assign(n);
    int threads = std::max(1, omp_get_max_threads());
    std::vector<uint32_t> local_counts(size_t(threads) * k);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t* lc = local_counts.data() + size_t(tid) * k;
        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < n; ++i) {
            const int16_t* p = vectors.data() + size_t(i) * rinha::Dims;
            uint16_t c = nearest_centroid(p, centroids);
            assign[i] = c;
            ++lc[c];
        }
    }

    counts.assign(k, 0);
    for (int t = 0; t < threads; ++t) {
        for (uint32_t c = 0; c < k; ++c) counts[c] += local_counts[size_t(t) * k + c];
    }
    return assign;
}

static void write_index(
    const std::string& out_path,
    const std::vector<int16_t>& vectors,
    const std::vector<uint8_t>& labels,
    const std::vector<std::array<float, rinha::Dims>>& centroids,
    const std::vector<uint16_t>& assign,
    const std::vector<uint32_t>& counts
) {
    uint32_t n = static_cast<uint32_t>(labels.size());
    uint32_t k = static_cast<uint32_t>(centroids.size());

    std::vector<uint32_t> starts(k + 1);
    for (uint32_t c = 0; c < k; ++c) starts[c + 1] = starts[c] + counts[c];
    std::vector<uint32_t> cursor = starts;
    std::vector<uint32_t> order(n);
    for (uint32_t i = 0; i < n; ++i) order[cursor[assign[i]]++] = i;

    std::vector<uint32_t> block_offsets(k + 1);
    for (uint32_t c = 0; c < k; ++c) {
        block_offsets[c + 1] = block_offsets[c] + (counts[c] + rinha::Block - 1) / rinha::Block;
    }
    uint32_t total_blocks = block_offsets[k];

    std::vector<int16_t> qcentroids(size_t(k) * rinha::Dims);
    for (uint32_t c = 0; c < k; ++c) {
        for (int d = 0; d < rinha::Dims; ++d) {
            qcentroids[size_t(c) * rinha::Dims + d] = static_cast<int16_t>(__builtin_llround(centroids[c][d]));
        }
    }

    std::vector<int16_t> bmin(size_t(k) * rinha::Dims, std::numeric_limits<int16_t>::max());
    std::vector<int16_t> bmax(size_t(k) * rinha::Dims, std::numeric_limits<int16_t>::min());
    std::vector<uint8_t> out_labels(size_t(total_blocks) * rinha::Block);
    std::vector<int16_t> blocks(size_t(total_blocks) * rinha::Dims * rinha::Block);

    for (uint32_t c = 0; c < k; ++c) {
        if (counts[c] == 0) {
            for (int d = 0; d < rinha::Dims; ++d) {
                bmin[size_t(c) * rinha::Dims + d] = 0;
                bmax[size_t(c) * rinha::Dims + d] = 0;
            }
            continue;
        }
        for (uint32_t pos = 0; pos < counts[c]; ++pos) {
            uint32_t orig = order[starts[c] + pos];
            uint32_t block = block_offsets[c] + pos / rinha::Block;
            uint32_t lane = pos % rinha::Block;
            out_labels[size_t(block) * rinha::Block + lane] = labels[orig];
            const int16_t* src = vectors.data() + size_t(orig) * rinha::Dims;
            int16_t* dst = blocks.data() + size_t(block) * rinha::Dims * rinha::Block;
            for (int d = 0; d < rinha::Dims; ++d) {
                int16_t v = src[d];
                dst[d * rinha::Block + lane] = v;
                auto idx = size_t(c) * rinha::Dims + d;
                bmin[idx] = std::min(bmin[idx], v);
                bmax[idx] = std::max(bmax[idx], v);
            }
        }
    }

    rinha::FileHeader h{};
    h.magic = rinha::kMagic;
    h.version = rinha::kVer;
    h.n = n;
    h.k = k;
    h.total_blocks = total_blocks;
    h.block_size = rinha::Block;
    h.dims = rinha::Dims;

    rinha::SectionOffsets layout = rinha::compute_sections(k, total_blocks);
    std::vector<char> zero(64);
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + out_path);

    auto write_at = [&](size_t off, const void* data, size_t len) {
        out.seekp(static_cast<std::streamoff>(off));
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(len));
        if (!out) throw std::runtime_error("write failed");
    };

    write_at(0, &h, sizeof(h));
    write_at(layout.centroids, qcentroids.data(), qcentroids.size() * sizeof(int16_t));
    write_at(layout.bbox_min, bmin.data(), bmin.size() * sizeof(int16_t));
    write_at(layout.bbox_max, bmax.data(), bmax.size() * sizeof(int16_t));
    write_at(layout.offsets, block_offsets.data(), block_offsets.size() * sizeof(uint32_t));
    write_at(layout.counts, counts.data(), counts.size() * sizeof(uint32_t));
    write_at(layout.labels, out_labels.data(), out_labels.size());
    write_at(layout.blocks, blocks.data(), blocks.size() * sizeof(int16_t));
    out.seekp(static_cast<std::streamoff>(layout.total - 1));
    out.put('\0');
    std::cerr << "index written: " << out_path << " (" << (layout.total / (1024 * 1024)) << " MB)\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: build-index references.json.gz index.bin [k=4096] [sample=50000] [iters=10]\n";
        return 2;
    }
    std::string refs = argv[1];
    std::string out = argv[2];
    uint32_t k = argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 4096;
    uint32_t sample_size = argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 50000;
    int iters = argc > 5 ? std::stoi(argv[5]) : 10;

    auto t0 = Clock::now();
    std::vector<int16_t> vectors;
    std::vector<uint8_t> labels;
    parse_refs(refs, vectors, labels);
    uint32_t n = static_cast<uint32_t>(labels.size());
    std::cerr << "parsed " << n << " refs in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count()
              << "ms\n";

    auto sample = make_sample(n, sample_size, 42);
    auto centroids = init_kmeans_pp(vectors, sample, k, 42);
    train_sample_kmeans(vectors, sample, centroids, iters);

    std::vector<uint32_t> counts;
    auto t_assign = Clock::now();
    auto assign = assign_all(vectors, centroids, counts);
    std::cerr << "assigned all refs in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t_assign).count()
              << "ms\n";

    auto [min_it, max_it] = std::minmax_element(counts.begin(), counts.end());
    uint64_t sum = std::accumulate(counts.begin(), counts.end(), uint64_t(0));
    std::cerr << "cluster sizes min=" << *min_it << " max=" << *max_it
              << " mean=" << (sum / counts.size()) << "\n";

    write_index(out, vectors, labels, centroids, assign, counts);
    std::cerr << "done in "
              << std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - t0).count()
              << "s\n";
    return 0;
}
