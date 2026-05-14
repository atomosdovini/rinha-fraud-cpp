#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace rinha {

[[noreturn]] inline void die() { std::_Exit(1); }

// ── Constants ─────────────────────────────────────────────────────────────────

constexpr int     Dims    = 14;
constexpr int     Block   = 8;
constexpr uint64_t kMagic = 0x3149564936324852ULL;
constexpr uint32_t kVer   = 1;

// ── File layout ───────────────────────────────────────────────────────────────

struct FileHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t n;
    uint32_t k;
    uint32_t total_blocks;
    uint32_t block_size;
    uint32_t dims;
    uint32_t reserved[8];
};
static_assert(sizeof(FileHeader) == 64);

struct SectionOffsets {
    size_t centroids;
    size_t bbox_min;
    size_t bbox_max;
    size_t offsets;
    size_t counts;
    size_t labels;
    size_t blocks;
    size_t total;
};

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

inline SectionOffsets compute_sections(uint32_t k, uint32_t total_blocks) {
    SectionOffsets s{};
    size_t p = sizeof(FileHeader);
    s.centroids = p; p += size_t(k) * Dims * sizeof(int16_t);
    s.bbox_min  = p; p += size_t(k) * Dims * sizeof(int16_t);
    s.bbox_max  = p; p += size_t(k) * Dims * sizeof(int16_t);
    p = align_up(p, alignof(uint32_t));
    s.offsets   = p; p += size_t(k + 1) * sizeof(uint32_t);
    s.counts    = p; p += size_t(k) * sizeof(uint32_t);
    s.labels    = p; p += size_t(total_blocks) * Block;
    p = align_up(p, alignof(int16_t));
    s.blocks    = p; p += size_t(total_blocks) * Dims * Block * sizeof(int16_t);
    s.total     = p;
    return s;
}

// ── Quantisation helpers ──────────────────────────────────────────────────────

inline int16_t qround(double v) {
    v = v < -1.0 ? -1.0 : v > 1.0 ? 1.0 : v;
    return static_cast<int16_t>(__builtin_llround(v * 10000.0));
}

inline int16_t qclamp01(double v) {
    v = v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v;
    return static_cast<int16_t>(__builtin_llround(v * 10000.0));
}

// ── Distance primitives ───────────────────────────────────────────────────────

inline uint64_t l2sq_q16(const int16_t* __restrict__ a, const int16_t* __restrict__ b) {
    __m128i va   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    __m128i vb   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    __m128i diff = _mm_sub_epi16(va, vb);
    __m128i sq   = _mm_madd_epi16(diff, diff);
    sq = _mm_hadd_epi32(sq, sq);
    sq = _mm_hadd_epi32(sq, sq);
    uint64_t acc = uint32_t(_mm_cvtsi128_si32(sq));
    for (int d = 8; d < Dims; ++d) {
        int64_t e = int64_t(a[d]) - int64_t(b[d]);
        acc += uint64_t(e * e);
    }
    return acc;
}

inline uint64_t centroid_dist(const int16_t* q, const int16_t* centroids, uint32_t c) {
    return l2sq_q16(q, centroids + size_t(c) * Dims);
}

inline uint64_t bbox_lb(const int16_t* q,
                        const int16_t* bmin, const int16_t* bmax, uint32_t c) {
    const int16_t* lo = bmin + size_t(c) * Dims;
    const int16_t* hi = bmax + size_t(c) * Dims;
    uint64_t acc = 0;
    for (int d = 0; d < Dims; ++d) {
        int64_t e = 0;
        if      (q[d] < lo[d]) e = int64_t(lo[d]) - q[d];
        else if (q[d] > hi[d]) e = int64_t(q[d])  - hi[d];
        acc += uint64_t(e * e);
    }
    return acc;
}

// ── Search diagnostics (optional, passed as pointer) ─────────────────────────

struct QueryTrace {
    uint32_t initial_scanned  = 0;
    uint32_t initial_pruned   = 0;
    uint32_t repair_scanned   = 0;
    uint32_t repair_pruned    = 0;
    bool     repair_triggered = false;
    uint8_t  initial_fraud    = 0;
    uint8_t  final_fraud      = 0;
    uint64_t initial_worst    = 0;
    uint64_t final_worst      = 0;
};

// ── IvfIndex ──────────────────────────────────────────────────────────────────

class IvfIndex {
public:
    explicit IvfIndex(const std::string& path) {
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
            ::madvise(raw_, size_, MADV_WILLNEED | MADV_SEQUENTIAL);
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

        sec_ = compute_sections(hdr_->k, hdr_->total_blocks);
        if (sec_.total > size_) die();

        centroids_ = ptr<int16_t>(sec_.centroids);
        bmin_      = ptr<int16_t>(sec_.bbox_min);
        bmax_      = ptr<int16_t>(sec_.bbox_max);
        offsets_   = ptr<uint32_t>(sec_.offsets);
        counts_    = ptr<uint32_t>(sec_.counts);
        labels_    = raw_ + sec_.labels;
        vecs_      = ptr<int16_t>(sec_.blocks);

        transpose_centroids();
        prefault();
    }

    IvfIndex(const IvfIndex&)            = delete;
    IvfIndex& operator=(const IvfIndex&) = delete;

    ~IvfIndex() {
        if (mmap_ && raw_ && raw_ != MAP_FAILED) ::munmap(raw_, size_);
        raw_ = nullptr;
    }

    uint32_t num_clusters() const { return hdr_->k; }
    uint32_t num_vectors()  const { return hdr_->n; }

    // Returns number of fraud labels among the 5 nearest neighbours.
    uint8_t query(const int16_t q[Dims], int nprobe,
                  int repair_min = 2, int repair_max = 3,
                  QueryTrace* trace = nullptr) const {
        const uint32_t K = num_clusters();
        nprobe = std::clamp(nprobe, 1, int(std::min<uint32_t>(64, K)));

        if (repair_min > repair_max && !csoa_.empty()) {
            if (nprobe == 1 && K >= 1) return query_top1_avx2(q, trace);
            if (nprobe == 2 && K >= 2) return query_top2_avx2(q, trace);
            return query_topn_avx2(q, nprobe, trace);
        }

        std::array<uint32_t, 64> bc{};
        std::array<uint64_t, 64> bd{};
        int used = 0; uint64_t worst = 0; int wi = 0;
        rank_centroids(q, nprobe, K, bc, bd, used, worst, wi);

        // sort selected clusters by distance (insertion sort, nprobe ≤ 64)
        for (int i = 1; i < used; ++i) {
            uint32_t tc = bc[i]; uint64_t td = bd[i]; int j = i - 1;
            while (j >= 0 && bd[j] > td) { bd[j+1] = bd[j]; bc[j+1] = bc[j]; --j; }
            bd[j+1] = td; bc[j+1] = tc;
        }

        KnnHeap heap;
        std::array<uint64_t, 128> visited{};
        const uint32_t vwords = (K + 63) / 64;
        if (vwords > visited.size()) die();

        for (int i = 0; i < used; ++i) {
            uint32_t c = bc[i];
            if (i > 0 && bbox_lb(q, bmin_, bmax_, c) >= heap.gate()) {
                visited[c >> 6] |= uint64_t(1) << (c & 63);
                if (trace) ++trace->initial_pruned;
                continue;
            }
            probe_cluster(c, q, heap);
            if (trace) ++trace->initial_scanned;
            visited[c >> 6] |= uint64_t(1) << (c & 63);
        }

        uint8_t fraud = heap.fraud_count();
        if (trace) { trace->initial_fraud = fraud; trace->initial_worst = heap.gate(); }

        if (fraud >= repair_min && fraud <= repair_max) {
            if (trace) trace->repair_triggered = true;
            for (uint32_t c = 0; c < K; ++c) {
                if (visited[c >> 6] & (uint64_t(1) << (c & 63))) continue;
                if (bbox_lb(q, bmin_, bmax_, c) >= heap.gate()) {
                    if (trace) ++trace->repair_pruned;
                    continue;
                }
                probe_cluster(c, q, heap);
                if (trace) ++trace->repair_scanned;
            }
            fraud = heap.fraud_count();
        }
        if (trace) { trace->final_fraud = fraud; trace->final_worst = heap.gate(); }
        return fraud;
    }

private:
    // Fixed-size max-heap tracking 5 nearest neighbours.
    struct KnnHeap {
        std::array<uint64_t, 5> dist;
        std::array<uint8_t,  5> label;
        int gate_idx = 0;

        KnnHeap() { dist.fill(UINT64_MAX); label.fill(0); }

        uint64_t gate() const { return dist[gate_idx]; }

        void insert(uint64_t d, uint8_t l) {
            if (d >= dist[gate_idx]) return;
            dist[gate_idx]  = d;
            label[gate_idx] = l;
            gate_idx = 0;
            for (int i = 1; i < 5; ++i)
                if (dist[i] > dist[gate_idx]) gate_idx = i;
        }

        uint8_t fraud_count() const {
            return uint8_t(label[0] + label[1] + label[2] + label[3] + label[4]);
        }
    };

    // ── AVX2 centroid ranking ─────────────────────────────────────────────────

    void rank_centroids(const int16_t q[Dims], int nprobe, uint32_t K,
                        std::array<uint32_t,64>& bc, std::array<uint64_t,64>& bd,
                        int& used, uint64_t& worst, int& wi) const {
        alignas(32) uint32_t dbuf[8];
        uint32_t c = 0;
        for (; c + 8 <= K; c += 8) {
            __m256i acc = avx2_dist8(q, c);
            _mm256_store_si256(reinterpret_cast<__m256i*>(dbuf), acc);
            for (uint32_t lane = 0; lane < 8; ++lane) {
                uint64_t d = dbuf[lane];
                uint32_t cl = c + lane;
                if (used < nprobe) {
                    bc[used] = cl; bd[used] = d;
                    if (used == 0 || d > worst) { worst = d; wi = used; }
                    ++used;
                } else if (d < worst) {
                    bc[wi] = cl; bd[wi] = d;
                    worst = bd[0]; wi = 0;
                    for (int i = 1; i < used; ++i)
                        if (bd[i] > worst) { worst = bd[i]; wi = i; }
                }
            }
        }
        for (; c < K; ++c) {
            uint64_t d = centroid_dist(q, centroids_, c);
            if (used < nprobe) {
                bc[used] = c; bd[used] = d;
                if (used == 0 || d > worst) { worst = d; wi = used; }
                ++used;
            } else if (d < worst) {
                bc[wi] = c; bd[wi] = d;
                worst = bd[0]; wi = 0;
                for (int i = 1; i < used; ++i)
                    if (bd[i] > worst) { worst = bd[i]; wi = i; }
            }
        }
    }

    // Compute squared L2 distances from q to 8 consecutive centroids via AVX2.
    __m256i avx2_dist8(const int16_t q[Dims], uint32_t c_base) const {
        const uint32_t K = num_clusters();
        __m256i acc = _mm256_setzero_si256();
        for (int d = 0; d < Dims; d += 2) {
            const int16_t* r0 = csoa_.data() + size_t(d)     * K + c_base;
            const int16_t* r1 = csoa_.data() + size_t(d + 1) * K + c_base;
            __m128i ref0  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(r0));
            __m128i ref1  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(r1));
            __m128i diff0 = _mm_sub_epi16(_mm_set1_epi16(q[d]),     ref0);
            __m128i diff1 = _mm_sub_epi16(_mm_set1_epi16(q[d + 1]), ref1);
            __m128i lo    = _mm_unpacklo_epi16(diff0, diff1);
            __m128i hi    = _mm_unpackhi_epi16(diff0, diff1);
            __m256i pairs = _mm256_set_m128i(hi, lo);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, pairs));
        }
        return acc;
    }

    // ── Fast paths: no repair, fixed nprobe ──────────────────────────────────

    uint8_t query_top1_avx2(const int16_t q[Dims], QueryTrace* trace) const {
        const uint32_t K = num_clusters();
        const __m256i lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i best_d = _mm256_set1_epi32(INT32_MAX);
        __m256i best_i = _mm256_setzero_si256();
        uint32_t c = 0;
        for (; c + 8 <= K; c += 8) {
            __m256i acc  = avx2_dist8(q, c);
            __m256i idx  = _mm256_add_epi32(_mm256_set1_epi32(int(c)), lane_ids);
            __m256i mask = _mm256_cmpgt_epi32(best_d, acc);
            best_d = _mm256_blendv_epi8(best_d, acc, mask);
            best_i = _mm256_blendv_epi8(best_i, idx, mask);
        }
        alignas(32) uint32_t bd[8], bi[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(bd), best_d);
        _mm256_store_si256(reinterpret_cast<__m256i*>(bi), best_i);
        uint32_t best = 0; uint32_t best_dist = UINT32_MAX;
        for (uint32_t lane = 0; lane < 8; ++lane)
            if (bd[lane] < best_dist) { best_dist = bd[lane]; best = bi[lane]; }
        for (; c < K; ++c) {
            uint32_t d = static_cast<uint32_t>(centroid_dist(q, centroids_, c));
            if (d < best_dist) { best_dist = d; best = c; }
        }

        KnnHeap heap;
        probe_cluster(best, q, heap);
        uint8_t fraud = heap.fraud_count();
        if (trace) {
            trace->initial_scanned = 1; trace->initial_fraud = fraud; trace->final_fraud = fraud;
            trace->initial_worst = heap.gate(); trace->final_worst = heap.gate();
        }
        return fraud;
    }

    uint8_t query_top2_avx2(const int16_t q[Dims], QueryTrace* trace) const {
        const uint32_t K = num_clusters();
        uint32_t c0 = 0, c1 = 1; uint32_t d0 = UINT32_MAX, d1 = UINT32_MAX;
        alignas(32) uint32_t dbuf[8];
        uint32_t c = 0;
        for (; c + 8 <= K; c += 8) {
            __m256i acc = avx2_dist8(q, c);
            _mm256_store_si256(reinterpret_cast<__m256i*>(dbuf), acc);
            for (uint32_t lane = 0; lane < 8; ++lane) {
                uint32_t d = dbuf[lane]; uint32_t cl = c + lane;
                if      (d < d0) { d1 = d0; c1 = c0; d0 = d; c0 = cl; }
                else if (d < d1) { d1 = d;  c1 = cl; }
            }
        }
        for (; c < K; ++c) {
            uint32_t d = static_cast<uint32_t>(centroid_dist(q, centroids_, c));
            if      (d < d0) { d1 = d0; c1 = c0; d0 = d; c0 = c; }
            else if (d < d1) { d1 = d;  c1 = c; }
        }

        KnnHeap heap;
        probe_cluster(c0, q, heap);
        if (trace) ++trace->initial_scanned;
        if (bbox_lb(q, bmin_, bmax_, c1) >= heap.gate()) {
            if (trace) ++trace->initial_pruned;
        } else {
            probe_cluster(c1, q, heap);
            if (trace) ++trace->initial_scanned;
        }
        uint8_t fraud = heap.fraud_count();
        if (trace) {
            trace->initial_fraud = fraud; trace->final_fraud = fraud;
            trace->initial_worst = heap.gate(); trace->final_worst = heap.gate();
        }
        return fraud;
    }

    uint8_t query_topn_avx2(const int16_t q[Dims], int nprobe, QueryTrace* trace) const {
        const uint32_t K = num_clusters();
        std::array<uint32_t, 64> bc{}; std::array<uint64_t, 64> bd{};
        int used = 0; uint64_t worst = 0; int wi = 0;
        rank_centroids(q, nprobe, K, bc, bd, used, worst, wi);

        for (int i = 1; i < used; ++i) {
            uint32_t tc = bc[i]; uint64_t td = bd[i]; int j = i - 1;
            while (j >= 0 && bd[j] > td) { bd[j+1] = bd[j]; bc[j+1] = bc[j]; --j; }
            bd[j+1] = td; bc[j+1] = tc;
        }

        KnnHeap heap;
        for (int i = 0; i < used; ++i) {
            uint32_t c = bc[i];
            if (i > 0 && bbox_lb(q, bmin_, bmax_, c) >= heap.gate()) {
                if (trace) ++trace->initial_pruned;
                continue;
            }
            if (i + 1 < used && counts_[bc[i+1]] > 0)
                __builtin_prefetch(vecs_ + size_t(offsets_[bc[i+1]]) * Dims * Block, 0, 1);
            probe_cluster(c, q, heap);
            if (trace) ++trace->initial_scanned;
        }
        uint8_t fraud = heap.fraud_count();
        if (trace) {
            trace->initial_fraud = fraud; trace->final_fraud = fraud;
            trace->initial_worst = heap.gate(); trace->final_worst = heap.gate();
        }
        return fraud;
    }

    // ── Cluster / block scanning ──────────────────────────────────────────────

    void probe_cluster(uint32_t c, const int16_t q[Dims], KnnHeap& heap) const {
        const uint32_t blk_start = offsets_[c];
        const uint32_t cnt       = counts_[c];
        const uint32_t nblocks   = (cnt + Block - 1) / Block;
        for (uint32_t b = 0; b < nblocks; ++b) {
            uint32_t blk_id = blk_start + b;
            uint32_t valid  = std::min<uint32_t>(Block, cnt - b * Block);
            const int16_t* vblock = vecs_   + size_t(blk_id) * Dims * Block;
            const uint8_t* lbls   = labels_ + size_t(blk_id) * Block;
            if (b + 1 < nblocks) {
                __builtin_prefetch(vecs_   + size_t(blk_id + 1) * Dims * Block, 0, 1);
                __builtin_prefetch(labels_ + size_t(blk_id + 1) * Block,         0, 1);
            }
            if (valid == Block) { score_block8(vblock, lbls, q, heap); continue; }
            for (uint32_t lane = 0; lane < valid; ++lane) {
                uint64_t acc = 0, gate = heap.gate();
                for (int d = 0; d < Dims; ++d) {
                    int64_t e = int64_t(q[d]) - int64_t(vblock[d * Block + lane]);
                    acc += uint64_t(e * e);
                    if (acc >= gate) break;
                }
                heap.insert(acc, lbls[lane]);
            }
        }
    }

    static void score_block8(const int16_t* vblock, const uint8_t* lbls,
                              const int16_t q[Dims], KnnHeap& heap) {
        __m256i acc = _mm256_setzero_si256();
        for (int d = 0; d < Dims; d += 2) {
            __m128i ref0  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vblock + d * Block));
            __m128i ref1  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vblock + (d+1) * Block));
            __m128i diff0 = _mm_sub_epi16(_mm_set1_epi16(q[d]),     ref0);
            __m128i diff1 = _mm_sub_epi16(_mm_set1_epi16(q[d + 1]), ref1);
            __m128i lo    = _mm_unpacklo_epi16(diff0, diff1);
            __m128i hi    = _mm_unpackhi_epi16(diff0, diff1);
            __m256i pairs = _mm256_set_m128i(hi, lo);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, pairs));
        }
        alignas(32) uint32_t dist[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(dist), acc);
        uint64_t gate = heap.gate();
        for (uint32_t lane = 0; lane < 8; ++lane) {
            if (dist[lane] < gate) {
                heap.insert(dist[lane], lbls[lane]);
                gate = heap.gate();
            }
        }
    }

    // ── Initialisation helpers ────────────────────────────────────────────────

    void transpose_centroids() {
        const uint32_t K = num_clusters();
        csoa_.resize(size_t(K) * Dims);
        for (uint32_t c = 0; c < K; ++c) {
            const int16_t* src = centroids_ + size_t(c) * Dims;
            for (int d = 0; d < Dims; ++d)
                csoa_[size_t(d) * K + c] = src[d];
        }
    }

    void prefault() {
        long page = ::sysconf(_SC_PAGESIZE);
        if (page <= 0) page = 4096;
        volatile uint8_t sink = 0;
        for (size_t off = 0; off < size_; off += size_t(page)) sink ^= raw_[off];
        if (size_ > 0) sink ^= raw_[size_ - 1];
        warmup_ = sink;
    }

    template<typename T>
    const T* ptr(size_t off) const { return reinterpret_cast<const T*>(raw_ + off); }

    // ── Data members ─────────────────────────────────────────────────────────

    size_t              size_   = 0;
    std::vector<uint8_t> buf_;
    bool                mmap_   = false;
    uint8_t*            raw_    = nullptr;
    uint8_t             warmup_ = 0;

    const FileHeader*  hdr_       = nullptr;
    SectionOffsets     sec_{};
    const int16_t*     centroids_ = nullptr;
    const int16_t*     bmin_      = nullptr;
    const int16_t*     bmax_      = nullptr;
    const uint32_t*    offsets_   = nullptr;
    const uint32_t*    counts_    = nullptr;
    const uint8_t*     labels_    = nullptr;
    const int16_t*     vecs_      = nullptr;
    std::vector<int16_t> csoa_;
};

} // namespace rinha
