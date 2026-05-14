// Microbenchmark for the search hot path.
// Usage: bench-search <index-path> [iters=200000]
//
// Times do_search-equivalent calls with realistic random int16_t queries.
// Reports min/p50/p90/p99/p99.9/max in nanoseconds and CPU cycles.

#include "index.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <x86intrin.h>

using namespace rinha;

static inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <index> [iters]\n", argv[0]); return 1; }
    const char* path  = argv[1];
    int         iters = argc > 2 ? std::atoi(argv[2]) : 200000;

    IvfIndex idx(path);
    std::fprintf(stderr, "Loaded index: k=%u n=%u\n", idx.num_clusters(), idx.num_vectors());

    // Generate queries by jittering vectors sampled from the index range.
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_int_distribution<int> jitter(-200, 200);
    std::vector<std::array<int16_t, Dims>> queries(iters);
    for (int i = 0; i < iters; ++i)
        for (int d = 0; d < Dims; ++d)
            queries[i][d] = int16_t(jitter(rng));

    // Warm-up to JIT branch predictors / fill caches.
    volatile uint8_t sink = 0;
    for (int i = 0; i < 5000; ++i) sink ^= idx.query(queries[i % iters].data(), 1, 99, 0);

    // Scenario A: fast path only (NPROBE=1, repair_min > repair_max → fast).
    std::vector<uint64_t> t_fast(iters);
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = now_ns();
        sink ^= idx.query(queries[i].data(), 1, 99, 0);
        t_fast[i] = now_ns() - t0;
    }

    // Scenario B: fast path nprobe=2.
    std::vector<uint64_t> t_fast2(iters);
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = now_ns();
        sink ^= idx.query(queries[i].data(), 2, 99, 0);
        t_fast2[i] = now_ns() - t0;
    }

    // Scenario C: full nprobe=20 with repair window.
    std::vector<uint64_t> t_full(iters);
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = now_ns();
        sink ^= idx.query(queries[i].data(), 20, 2, 4);
        t_full[i] = now_ns() - t0;
    }

    auto report = [&](const char* name, std::vector<uint64_t>& v) {
        std::sort(v.begin(), v.end());
        auto pct = [&](double p) { return v[size_t((v.size() - 1) * p)]; };
        std::fprintf(stderr,
            "%-12s  min=%6lu  p50=%6lu  p90=%6lu  p99=%6lu  p99.9=%6lu  max=%6lu  (ns)\n",
            name, v.front(), pct(0.50), pct(0.90), pct(0.99), pct(0.999), v.back());
    };
    report("nprobe=1",  t_fast);
    report("nprobe=2",  t_fast2);
    report("nprobe=20", t_full);

    (void)sink;
    return 0;
}
