// Microbenchmark for the exact BVH search hot path.
// Usage: bench-search <index-path> [iters=200000]
//
// Times query() calls with realistic random int16_t queries and reports
// min/p50/p90/p99/p99.9/max in nanoseconds.

#include "index.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace rinha;

static inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <index> [iters]\n", argv[0]); return 1; }
    const char* path  = argv[1];
    int         iters = argc > 2 ? std::atoi(argv[2]) : 200000;

    BvhIndex idx(path);
    std::fprintf(stderr, "Loaded index: n=%u partitions=%d nodes=%d\n",
                 idx.num_vectors(), idx.num_partitions(), idx.num_nodes());

    // Generate queries by jittering vectors from the index range.
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_int_distribution<int> jitter(-200, 200);
    std::vector<std::array<int16_t, Dims>> queries(iters);
    for (int i = 0; i < iters; ++i)
        for (int d = 0; d < Dims; ++d)
            queries[i][d] = int16_t(jitter(rng));

    volatile uint8_t sink = 0;
    for (int i = 0; i < 5000; ++i) sink ^= idx.query(queries[i % iters].data());

    std::vector<uint64_t> t(iters);
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = now_ns();
        sink ^= idx.query(queries[i].data());
        t[i] = now_ns() - t0;
    }

    std::sort(t.begin(), t.end());
    auto pct = [&](double p) { return t[size_t((t.size() - 1) * p)]; };
    std::fprintf(stderr,
        "query   min=%6lu  p50=%6lu  p90=%6lu  p99=%6lu  p99.9=%6lu  max=%6lu  (ns)\n",
        t.front(), pct(0.50), pct(0.90), pct(0.99), pct(0.999), t.back());

    (void)sink;
    return 0;
}
