# rinha-fraud-cpp

A high-performance C++ fraud-scoring service built for the **Rinha de Backend 2026** challenge. Given a transaction payload over HTTP, the service returns a fraud score derived from a nearest-neighbour lookup against ~millions of reference transactions, all within tight CPU/memory budgets (≈1 CPU and 350 MB of RAM split across the whole stack).

The project is written from scratch in C++20 with no web framework, no JSON library, and no vector-search library — every layer is purpose-built for the workload.

## Stack

- **Language:** C++20, compiled with `-O3 -march=haswell -mavx2 -mfma -flto`
- **I/O:** `io_uring` on the API workers, raw epoll/sendfile-free reactor on the load balancer
- **Concurrency:** single-threaded reactors per process, multiple processes behind a custom LB
- **Index:** custom IVF (Inverted-File) with int16 quantisation and AVX2 distance kernels
- **Transport:** HTTP/1.1 over Unix domain sockets between LB ↔ API
- **Runtime:** Debian-slim container, statically linked libstdc++/libgcc

## Architecture

```
                ┌────────────────────────┐
   client ──▶   │  lb  (port 9999)       │   round-robin over UDS
                │  src/lb.cpp            │
                └──────────┬─────────────┘
                           │  /sockets/api1.sock
                           │  /sockets/api2.sock
                ┌──────────▼─────────────┐
                │  api1, api2            │   io_uring HTTP server
                │  src/server.cpp        │   parses JSON → feature vec
                │                        │   IVF search → score
                └──────────┬─────────────┘
                           │  mmap, read-only
                ┌──────────▼─────────────┐
                │  /index/index.bin      │   prebuilt IVF index
                │  src/build_index.cpp   │   (baked into image)
                └────────────────────────┘
```

Three containers run side-by-side:

1. **`lb`** — a tiny TCP-on-:9999 → UDS load balancer (`src/lb.cpp`). Round-robins requests across the API replicas, no buffering, no parsing.
2. **`api1` / `api2`** — identical replicas of the scoring service (`src/server.cpp`). Each is single-worker, single-threaded, and uses `io_uring` for accept / recv / send. The IVF index is `mmap`'d read-only so both replicas share the same physical pages.
3. **Index build** — happens at image-build time inside the Dockerfile. `build-index` downloads `references.json.gz` from the challenge repo, parses each reference transaction, projects it to a 14-dim int16 feature vector, runs k-means to produce `k=65536` clusters in blocks of 8 vectors, and writes a single mmap-friendly `index.bin`.

## Source layout

| File | Role |
|------|------|
| `src/index.hpp` | IVF file layout, quantisation helpers, AVX2 distance kernels, search routine |
| `src/tx.hpp` | Zero-allocation JSON parser specialised for the fraud-score request body; emits a 14-dim feature vector in a single forward pass |
| `src/build_index.cpp` | Offline index builder: gzip-stream JSON parse → k-means → packed binary index |
| `src/server.cpp` | `io_uring` HTTP/1.1 server, request router, scoring pipeline |
| `src/lb.cpp` | Minimal TCP→UDS round-robin load balancer |
| `src/bench_search.cpp` | Local micro-benchmark for the search kernel |

## Hot path

For each `POST /fraud-score` request, a worker:

1. Reads the body from the UDS via `io_uring`.
2. Parses the JSON in `tx.hpp` directly into a `int16[14]` feature vector — no tokenisation, no allocations, single forward pass over keys.
3. Probes the IVF index in `index.hpp`: picks the `NPROBE` closest centroids (with `FAST_NPROBE` / `ADAPTIVE_MIN..MAX` tuning), then scans the 8-wide quantised blocks of each cluster with an AVX2 L2 kernel.
4. Reduces the top-k distances to a final fraud score and writes the HTTP response back through `io_uring`.

## Tuning knobs

Set per-replica via env vars in `docker-compose.yml`:

- `NPROBE` / `FAST_NPROBE` — number of IVF clusters to scan.
- `ADAPTIVE_MIN` / `ADAPTIVE_MAX` — bounds for adaptive probing.
- `EXTREME{0,1,5}_WORST_THRESHOLD` — bail-out thresholds for outlier inputs.
- `REPAIR_MIN` / `REPAIR_MAX` — recall-vs-latency repair window.
- `INDEX_MMAP=1` — map the index file shared between replicas.
- `MALLOC_ARENA_MAX=1` — keep glibc malloc arenas tight to fit the memory cap.

## Build & run

### Locally (host toolchain)

```bash
make build      # compile build-index + server with the host g++
make index      # download references and build a small dev index into build/index/
make run        # run a single server on /rinha-sockets/api.sock
```

### Docker (production layout)

```bash
make docker-up      # build image + start lb, api1, api2
make docker-down    # stop and remove containers + sockets volume
```

The compose file enforces the challenge's resource limits (`lb`: 0.16 CPU / 30 MB; each api: 0.42 CPU / 160 MB).

### Benchmark

The official k6 harness lives outside this repo:

```bash
make perf       # runs /tmp/rinha-de-backend-2026/test/test.js and prints results.json
```

## Resource budget

| Component | CPU | RAM |
|-----------|-----|-----|
| `lb` | 0.16 | 30 MB |
| `api1` | 0.42 | 160 MB |
| `api2` | 0.42 | 160 MB |
| **Total** | **1.0** | **350 MB** |

The `index.bin` file is mapped read-only and shared between `api1` and `api2`, so the index cost is paid once at the page-cache level rather than per replica.

## License & author

Vinicius Felpe — <https://github.com/atomosdovini/rinha-fraud-cpp>
