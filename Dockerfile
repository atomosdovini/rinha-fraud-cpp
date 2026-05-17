# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 alpine:3.20 AS references
WORKDIR /work
RUN apk add --no-cache curl
RUN curl -fsSL -o references.json.gz \
    https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz

FROM --platform=linux/amd64 gcc:16 AS builder
WORKDIR /src
RUN apt-get update \
    && apt-get install -y --no-install-recommends zlib1g-dev liburing-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY src ./src
RUN mkdir -p /out /index \
    && g++ -O3 -DNDEBUG -std=c++20 -march=haswell -mtune=haswell -mavx2 -mfma -flto \
       src/build_index.cpp -lz -o /out/build-index \
    && g++ -O3 -DNDEBUG -std=c++20 -march=haswell -mtune=haswell -mavx2 -mfma -flto \
       -fno-exceptions -pthread -static-libstdc++ -static-libgcc \
       src/server.cpp -o /out/server \
    && g++ -O3 -DNDEBUG -std=c++20 -march=haswell -mtune=haswell -flto \
       -fno-exceptions -pthread -static-libstdc++ -static-libgcc \
       src/lb.cpp -o /out/lb
COPY --from=references /work/references.json.gz /tmp/references.json.gz
RUN /out/build-index /tmp/references.json.gz /index/index.bin 128 \
    && ls -lh /index/index.bin

FROM --platform=linux/amd64 debian:trixie-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends liburing2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /out/server /server
COPY --from=builder /out/lb /lb
COPY --from=builder /index /index
ENTRYPOINT ["/server"]
