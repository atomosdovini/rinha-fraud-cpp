CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -march=haswell -flto -Wall -Wextra -Wno-unused-parameter

.PHONY: build index run clean docker-up docker-down perf

build:
	mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp src/build_index.cpp -lz -o build/build-index
	$(CXX) $(CXXFLAGS) -pthread src/server.cpp -o build/server

index: build
	mkdir -p build/index
	./build/build-index bench/references.json.gz build/index/index.bin 4096 50000 10

run: build
	mkdir -p /rinha-sockets
	INDEX_PATH=build/index/index.bin LISTEN=/rinha-sockets/api.sock WORKERS=2 ./build/server

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

perf:
	cd /tmp/rinha-de-backend-2026 && K6_NO_USAGE_REPORT=true k6 run test/test.js >/dev/null && jq . test/results.json

clean:
	rm -rf build
