// server.cpp — Fraud-detection HTTP server, thread-per-connection.
//
// The load balancer (lb.cpp) accepts client TCP sockets on :9999 and hands
// their file descriptors to us over a Unix control socket via SCM_RIGHTS.
// For each received fd we spawn one detached worker thread that owns that
// connection start to finish: blocking recv → parse → search → send, looping
// for HTTP/1.1 keep-alive.
//
// No io_uring. At the test's concurrency (a few hundred keep-alive
// connections, ~1 request in flight each) a thread blocked in recv() is woken
// directly by the kernel with zero head-of-line blocking — lower tail latency
// than a single shared event-loop reactor on a CPU-contended host. (Measured:
// the io_uring reactor sat at p99 ~1.02ms on the Rinha host; this model is
// what the top finishers use.)

#include "index.hpp"
#include "tx.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <sys/mman.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <string_view>
#include <pthread.h>

// ── Sizing ────────────────────────────────────────────────────────────────────
static constexpr size_t RxCap        = 8192;          // per-connection rx buffer
static constexpr size_t WorkerStack   = 128 * 1024;    // detached worker stack

// ── Pre-built responses ───────────────────────────────────────────────────────
static constexpr struct { const char* data; uint32_t len; } k_fraud[6] = {
    {"HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",  74},
    {"HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",  74},
    {"HTTP/1.1 200 OK\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",  74},
    {"HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}", 75},
    {"HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}", 75},
    {"HTTP/1.1 200 OK\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}", 75},
};
static constexpr char k_ready[] =
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}";
static constexpr uint32_t k_ready_len = sizeof(k_ready) - 1;

// ── Shared, read-only after startup ───────────────────────────────────────────
static rinha::BvhIndex* g_index = nullptr;
static pthread_attr_t   g_worker_attr;

// ── Exact BVH search ──────────────────────────────────────────────────────────
// Single-pass branch-and-bound k-NN — exact, no re-run, constant-ish cost. The
// IVF re-run (a 20-cluster, ~47k-vector rescan) was the p99 spike; this has no
// such tail.
static uint8_t do_search(const int16_t q[rinha::Dims]) noexcept {
    return g_index->query(q);
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────
static std::string_view http_path(std::string_view req) noexcept {
    size_t a = req.find(' ');
    if (a == std::string_view::npos) return {};
    size_t b = req.find(' ', a + 1);
    if (b == std::string_view::npos) return {};
    return req.substr(a + 1, b - a - 1);
}

static int http_content_len(std::string_view hdr) noexcept {
    for (const char* tag : {"Content-Length:", "content-length:"}) {
        size_t p = hdr.find(tag);
        if (p == std::string_view::npos) continue;
        p += 15;
        while (p < hdr.size() && hdr[p] == ' ') ++p;
        int n = 0;
        while (p < hdr.size() && hdr[p] >= '0' && hdr[p] <= '9')
            n = n * 10 + (hdr[p++] - '0');
        return n;
    }
    return 0;
}

static bool write_all(int fd, const char* p, size_t len) noexcept {
    while (len > 0) {
        ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
        if (n > 0) { p += n; len -= size_t(n); continue; }
        if (n < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

// ── Connection worker — owns one client fd start to finish ────────────────────
static void* serve_client(void* arg) {
    int  fd = int(intptr_t(arg));
    char buf[RxCap];
    int  have = 0;

    for (;;) {
        ssize_t n = ::recv(fd, buf + have, RxCap - size_t(have), 0);
        if (n <= 0) { if (n < 0 && errno == EINTR) continue; break; }
        have += int(n);

        int consumed = 0;
        while (consumed < have) {
            std::string_view data(buf + consumed, size_t(have - consumed));

            size_t hdr_end = data.find("\r\n\r\n");
            if (hdr_end == std::string_view::npos) break;

            std::string_view hdr  = data.substr(0, hdr_end + 4);
            int              clen = http_content_len(hdr);
            size_t           need = hdr_end + 4 + size_t(clen);
            if (data.size() < need) break;            // body not fully buffered

            std::string_view path = http_path(hdr);
            std::string_view body = data.substr(hdr_end + 4, size_t(clen));

            const char* resp; uint32_t rlen;
            if (path == "/fraud-score") {
                int16_t q[rinha::Dims];
                uint8_t b = fraud::extract(body, q) ? do_search(q) : 0;
                if (b > 5) b = 5;
                resp = k_fraud[b].data; rlen = k_fraud[b].len;
            } else if (path == "/ready") {
                resp = k_ready;        rlen = k_ready_len;
            } else {
                resp = k_fraud[0].data; rlen = k_fraud[0].len;
            }
            if (!write_all(fd, resp, rlen)) { ::close(fd); return nullptr; }

            consumed += int(need);
        }

        if (consumed > 0) {
            have -= consumed;
            if (have > 0) ::memmove(buf, buf + consumed, size_t(have));
        }
        if (have == int(RxCap)) break;                // request larger than buffer
    }

    ::close(fd);
    return nullptr;
}

static void spawn_client(int fd) {
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,  &one, sizeof(one));
    ::setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
    pthread_t tid;
    if (pthread_create(&tid, &g_worker_attr, serve_client, (void*)intptr_t(fd)) != 0)
        ::close(fd);
}

// ── Receive one fd from the load balancer over a control socket ───────────────
static int recv_fd(int ctrl_fd) noexcept {
    char   b[1];
    char   cmsg[CMSG_SPACE(sizeof(int))];
    iovec  iov { b, 1 };
    msghdr mh  {};
    mh.msg_iov        = &iov;
    mh.msg_iovlen     = 1;
    mh.msg_control    = cmsg;
    mh.msg_controllen = sizeof(cmsg);

    ssize_t n;
    do { n = ::recvmsg(ctrl_fd, &mh, 0); } while (n < 0 && errno == EINTR);
    if (n <= 0) return -1;

    for (cmsghdr* c = CMSG_FIRSTHDR(&mh); c; c = CMSG_NXTHDR(&mh, c)) {
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
            int fd;
            ::memcpy(&fd, CMSG_DATA(c), sizeof(fd));
            return fd;
        }
    }
    return -1;
}

// One control connection from the LB carries a stream of client fds.
static void* serve_control(void* arg) {
    int ctrl_fd = int(intptr_t(arg));
    for (;;) {
        int fd = recv_fd(ctrl_fd);
        if (fd < 0) break;
        spawn_client(fd);
    }
    ::close(ctrl_fd);
    return nullptr;
}

// ── Listen sockets ────────────────────────────────────────────────────────────
static int tcp_listen(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) { ::perror("socket"); ::exit(1); }
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(uint16_t(port));
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::perror("bind tcp"); ::exit(1);
    }
    if (::listen(fd, 65535) != 0) { ::perror("listen"); ::exit(1); }
    return fd;
}

static int unix_listen(const char* path) {
    ::unlink(path);
    int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) { ::perror("socket"); ::exit(1); }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    ::strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::perror("bind"); ::exit(1);
    }
    ::chmod(path, 0777);
    if (::listen(fd, 65535) != 0) { ::perror("listen"); ::exit(1); }
    return fd;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    auto gi = [](const char* k, int d)      { const char* v = ::getenv(k); return v ? ::atoi(v) : d; };
    auto gs = [](const char* k, const char* d) -> const char* { const char* v = ::getenv(k); return v ? v : d; };

    const char* idx_path    = gs("INDEX_PATH", "/index/index.bin");
    const char* listen_path = gs("LISTEN",     "/sockets/api.sock");

    ::signal(SIGPIPE, SIG_IGN);

    // PM QoS: cap the CPU idle wake-up latency via /dev/cpu_dma_latency.
    // PM_QOS is the target in µs; the kernel then forbids any C-state whose
    // exit latency exceeds it. ~10 keeps shallow C1/C1E (fast wake) but
    // forbids C6 (~85 µs). 0/unset = disabled. Never use 0 as the target —
    // it forbids every C-state and the host thermal-throttles. fd leaked on
    // purpose so the constraint holds for the process lifetime.
    int pm_qos = gi("PM_QOS", 0);
    if (pm_qos > 0) {
        int pm_fd = ::open("/dev/cpu_dma_latency", O_WRONLY | O_CLOEXEC);
        if (pm_fd >= 0) {
            int32_t target = pm_qos;
            if (::write(pm_fd, &target, sizeof(target)) != sizeof(target))
                ::close(pm_fd);
        }
    }

    static rinha::BvhIndex index(idx_path);
    g_index = &index;

    // Keep every page resident so the hot path never eats a minor-fault
    // stall. Best-effort: silently skipped without RLIMIT_MEMLOCK headroom.
    ::mlockall(MCL_CURRENT | MCL_FUTURE);

    pthread_attr_init(&g_worker_attr);
    pthread_attr_setdetachstate(&g_worker_attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&g_worker_attr, WorkerStack);

    int tcp_port = gi("TCP_PORT", 0);
    if (tcp_port > 0) {
        // Direct mode: accept TCP clients ourselves (no load balancer).
        int srv = tcp_listen(tcp_port);
        for (;;) {
            int fd = ::accept4(srv, nullptr, nullptr, SOCK_CLOEXEC);
            if (fd < 0) { if (errno == EINTR) continue; continue; }
            spawn_client(fd);
        }
    } else {
        // LB mode: the load balancer connects to our control socket and
        // streams client fds over it. One thread per control connection.
        std::string ctrl_path = std::string(listen_path) + ".ctrl";
        int srv_ctrl = unix_listen(ctrl_path.c_str());
        for (;;) {
            int cfd = ::accept4(srv_ctrl, nullptr, nullptr, SOCK_CLOEXEC);
            if (cfd < 0) { if (errno == EINTR) continue; continue; }
            pthread_t tid;
            if (pthread_create(&tid, &g_worker_attr, serve_control,
                               (void*)intptr_t(cfd)) != 0)
                ::close(cfd);
        }
    }
    return 0;
}
