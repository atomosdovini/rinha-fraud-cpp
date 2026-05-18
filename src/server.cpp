// server.cpp — Fraud-detection HTTP server, single-thread epoll reactor.
//
// The load balancer (lb) accepts client TCP sockets on :9999 and hands their
// file descriptors to us over a Unix control socket via SCM_RIGHTS. One epoll
// loop, one thread, owns every client fd: on readiness it does recv → parse →
// search → send. HTTP/1.1 keep-alive.
//
// No io_uring: the Rinha host forbids `security_opt: seccomp:unconfined`, and
// the default seccomp profile blocks the io_uring_* syscalls. epoll uses only
// standard syscalls. A single reactor thread is put on SCHED_FIFO at startup
// (env WORKER_RT) so it preempts the SCHED_OTHER load generator the instant a
// packet arrives — the recipe the top finishers use.

#include "index.hpp"
#include "tx.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <sys/epoll.h>
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
#include <poll.h>
#include <sched.h>
#include <string>
#include <string_view>

// ── Sizing ────────────────────────────────────────────────────────────────────
static constexpr int    MaxConns   = 2048;
static constexpr size_t RxCap      = 8192;
static constexpr int    MaxEvents  = 256;

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
static rinha::IvfIndex* g_index = nullptr;

static struct {
    int      nprobe = 20, repair_min = 99, repair_max = 0;
    int      fast_nprobe = 1, adapt_min = 2, adapt_max = 4;
    uint64_t thr0 = 0, thr1 = 0, thr5 = 0, thr_any = 0;
} g_cfg;

// ── IVF search ────────────────────────────────────────────────────────────────
static uint8_t do_search(const int16_t q[rinha::Dims]) noexcept {
    const auto& c = g_cfg;
    if (c.fast_nprobe > 0 && c.fast_nprobe < c.nprobe) {
        rinha::QueryTrace st;
        uint8_t r = g_index->query(q, c.fast_nprobe, 99, 0, &st);
        if (r < uint8_t(c.adapt_min) || r > uint8_t(c.adapt_max)) {
            uint64_t thr = (r==0) ? c.thr0 : (r==1) ? c.thr1 : (r==5) ? c.thr5 : 0;
            bool rerun = (c.thr_any > 0 && st.final_worst >= c.thr_any)
                      || (thr       > 0 && st.final_worst >= thr);
            if (rerun) return g_index->query(q, c.nprobe, c.repair_min, c.repair_max);
            return r;
        }
    }
    return g_index->query(q, c.nprobe, c.repair_min, c.repair_max);
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
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd pfd { fd, POLLOUT, 0 };
            ::poll(&pfd, 1, 50);
            continue;
        }
        return false;
    }
    return true;
}

// ── Connection table (slot pool) ──────────────────────────────────────────────
struct Conn {
    int      fd   = -1;
    uint32_t have = 0;
    char     buf[RxCap];
};
static Conn g_conns[MaxConns];
static int  g_free[MaxConns];
static int  g_free_top = 0;
static int  g_epfd     = -1;

// epoll token: high 32 bits = kind, low 32 = slot (or fd for ctrl/listen).
enum Kind : uint32_t { K_LISTEN = 1, K_CTRL_LISTEN = 2, K_CTRL = 3, K_CLIENT = 4 };
static uint64_t tok(Kind k, uint32_t v) { return (uint64_t(k) << 32) | v; }
static Kind     tok_kind(uint64_t t)    { return Kind(t >> 32); }
static uint32_t tok_val (uint64_t t)    { return uint32_t(t); }

static void pool_init() {
    for (int i = 0; i < MaxConns; ++i) g_free[i] = MaxConns - 1 - i;
    g_free_top = MaxConns;
}
static int  pool_alloc() { return g_free_top > 0 ? g_free[--g_free_top] : -1; }
static void pool_free(int slot) { g_free[g_free_top++] = slot; }

static void epoll_add(int fd, Kind k, uint32_t v) {
    epoll_event ev{};
    ev.events   = EPOLLIN;
    ev.data.u64 = tok(k, v);
    ::epoll_ctl(g_epfd, EPOLL_CTL_ADD, fd, &ev);
}

static void close_client(int slot) {
    int fd = g_conns[slot].fd;
    ::epoll_ctl(g_epfd, EPOLL_CTL_DEL, fd, nullptr);
    ::close(fd);
    g_conns[slot].fd   = -1;
    g_conns[slot].have = 0;
    pool_free(slot);
}

// Register a freshly accepted client fd into the reactor.
static void add_client(int fd) {
    int slot = pool_alloc();
    if (slot < 0) { ::close(fd); return; }
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,  &one, sizeof(one));
    ::setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
    int fl = ::fcntl(fd, F_GETFL, 0);
    if (fl >= 0) ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    g_conns[slot].fd   = fd;
    g_conns[slot].have = 0;
    epoll_add(fd, K_CLIENT, uint32_t(slot));
}

// ── Process buffered HTTP and reply ───────────────────────────────────────────
static void handle_client(int slot) {
    Conn& c = g_conns[slot];

    ssize_t n = ::recv(c.fd, c.buf + c.have, RxCap - c.have, 0);
    if (n == 0) { close_client(slot); return; }
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return;
        close_client(slot);
        return;
    }
    c.have += uint32_t(n);

    uint32_t consumed = 0;
    while (consumed < c.have) {
        std::string_view data(c.buf + consumed, c.have - consumed);

        size_t hdr_end = data.find("\r\n\r\n");
        if (hdr_end == std::string_view::npos) break;

        std::string_view hdr  = data.substr(0, hdr_end + 4);
        int              clen = http_content_len(hdr);
        size_t           need = hdr_end + 4 + size_t(clen);
        if (data.size() < need) break;

        std::string_view path = http_path(hdr);
        std::string_view body = data.substr(hdr_end + 4, size_t(clen));

        const char* resp; uint32_t rlen;
        if (path == "/fraud-score") {
            int16_t q[rinha::Dims];
            uint8_t b = fraud::extract(body, q) ? do_search(q) : 0;
            if (b > 5) b = 5;
            resp = k_fraud[b].data; rlen = k_fraud[b].len;
        } else if (path == "/ready") {
            resp = k_ready;         rlen = k_ready_len;
        } else {
            resp = k_fraud[0].data; rlen = k_fraud[0].len;
        }
        if (!write_all(c.fd, resp, rlen)) { close_client(slot); return; }

        consumed += uint32_t(need);
    }

    if (consumed > 0) {
        c.have -= consumed;
        if (c.have > 0) ::memmove(c.buf, c.buf + consumed, c.have);
    }
    if (c.have == RxCap) close_client(slot);   // request larger than buffer
}

// ── Receive an fd from the load balancer over a control socket ────────────────
// Returns a client fd, -2 if the socket has no more data, -1 on close/error.
static int recv_fd(int ctrl_fd) noexcept {
    char   b[1];
    char   cmsg[CMSG_SPACE(sizeof(int))];
    iovec  iov { b, 1 };
    msghdr mh  {};
    mh.msg_iov        = &iov;
    mh.msg_iovlen     = 1;
    mh.msg_control    = cmsg;
    mh.msg_controllen = sizeof(cmsg);

    ssize_t n = ::recvmsg(ctrl_fd, &mh, 0);
    if (n <= 0) return (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) ? -2 : -1;

    for (cmsghdr* cm = CMSG_FIRSTHDR(&mh); cm; cm = CMSG_NXTHDR(&mh, cm)) {
        if (cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS) {
            int fd;
            ::memcpy(&fd, CMSG_DATA(cm), sizeof(fd));
            return fd;
        }
    }
    return -1;
}

static void close_ctrl(int fd) {
    ::epoll_ctl(g_epfd, EPOLL_CTL_DEL, fd, nullptr);
    ::close(fd);
}

// ── Listen sockets ────────────────────────────────────────────────────────────
static int tcp_listen(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
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
    int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
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
    auto gu = [](const char* k, uint64_t d) { const char* v = ::getenv(k); return v ? ::strtoull(v,nullptr,10) : d; };
    auto gs = [](const char* k, const char* d) -> const char* { const char* v = ::getenv(k); return v ? v : d; };

    const char* idx_path    = gs("INDEX_PATH", "/index/index.bin");
    const char* listen_path = gs("LISTEN",     "/sockets/api.sock");

    ::signal(SIGPIPE, SIG_IGN);

    g_cfg.nprobe      = gi("NPROBE",                  20);
    g_cfg.repair_min  = gi("REPAIR_MIN",              99);
    g_cfg.repair_max  = gi("REPAIR_MAX",               0);
    g_cfg.fast_nprobe = gi("FAST_NPROBE",              1);
    g_cfg.adapt_min   = gi("ADAPTIVE_MIN",             2);
    g_cfg.adapt_max   = gi("ADAPTIVE_MAX",             4);
    g_cfg.thr0        = gu("EXTREME0_WORST_THRESHOLD", 3501932);
    g_cfg.thr1        = gu("EXTREME1_WORST_THRESHOLD", 3569273);
    g_cfg.thr5        = gu("EXTREME5_WORST_THRESHOLD", 4594089);
    g_cfg.thr_any     = gu("EXTREME_WORST_THRESHOLD",  0);

    static rinha::IvfIndex index(idx_path);
    g_index = &index;

    // Keep every page resident — the hot path never eats a minor-fault stall.
    ::mlockall(MCL_CURRENT | MCL_FUTURE);

    // SCHED_FIFO on this single reactor thread: woken by an inbound packet it
    // preempts the SCHED_OTHER load generator at once instead of waiting for a
    // CPU slice. Needs RLIMIT_RTPRIO headroom (`ulimits: rtprio` in compose),
    // NOT a capability. Graceful no-op if refused.
    if (int rt = gi("WORKER_RT", 0); rt > 0) {
        sched_param sp{};
        sp.sched_priority = rt;
        if (::sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
            ::perror("sched_setscheduler(SCHED_FIFO) — RT disabled");
    }

    pool_init();
    g_epfd = ::epoll_create1(EPOLL_CLOEXEC);
    if (g_epfd < 0) { ::perror("epoll_create1"); ::exit(1); }

    int tcp_port = gi("TCP_PORT", 0);
    int tcp_srv = -1, ctrl_srv = -1;
    if (tcp_port > 0) {
        tcp_srv = tcp_listen(tcp_port);
        epoll_add(tcp_srv, K_LISTEN, uint32_t(tcp_srv));
    } else {
        std::string ctrl_path = std::string(listen_path) + ".ctrl";
        ctrl_srv = unix_listen(ctrl_path.c_str());
        epoll_add(ctrl_srv, K_CTRL_LISTEN, uint32_t(ctrl_srv));
    }

    epoll_event events[MaxEvents];
    for (;;) {
        int n = ::epoll_wait(g_epfd, events, MaxEvents, -1);
        if (n < 0) { if (errno == EINTR) continue; break; }

        for (int i = 0; i < n; ++i) {
            uint64_t t  = events[i].data.u64;
            uint32_t ev = events[i].events;
            switch (tok_kind(t)) {

            case K_LISTEN:
                for (;;) {
                    int fd = ::accept4(tcp_srv, nullptr, nullptr, SOCK_CLOEXEC);
                    if (fd < 0) break;
                    add_client(fd);
                }
                break;

            case K_CTRL_LISTEN:
                for (;;) {
                    int fd = ::accept4(ctrl_srv, nullptr, nullptr,
                                       SOCK_NONBLOCK | SOCK_CLOEXEC);
                    if (fd < 0) break;
                    epoll_add(fd, K_CTRL, uint32_t(fd));
                }
                break;

            case K_CTRL: {
                int ctrl_fd = int(tok_val(t));
                if (ev & (EPOLLHUP | EPOLLERR)) { close_ctrl(ctrl_fd); break; }
                for (;;) {
                    int fd = recv_fd(ctrl_fd);
                    if (fd == -2) break;                 // drained
                    if (fd == -1) { close_ctrl(ctrl_fd); break; }
                    add_client(fd);
                }
                break;
            }

            case K_CLIENT: {
                int slot = int(tok_val(t));
                if (g_conns[slot].fd < 0) break;
                if (ev & (EPOLLHUP | EPOLLERR)) { close_client(slot); break; }
                handle_client(slot);
                break;
            }
            }
        }
    }
    return 0;
}
