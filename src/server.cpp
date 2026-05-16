#include "index.hpp"
#include "tx.hpp"

#include <liburing.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <string_view>
#include <pthread.h>
#include <poll.h>

// ── Sizing ────────────────────────────────────────────────────────────────────
static constexpr uint32_t RingDepth     = 4096;
static constexpr uint32_t MaxConns      = 4096;
static constexpr uint32_t BufSize       = 4096;

// Provided-buffer ring used by multishot recv. The kernel picks a buffer
// from this ring for each incoming chunk and reports the id via cqe->flags.
static constexpr uint16_t BufGroup      = 1;
static constexpr uint32_t BufRingShift  = 9;
static constexpr uint32_t BufRingSize   = 1u << BufRingShift;   // 512 buffers
static constexpr uint32_t BufRingMask   = BufRingSize - 1;
static constexpr uint32_t BufEntrySize  = 2048;

// ── CQE user_data encoding: [44-bit gen | 12-bit slot | 8-bit op] ────────────
// Generation counter prevents stale CQEs from firing on recycled slots.
enum class Op : uint8_t {
    Accept    = 0,
    AcceptCtrl= 1,
    PollCtrl  = 2,
    Recv      = 3,
    Send      = 4,
};

static inline uint64_t make_ud(Op op, uint32_t slot, uint64_t gen = 0) noexcept {
    return (gen << 20) | (uint64_t(slot & 0xFFF) << 8) | uint8_t(op);
}
static inline Op       ud_op  (uint64_t ud) noexcept { return Op(ud & 0xFF); }
static inline uint32_t ud_slot(uint64_t ud) noexcept { return uint32_t((ud >> 8) & 0xFFF); }
static inline uint64_t ud_gen (uint64_t ud) noexcept { return ud >> 20; }

// ── Per-connection state ──────────────────────────────────────────────────────
struct alignas(64) Conn {
    int      fd           = -1;
    uint32_t have         = 0;    // bytes buffered
    uint16_t sends        = 0;    // in-flight sends (for pipelining)
    bool     keep_alive   = true;
    uint64_t gen          = 0;    // generation counter; incremented on reuse
    char     buf[BufSize];
};

// ── Per-worker state ──────────────────────────────────────────────────────────
struct Worker {
    io_uring ring{};
    int      srv      = -1;
    int      srv_ctrl = -1;

    Conn     conns[MaxConns];
    uint32_t free_stack[MaxConns];
    int      free_top = 0;

    // ctrl fds are persistent (not pooled) — track by fd value
    bool     is_ctrl[65536] = {};

    // Multishot recv buffer ring
    io_uring_buf_ring* buf_ring = nullptr;
    uint8_t*           buf_pool = nullptr;

    const rinha::IvfIndex* idx = nullptr;

    struct {
        int      nprobe      = 20;
        int      repair_min  = 99;
        int      repair_max  = 0;
        int      fast_nprobe = 1;
        int      adapt_min   = 2;
        int      adapt_max   = 4;
        int      busy_poll   = 0;   // spin iterations before blocking wait
        uint64_t thr0 = 0, thr1 = 0, thr5 = 0, thr_any = 0;
    } cfg;

    void pool_init() {
        for (uint32_t i = 0; i < MaxConns; ++i)
            free_stack[i] = MaxConns - 1 - i;
        free_top = int(MaxConns);
    }
    uint32_t pool_alloc() {
        return free_top > 0 ? free_stack[--free_top] : UINT32_MAX;
    }
    void pool_free(uint32_t slot) {
        conns[slot].fd         = -1;
        conns[slot].have       = 0;
        conns[slot].sends      = 0;
        conns[slot].keep_alive = true;
        ++conns[slot].gen;           // invalidate any in-flight CQEs for this slot
        free_stack[free_top++] = slot;
    }
};

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

// ── IVF search ────────────────────────────────────────────────────────────────
static uint8_t do_search(const Worker& w, const int16_t q[rinha::Dims]) noexcept {
    const auto& c = w.cfg;
    if (c.fast_nprobe > 0 && c.fast_nprobe < c.nprobe) {
        rinha::QueryTrace st;
        uint8_t r = w.idx->query(q, c.fast_nprobe, 99, 0, &st);
        if (r < uint8_t(c.adapt_min) || r > uint8_t(c.adapt_max)) {
            uint64_t thr = (r==0) ? c.thr0 : (r==1) ? c.thr1 : (r==5) ? c.thr5 : 0;
            bool rerun = (c.thr_any > 0 && st.final_worst >= c.thr_any)
                      || (thr       > 0 && st.final_worst >= thr);
            if (rerun) return w.idx->query(q, c.nprobe, c.repair_min, c.repair_max);
            return r;
        }
    }
    return w.idx->query(q, c.nprobe, c.repair_min, c.repair_max);
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

// ── io_uring submission helpers ───────────────────────────────────────────────
static void sq_multishot_accept(io_uring* ring, int srv, Op op) {
    io_uring_sqe* sqe = io_uring_get_sqe(ring);
    io_uring_prep_multishot_accept(sqe, srv, nullptr, nullptr, 0);
    io_uring_sqe_set_data64(sqe, make_ud(op, 0));
}

// Multishot recv: kernel keeps SQE armed, picks a buffer from the ring for
// each incoming chunk. One CQE per chunk (with cqe->flags carrying the buf id).
static void sq_recv(io_uring* ring, Worker& w, uint32_t slot) {
    Conn& c = w.conns[slot];
    io_uring_sqe* sqe = io_uring_get_sqe(ring);
    io_uring_prep_recv_multishot(sqe, c.fd, nullptr, 0, 0);
    sqe->buf_group = BufGroup;
    sqe->flags    |= IOSQE_BUFFER_SELECT;
    io_uring_sqe_set_data64(sqe, make_ud(Op::Recv, slot, c.gen));
}

// Return a buffer to the ring after we're done with its contents.
static inline void buf_release(Worker& w, uint16_t bid) {
    io_uring_buf_ring_add(w.buf_ring,
                          w.buf_pool + size_t(bid) * BufEntrySize,
                          BufEntrySize, bid, BufRingMask, 0);
    io_uring_buf_ring_advance(w.buf_ring, 1);
}

static void sq_send(io_uring* ring, Worker& w, uint32_t slot, const char* data, uint32_t len) {
    Conn& c = w.conns[slot];
    io_uring_sqe* sqe = io_uring_get_sqe(ring);
    io_uring_prep_send(sqe, c.fd, data, len, MSG_NOSIGNAL);
    io_uring_sqe_set_data64(sqe, make_ud(Op::Send, slot, c.gen));
    c.sends++;
}

static void sq_poll_ctrl(io_uring* ring, int ctrl_fd) {
    io_uring_sqe* sqe = io_uring_get_sqe(ring);
    io_uring_prep_poll_add(sqe, ctrl_fd, POLLIN);
    // slot field holds the ctrl fd directly (ctrl fds are not pooled)
    io_uring_sqe_set_data64(sqe, make_ud(Op::PollCtrl, uint32_t(ctrl_fd)));
}

// ── Close connection ──────────────────────────────────────────────────────────
static void drop_conn(Worker& w, uint32_t slot) {
    ::close(w.conns[slot].fd);
    w.pool_free(slot);
}

// ── Process buffered HTTP data ─────────────────────────────────────────────────
// Returns false if the connection should be dropped after pending sends flush.
static bool process(io_uring* ring, Worker& w, uint32_t slot) {
    Conn& c = w.conns[slot];
    size_t consumed = 0;

    while (consumed < c.have) {
        std::string_view data(c.buf + consumed, c.have - consumed);

        // Find end of HTTP headers within the current request window.
        size_t hdr_end = data.find("\r\n\r\n");
        if (hdr_end == std::string_view::npos) break;

        std::string_view hdr  = data.substr(0, hdr_end + 4);
        int              clen = http_content_len(hdr);
        size_t           need = hdr_end + 4 + size_t(clen);
        if (data.size() < need) break;   // body not yet fully buffered

        std::string_view path = http_path(hdr);
        std::string_view body = data.substr(hdr_end + 4, size_t(clen));

        if (path == "/fraud-score") {
            int16_t q[rinha::Dims];
            uint8_t bucket = fraud::extract(body, q) ? do_search(w, q) : 0;
            if (bucket > 5) bucket = 5;
            sq_send(ring, w, slot, k_fraud[bucket].data, k_fraud[bucket].len);
        } else if (path == "/ready") {
            sq_send(ring, w, slot, k_ready, k_ready_len);
        } else {
            sq_send(ring, w, slot, k_fraud[0].data, k_fraud[0].len);
        }

        consumed += need;
    }

    if (consumed > 0) {
        c.have -= uint32_t(consumed);
        if (c.have > 0) ::memmove(c.buf, c.buf + consumed, c.have);
    }

    return c.have < BufSize;   // false = buffer full → drop after sends flush
}

// ── Drain ctrl connection (fd-passing via SCM_RIGHTS) ─────────────────────────
static void drain_ctrl(io_uring* ring, Worker& w, int ctrl_fd) {
    char         byte[1];
    char         cmsg_buf[CMSG_SPACE(sizeof(int))];

    for (;;) {
        iovec  iov { byte, 1 };
        msghdr mh  {};
        mh.msg_iov        = &iov;
        mh.msg_iovlen     = 1;
        mh.msg_control    = cmsg_buf;
        mh.msg_controllen = sizeof(cmsg_buf);

        ssize_t n = ::recvmsg(ctrl_fd, &mh, MSG_DONTWAIT);
        if (n <= 0) break;

        cmsghdr* cm = CMSG_FIRSTHDR(&mh);
        if (!cm || cm->cmsg_level != SOL_SOCKET || cm->cmsg_type != SCM_RIGHTS) continue;

        int passed_fd;
        ::memcpy(&passed_fd, CMSG_DATA(cm), sizeof(int));
        if (passed_fd < 0) continue;

        uint32_t slot = w.pool_alloc();
        if (slot == UINT32_MAX) { ::close(passed_fd); continue; }

        // make non-blocking and tune
        int flags = ::fcntl(passed_fd, F_GETFL, 0);
        if (flags >= 0) ::fcntl(passed_fd, F_SETFL, flags | O_NONBLOCK);
        int one = 1;
        ::setsockopt(passed_fd, IPPROTO_TCP, TCP_NODELAY,  &one, sizeof(one));
        ::setsockopt(passed_fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));

        w.conns[slot].fd         = passed_fd;
        w.conns[slot].have       = 0;
        w.conns[slot].sends      = 0;
        w.conns[slot].keep_alive = true;
        sq_recv(ring, w, slot);
    }

    sq_poll_ctrl(ring, ctrl_fd);   // re-arm for next fd
}

// ── Event loop ─────────────────────────────────────────────────────────────────
static void run(Worker& w) {
    // Prefer DEFER_TASKRUN: completion handlers run on our thread inside
    // io_uring_enter() instead of from interrupt context. Cuts wake-up jitter,
    // requires SINGLE_ISSUER. COOP_TASKRUN reduces preemption overhead.
    // Fall back gracefully on older kernels.
    io_uring_params p{};
    p.flags = IORING_SETUP_SINGLE_ISSUER
            | IORING_SETUP_COOP_TASKRUN
            | IORING_SETUP_DEFER_TASKRUN;
    if (io_uring_queue_init_params(RingDepth, &w.ring, &p) < 0) {
        p = {};
        p.flags = IORING_SETUP_COOP_TASKRUN;
        if (io_uring_queue_init_params(RingDepth, &w.ring, &p) < 0) {
            if (io_uring_queue_init(RingDepth, &w.ring, 0) < 0) {
                ::perror("io_uring_queue_init"); ::exit(1);
            }
        }
    }
    io_uring* ring = &w.ring;

    // ── Provided buffer ring for multishot recv ──────────────────────────────
    if (::posix_memalign(reinterpret_cast<void**>(&w.buf_pool), 4096,
                         size_t(BufRingSize) * BufEntrySize) != 0) {
        ::perror("posix_memalign"); ::exit(1);
    }
    int br_ret = 0;
    w.buf_ring = io_uring_setup_buf_ring(ring, BufRingSize, BufGroup, 0, &br_ret);
    if (!w.buf_ring) {
        ::fprintf(stderr, "io_uring_setup_buf_ring failed: %d\n", br_ret);
        ::exit(1);
    }
    for (uint32_t i = 0; i < BufRingSize; ++i) {
        io_uring_buf_ring_add(w.buf_ring,
                              w.buf_pool + size_t(i) * BufEntrySize,
                              BufEntrySize, uint16_t(i), BufRingMask, int(i));
    }
    io_uring_buf_ring_advance(w.buf_ring, int(BufRingSize));

    sq_multishot_accept(ring, w.srv, Op::Accept);
    if (w.srv_ctrl >= 0)
        sq_multishot_accept(ring, w.srv_ctrl, Op::AcceptCtrl);
    io_uring_submit(ring);

    // Busy-poll: after draining completions, spin briefly probing the CQ
    // before falling back to a blocking wait. Keeps the core at high
    // frequency (avoids the powersave downclock) and removes scheduler
    // wake-up latency from the tail. Bounded so an idle worker still sleeps
    // and stays inside the CPU budget.
    const int spin = w.cfg.busy_poll;

    io_uring_cqe* cqe;
    while (true) {
        if (io_uring_peek_cqe(ring, &cqe) != 0) {
            int got = -1;
            for (int s = 0; s < spin; ++s) {
                io_uring_submit_and_get_events(ring);   // run deferred task work
                if (io_uring_peek_cqe(ring, &cqe) == 0) { got = 0; break; }
                for (int p = 0; p < 64; ++p) __builtin_ia32_pause();
            }
            if (got != 0 && io_uring_wait_cqe(ring, &cqe) < 0) continue;
        }

        uint32_t head, count = 0;
        io_uring_for_each_cqe(ring, head, cqe) {
            ++count;
            uint64_t ud   = io_uring_cqe_get_data64(cqe);
            Op       op   = ud_op(ud);
            uint32_t slot = ud_slot(ud);
            int      res  = cqe->res;

            switch (op) {
            case Op::Accept:
            case Op::AcceptCtrl: {
                if (res >= 0) {
                    int new_fd = res;
                    if (op == Op::Accept) {
                        uint32_t s = w.pool_alloc();
                        if (s == UINT32_MAX) { ::close(new_fd); break; }
                        int fl = ::fcntl(new_fd, F_GETFL, 0);
                        if (fl >= 0) ::fcntl(new_fd, F_SETFL, fl | O_NONBLOCK);
                        int one = 1;
                        ::setsockopt(new_fd, IPPROTO_TCP, TCP_NODELAY,  &one, sizeof(one));
                        ::setsockopt(new_fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
                        // Do NOT reset gen: pool_free already incremented it.
                        w.conns[s].fd         = new_fd;
                        w.conns[s].have       = 0;
                        w.conns[s].sends      = 0;
                        w.conns[s].keep_alive = true;
                        sq_recv(ring, w, s);
                    } else {
                        // ctrl connection from lb
                        if (new_fd < 65536) w.is_ctrl[new_fd] = true;
                        sq_poll_ctrl(ring, new_fd);
                    }
                }
                // multishot: re-arm only if exhausted
                if (!(cqe->flags & IORING_CQE_F_MORE))
                    sq_multishot_accept(ring, op == Op::Accept ? w.srv : w.srv_ctrl, op);
                break;
            }

            case Op::PollCtrl: {
                if (res > 0) drain_ctrl(ring, w, int(slot));
                else         sq_poll_ctrl(ring, int(slot));  // re-arm on error too
                break;
            }

            case Op::Recv: {
                bool more = (cqe->flags & IORING_CQE_F_MORE);
                bool has_buf = (cqe->flags & IORING_CQE_F_BUFFER);
                uint16_t bid = uint16_t(cqe->flags >> IORING_CQE_BUFFER_SHIFT);

                // Stale CQE for a recycled slot: just return any buffer the kernel
                // picked, then drop the rest.
                if (ud_gen(ud) != w.conns[slot].gen) {
                    if (has_buf) buf_release(w, bid);
                    break;
                }

                Conn& cn = w.conns[slot];

                if (res <= 0) {
                    if (has_buf) buf_release(w, bid);
                    cn.keep_alive = false;
                    if (cn.sends == 0) drop_conn(w, slot);
                    break;
                }

                if (!has_buf) {
                    // No buffer selected — kernel out of buffers or transient.
                    // Re-arm multishot to recover.
                    if (!more) sq_recv(ring, w, slot);
                    break;
                }

                const uint8_t* src  = w.buf_pool + size_t(bid) * BufEntrySize;
                uint32_t       need = uint32_t(res);

                if (cn.have + need > BufSize) {
                    // Per-conn accumulator full — drop this connection.
                    buf_release(w, bid);
                    cn.keep_alive = false;
                    if (cn.sends == 0) drop_conn(w, slot);
                    break;
                }

                ::memcpy(cn.buf + cn.have, src, need);
                cn.have += need;
                buf_release(w, bid);

                bool alive = process(ring, w, slot);
                if (!alive) {
                    cn.keep_alive = false;
                    if (cn.sends == 0) drop_conn(w, slot);
                }

                // If multishot was disarmed (rare, e.g. ENOBUFS), re-submit it.
                if (!more && alive) sq_recv(ring, w, slot);
                break;
            }

            case Op::Send: {
                // Discard stale CQEs from a recycled slot.
                if (ud_gen(ud) != w.conns[slot].gen) break;
                if (res <= 0) {
                    --w.conns[slot].sends;
                    w.conns[slot].keep_alive = false;
                    if (w.conns[slot].sends == 0) drop_conn(w, slot);
                    break;
                }
                if (--w.conns[slot].sends == 0 && !w.conns[slot].keep_alive)
                    drop_conn(w, slot);
                // recv is already pending from Op::Recv — nothing to repost.
                break;
            }
            }
        }
        io_uring_cq_advance(ring, count);
        io_uring_submit(ring);
    }
}

// ── TCP listen socket (SO_REUSEPORT — kernel load-balances across instances) ──
static int tcp_listen(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) { ::perror("socket"); ::exit(1); }
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
    // Defer accept(): kernel mantém a conexão em SYN_RCVD até chegar o primeiro
    // dado — elimina uma ida ao io_uring antes do request estar pronto.
    ::setsockopt(fd, IPPROTO_TCP, TCP_DEFER_ACCEPT, &one, sizeof(one));
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

// ── Unix listen socket ────────────────────────────────────────────────────────
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

static void* thread_entry(void* arg) {
    run(*static_cast<Worker*>(arg));
    return nullptr;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    auto gi  = [](const char* k, int d)      { const char* v = ::getenv(k); return v ? ::atoi(v) : d; };
    auto gu  = [](const char* k, uint64_t d) { const char* v = ::getenv(k); return v ? ::strtoull(v,nullptr,10) : d; };
    auto gs  = [](const char* k, const char* d) -> const char* { const char* v = ::getenv(k); return v ? v : d; };

    const char* idx_path    = gs("INDEX_PATH", "/index/index.bin");
    const char* listen_path = gs("LISTEN",     "/sockets/api.sock");
    int         workers     = gi("WORKERS", 1);
    if (workers < 1 || workers > 64) workers = 1;

    ::signal(SIGPIPE, SIG_IGN);

    // PM QoS: cap the CPU idle wake-up latency via /dev/cpu_dma_latency.
    // `PM_QOS` is the target in µs — the kernel then forbids any C-state
    // whose exit latency exceeds it. Picking ~10 µs keeps the shallow,
    // low-power C1/C1E states (fast wake, cool) but forbids C6 (~85 µs),
    // which is the dominant term in the p99 tail when a worker is woken.
    // NOTE: do NOT use 0 — that forbids every C-state, pinning idle cores
    // in C0 (polling) at full power → the laptop overheats and the CPU
    // thermal-throttles, making everything *slower*. 0/unset = disabled.
    // The fd must stay open for the constraint to hold; leaked on purpose.
    int pm_qos = gi("PM_QOS", 0);
    if (pm_qos > 0) {
        int pm_fd = ::open("/dev/cpu_dma_latency", O_WRONLY | O_CLOEXEC);
        if (pm_fd >= 0) {
            int32_t target = pm_qos;
            if (::write(pm_fd, &target, sizeof(target)) != sizeof(target))
                ::close(pm_fd);
        }
    }

    int tcp_port = gi("TCP_PORT", 0);
    int srv, srv_ctrl = -1;
    if (tcp_port > 0) {
        srv = tcp_listen(tcp_port);
    } else {
        std::string ctrl_path = std::string(listen_path) + ".ctrl";
        srv      = unix_listen(listen_path);
        srv_ctrl = unix_listen(ctrl_path.c_str());
    }

    rinha::IvfIndex idx(idx_path);

    // Each thread owns its own Worker (~16MB of conn buffers) — always heap-allocated.
    auto* wpool = new Worker[size_t(workers)];
    for (int i = 0; i < workers; ++i) {
        Worker& w    = wpool[i];
        w.srv        = srv;
        w.srv_ctrl   = srv_ctrl;
        w.idx        = &idx;
        w.cfg.nprobe      = gi("NPROBE",                  20);
        w.cfg.repair_min  = gi("REPAIR_MIN",              99);
        w.cfg.repair_max  = gi("REPAIR_MAX",               0);
        w.cfg.fast_nprobe = gi("FAST_NPROBE",              1);
        w.cfg.adapt_min   = gi("ADAPTIVE_MIN",             2);
        w.cfg.adapt_max   = gi("ADAPTIVE_MAX",             4);
        w.cfg.busy_poll   = gi("BUSY_POLL",                0);
        w.cfg.thr0        = gu("EXTREME0_WORST_THRESHOLD", 3501932);
        w.cfg.thr1        = gu("EXTREME1_WORST_THRESHOLD", 3569273);
        w.cfg.thr5        = gu("EXTREME5_WORST_THRESHOLD", 4594089);
        w.cfg.thr_any     = gu("EXTREME_WORST_THRESHOLD",  0);
        w.pool_init();
    }

    if (workers == 1) {
        run(wpool[0]);
        return 0;
    }

    pthread_t* threads = new pthread_t[size_t(workers - 1)];
    for (int i = 1; i < workers; ++i)
        pthread_create(&threads[i - 1], nullptr, thread_entry, &wpool[i]);

    run(wpool[0]);
    return 0;
}
