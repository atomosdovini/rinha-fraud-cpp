// fd-passing load balancer
// Accepts TCP on PORT, round-robins client fds to worker ctrl sockets via SCM_RIGHTS.
// Never writes to the client TCP socket — no spurious bytes to k6.

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
#include <vector>
#include <pthread.h>

// ── Connect (with retry) to a ctrl Unix socket ────────────────────────────────
static int connect_ctrl(const char* path) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    ::strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    for (int i = 0; i < 100; ++i) {
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0)
            return fd;
        ::usleep(100'000);
    }
    ::close(fd);
    return -1;
}

// ── Pass client_fd to worker via SCM_RIGHTS ───────────────────────────────────
static bool pass_fd(int ctrl_fd, int client_fd) {
    char      byte    = '!';
    iovec     iov     { &byte, 1 };
    char      cmbuf[CMSG_SPACE(sizeof(int))];
    msghdr    mh      {};
    mh.msg_iov        = &iov;
    mh.msg_iovlen     = 1;
    mh.msg_control    = cmbuf;
    mh.msg_controllen = sizeof(cmbuf);
    cmsghdr* cm       = CMSG_FIRSTHDR(&mh);
    cm->cmsg_level    = SOL_SOCKET;
    cm->cmsg_type     = SCM_RIGHTS;
    cm->cmsg_len      = CMSG_LEN(sizeof(int));
    ::memcpy(CMSG_DATA(cm), &client_fd, sizeof(int));
    return ::sendmsg(ctrl_fd, &mh, MSG_NOSIGNAL) > 0;
}

// ── Per-worker state ──────────────────────────────────────────────────────────
struct Worker {
    int                      srv = -1;
    std::vector<std::string> ctrl_paths;
    std::vector<int>         ctrl_fds;
    int                      rr = 0;
};

static void worker_run(Worker& w) {
    int n = int(w.ctrl_paths.size());
    w.ctrl_fds.assign(n, -1);

    for (int i = 0; i < n; ++i) {
        w.ctrl_fds[i] = connect_ctrl(w.ctrl_paths[i].c_str());
        if (w.ctrl_fds[i] < 0) {
            ::fprintf(stderr, "lb: failed to connect ctrl %s\n", w.ctrl_paths[i].c_str());
            ::exit(1);
        }
        ::fprintf(stderr, "lb: ctrl[%d] connected to %s\n", i, w.ctrl_paths[i].c_str());
    }

    while (true) {
        int cfd = ::accept4(w.srv, nullptr, nullptr, SOCK_CLOEXEC);
        if (cfd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            ::perror("accept4");
            continue;
        }

        // Round-robin with reconnect on failure.
        bool sent = false;
        for (int attempt = 0; attempt < n && !sent; ++attempt) {
            int idx = w.rr % n;
            w.rr    = (w.rr + 1) % n;

            if (w.ctrl_fds[idx] < 0)
                w.ctrl_fds[idx] = connect_ctrl(w.ctrl_paths[idx].c_str());

            if (w.ctrl_fds[idx] >= 0 && pass_fd(w.ctrl_fds[idx], cfd)) {
                sent = true;
            } else {
                if (w.ctrl_fds[idx] >= 0) { ::close(w.ctrl_fds[idx]); w.ctrl_fds[idx] = -1; }
            }
        }

        // Always close our copy — the worker has its own duplicate.
        ::close(cfd);

        if (!sent) ::fprintf(stderr, "lb: dropped connection, no healthy upstream\n");
    }
}

static void* thread_entry(void* arg) {
    worker_run(*static_cast<Worker*>(arg));
    return nullptr;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    ::signal(SIGPIPE, SIG_IGN);

    auto gi = [](const char* k, int d) { const char* v = ::getenv(k); return v ? ::atoi(v) : d; };

    int         port        = gi("PORT", 9999);
    int         num_workers = gi("WORKERS", 1);
    const char* ups_env     = ::getenv("UPSTREAMS");
    if (!ups_env) { ::fprintf(stderr, "lb: UPSTREAMS env required\n"); return 1; }
    if (num_workers < 1) num_workers = 1;

    // Parse comma-separated upstream paths; ctrl path = upstream + ".ctrl"
    std::vector<std::string> ctrl_paths;
    for (const char* p = ups_env;;) {
        const char* comma = ::strchr(p, ',');
        std::string path(p, comma ? size_t(comma - p) : ::strlen(p));
        ctrl_paths.push_back(path + ".ctrl");
        if (!comma) break;
        p = comma + 1;
    }

    // TCP listen socket shared by all workers.
    int srv = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (srv < 0) { ::perror("socket"); return 1; }
    int one = 1;
    ::setsockopt(srv, SOL_SOCKET,  SO_REUSEADDR,      &one, sizeof(one));
    // Defer accept(): kernel keeps the conn in SYN_RCVD until first data arrives.
    // Eliminates one wake-up and (usually) one RTT for new connections.
    int defer = 1;
    ::setsockopt(srv, IPPROTO_TCP, TCP_DEFER_ACCEPT, &defer, sizeof(defer));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(uint16_t(port));
    if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::perror("bind"); return 1;
    }
    if (::listen(srv, 65535) != 0) { ::perror("listen"); return 1; }

    ::fprintf(stderr, "lb: port=%d upstreams=%zu workers=%d\n",
              port, ctrl_paths.size(), num_workers);

    auto* pool    = new Worker[size_t(num_workers)];
    auto* threads = new pthread_t[size_t(num_workers - 1)];
    for (int i = 0; i < num_workers; ++i) {
        pool[i].srv        = srv;
        pool[i].ctrl_paths = ctrl_paths;
    }
    for (int i = 1; i < num_workers; ++i)
        ::pthread_create(&threads[i - 1], nullptr, thread_entry, &pool[i]);

    worker_run(pool[0]);
    return 0;
}
