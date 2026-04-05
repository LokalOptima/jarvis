/**
 * net.h - Framed binary protocol for jarvis client/server.
 *
 * Wire format: 5-byte header (uint8 type, uint32 length LE) + payload.
 *
 * Client → Server:
 *   AUDIO (0x01): float32 PCM samples (200ms = 3200 samples = 12800 bytes)
 *
 * Server → Client:
 *   DETECT (0x81): null-terminated keyword name + float32 score
 *   STATUS (0x82): uint8 state (0=buffering, 1=ready, 2=cooldown)
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

// macOS doesn't have MSG_NOSIGNAL; it uses SO_NOSIGPIPE per-socket instead.
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

static constexpr int JARVIS_PORT = 7287;

// Message types
static constexpr uint8_t MSG_AUDIO    = 0x01;
static constexpr uint8_t MSG_DETECT   = 0x81;
static constexpr uint8_t MSG_STATUS   = 0x82;
static constexpr uint8_t MSG_RESPONSE = 0x83;  // text (uint32 len + chars) followed by WAV audio

// Status codes
static constexpr uint8_t STATUS_BUFFERING = 0;
static constexpr uint8_t STATUS_READY     = 1;

struct MsgHeader {
    uint8_t  type;
    uint32_t length;
} __attribute__((packed));

// ---- Low-level I/O ----

static inline bool send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        p += n;
        len -= n;
    }
    return true;
}

static inline bool recv_all(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    while (len > 0) {
        ssize_t n = ::recv(fd, p, len, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        p += n;
        len -= n;
    }
    return true;
}

// ---- Framed messages ----

static inline bool send_msg(int fd, uint8_t type, const void *data, uint32_t len) {
    MsgHeader hdr = { type, len };
    return send_all(fd, &hdr, sizeof(hdr)) && (len == 0 || send_all(fd, data, len));
}

static inline bool recv_msg(int fd, MsgHeader &hdr, std::vector<uint8_t> &payload) {
    if (!recv_all(fd, &hdr, sizeof(hdr))) return false;
    payload.resize(hdr.length);
    if (hdr.length > 0 && !recv_all(fd, payload.data(), hdr.length)) return false;
    return true;
}

// ---- Socket helpers ----

static inline int tcp_listen(int port) {
    int fd = socket(AF_INET6, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    // Dual-stack: accept IPv4 and IPv6
    int off = 0;
    setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    struct sockaddr_in6 addr = {};
    addr.sin6_family = AF_INET6;
    addr.sin6_port = htons(port);
    addr.sin6_addr = in6addr_any;

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { close(fd); return -1; }
    if (listen(fd, 1) < 0) { close(fd); return -1; }
    return fd;
}

static inline int tcp_connect(const char *host, int port) {
    struct addrinfo hints = {}, *res;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    if (getaddrinfo(host, port_str, &hints, &res) != 0) return -1;

    int fd = -1;
    for (struct addrinfo *rp = res; rp; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);

    if (fd >= 0) {
        int opt = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#ifdef SO_NOSIGPIPE
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &opt, sizeof(opt));
#endif
    }
    return fd;
}
