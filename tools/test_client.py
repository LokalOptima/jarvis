#!/usr/bin/env python3
"""Test client for jarvis detection server (Unix socket or TCP)."""

import argparse
import json
import socket
import struct
import sys
import wave
import tempfile


def connect(addr: str) -> socket.socket:
    """Connect to server. addr is a path (Unix) or tcp:host:port / tcp:port."""
    if addr.startswith("tcp:"):
        rest = addr[4:]
        if ":" in rest:
            host, port = rest.rsplit(":", 1)
        else:
            host, port = "127.0.0.1", rest
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, int(port)))
    else:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(addr)
    return sock


def main():
    p = argparse.ArgumentParser(description="Jarvis test client")
    p.add_argument("keywords", nargs="*", default=["hey_jarvis", "weather"],
                   help="Keywords to subscribe to (default: hey_jarvis weather)")
    p.add_argument("--addr", default="/tmp/jarvis.sock",
                   help="Server address: /path/to/sock or tcp:host:port (default: /tmp/jarvis.sock)")
    p.add_argument("--save-audio", action="store_true",
                   help="Save voice recordings as WAV files")
    args = p.parse_args()

    sock = connect(args.addr)

    sub = json.dumps({"subscribe": args.keywords}) + "\n"
    sock.sendall(sub.encode())
    print(f"subscribed: {args.keywords}")

    buf = b""
    pcm_remaining = 0
    pcm_event = None

    try:
        while True:
            data = sock.recv(4096)
            if not data:
                print("server closed connection")
                break

            buf += data

            # Reading PCM payload
            while pcm_remaining > 0 and len(buf) >= pcm_remaining:
                pcm_bytes = buf[:pcm_remaining]
                buf = buf[pcm_remaining:]
                pcm_remaining = 0

                n_samples = len(pcm_bytes) // 4
                samples = struct.unpack(f"{n_samples}f", pcm_bytes)
                duration = n_samples / 16000

                print(f"  audio: {n_samples} samples ({duration:.1f}s)")

                if args.save_audio:
                    path = tempfile.mktemp(suffix=".wav", prefix="jarvis_")
                    with wave.open(path, "w") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(16000)
                        pcm16 = b"".join(
                            struct.pack("<h", max(-32768, min(32767, int(s * 32767))))
                            for s in samples
                        )
                        w.writeframes(pcm16)
                    print(f"  saved: {path}")

            # Process JSON lines
            while b"\n" in buf and pcm_remaining == 0:
                line, buf = buf.split(b"\n", 1)
                if not line:
                    continue

                evt = json.loads(line)
                kw = evt["keyword"]
                score = evt["score"]

                if "audio_length" in evt:
                    n = evt["audio_length"]
                    pcm_remaining = n * 4  # float32
                    pcm_event = evt
                    print(f"[{kw}] score={score:.2f} (voice, waiting for {n} samples)")
                else:
                    print(f"[{kw}] score={score:.2f}")

    except KeyboardInterrupt:
        print()
    finally:
        sock.close()

if __name__ == "__main__":
    main()
