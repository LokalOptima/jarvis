#include "playback.h"
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

void play_wav(const uint8_t *data, size_t size, bool wait) {
    char tmp[] = "/tmp/jarvis-XXXXXX.wav";
    int fd = mkstemps(tmp, 4);
    if (fd < 0) return;
    if (write(fd, data, size) < 0) { ::close(fd); return; }
    ::close(fd);
    if (!wait) {
        // Double-fork so nobody waits. Grandchild plays + cleans up.
        pid_t pid = fork();
        if (pid == 0) {
            if (fork() != 0) _exit(0);  // child exits, grandchild continues
#ifdef __APPLE__
            execlp("sh", "sh", "-c", (std::string("afplay ") + tmp + "; rm -f " + tmp).c_str(), nullptr);
#else
            execlp("sh", "sh", "-c", (std::string("aplay -q ") + tmp + " 2>/dev/null || paplay " + tmp + " 2>/dev/null; rm -f " + tmp).c_str(), nullptr);
#endif
            _exit(127);
        }
        if (pid > 0) waitpid(pid, nullptr, 0);  // reap child (instant)
    } else {
        pid_t pid = fork();
        if (pid == 0) {
#ifdef __APPLE__
            execlp("afplay", "afplay", tmp, nullptr);
#else
            execlp("aplay", "aplay", "-q", tmp, nullptr);
            execlp("paplay", "paplay", tmp, nullptr);
#endif
            _exit(127);
        }
        if (pid > 0) {
            waitpid(pid, nullptr, 0);
            unlink(tmp);
        }
    }
}
