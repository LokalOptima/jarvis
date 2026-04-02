#!/usr/bin/env python3
"""Web UI to review and manage enrolled clips.

    uv run python -m jarvis.review                # show all clips
    uv run python -m jarvis.review --only clip_0019.wav clip_0020.wav  # show specific clips

Opens a browser with audio players for each clip. Delete bad ones, then
run 'python -m jarvis.enroll --build' to regenerate templates.
"""

import argparse
import json
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

CLIPS_DIR = Path(__file__).parent.parent / "data" / "clips"
PORT = 8457

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Jarvis Clip Review</title>
<style>
  body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #1a1a2e; color: #eee; }
  h1 { color: #e94560; }
  .clip { display: flex; align-items: center; gap: 12px; padding: 8px 12px; margin: 4px 0; background: #16213e; border-radius: 6px; }
  .clip:hover { background: #1a2744; }
  .name { font-family: monospace; min-width: 140px; }
  .dur { color: #888; min-width: 50px; font-size: 0.9em; }
  audio { height: 32px; flex: 1; }
  .del { background: #e94560; color: white; border: none; padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 0.9em; }
  .del:hover { background: #c73650; }
  .count { color: #888; margin-bottom: 20px; }
  .empty { color: #888; padding: 40px; text-align: center; }
  .actions { margin: 20px 0; display: flex; gap: 10px; }
  .btn { background: #0f3460; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
  .btn:hover { background: #1a4a7a; }
  .session-label { margin-top: 20px; padding: 6px 12px; color: #0f3460; background: #e94560; border-radius: 4px; font-size: 0.85em; font-weight: bold; display: inline-block; }
</style>
</head>
<body>
<h1>Clip Review</h1>
<div class="count" id="count"></div>
<div class="actions">
  <button class="btn" onclick="playAll()">Play All</button>
  <button class="btn" onclick="location.reload()">Refresh</button>
</div>
<div id="clips"></div>
<script>
async function load() {
  const res = await fetch('/api/clips');
  const clips = await res.json();
  const el = document.getElementById('clips');
  document.getElementById('count').textContent = clips.length + ' clips';
  if (!clips.length) { el.innerHTML = '<div class="empty">No clips. Run enroll first.</div>'; return; }

  // Group by session: clips within 10s of each other
  let html = '';
  let sessionStart = clips[0].mtime;
  let sessionNum = 1;
  const ts = t => new Date(t * 1000).toLocaleString();
  html += `<div class="session-label">Session ${sessionNum} — ${ts(sessionStart)}</div>`;
  for (let i = 0; i < clips.length; i++) {
    const c = clips[i];
    if (i > 0 && c.mtime - clips[i-1].mtime > 10) {
      sessionNum++;
      sessionStart = c.mtime;
      html += `<div class="session-label">Session ${sessionNum} — ${ts(sessionStart)}</div>`;
    }
    html += `
      <div class="clip" id="row-${c.name}">
        <span class="name">${c.name}</span>
        <span class="dur">${c.duration}s</span>
        <audio controls preload="metadata" src="/clip/${c.name}"></audio>
        <button class="del" onclick="del_clip('${c.name}')">delete</button>
      </div>`;
  }
  el.innerHTML = html;
}
async function del_clip(name) {
  if (!confirm('Delete ' + name + '?')) return;
  await fetch('/api/delete?name=' + name, {method: 'POST'});
  document.getElementById('row-' + name).remove();
  const el = document.getElementById('count');
  const n = document.querySelectorAll('.clip').length;
  el.textContent = n + ' clips';
}
async function playAll() {
  const audios = document.querySelectorAll('audio');
  for (const a of audios) {
    a.play();
    await new Promise(r => a.onended = r);
  }
}
load();
</script>
</body>
</html>"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence logs

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif parsed.path == "/api/clips":
            clips = []
            if CLIPS_DIR.exists():
                for f in sorted(CLIPS_DIR.glob("*.wav")):
                    import wave
                    with wave.open(str(f), "rb") as wf:
                        dur = round(wf.getnframes() / wf.getframerate(), 1)
                    mtime = f.stat().st_mtime
                    clips.append({"name": f.name, "duration": dur, "mtime": mtime})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(clips).encode())

        elif parsed.path.startswith("/clip/"):
            name = parsed.path[6:]
            fpath = CLIPS_DIR / name
            if fpath.exists() and fpath.parent == CLIPS_DIR:
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
            else:
                self.send_error(404)

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/delete":
            qs = parse_qs(parsed.query)
            name = qs.get("name", [""])[0]
            fpath = CLIPS_DIR / name
            if fpath.exists() and fpath.parent == CLIPS_DIR and name.endswith(".wav"):
                fpath.unlink()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')
            else:
                self.send_error(404)
        else:
            self.send_error(404)


def main():
    if not CLIPS_DIR.exists() or not list(CLIPS_DIR.glob("*.wav")):
        print(f"No clips in {CLIPS_DIR}/")
        print("Run 'python -m jarvis.enroll' first.")
        return

    n = len(list(CLIPS_DIR.glob("*.wav")))
    print(f"{n} clips in {CLIPS_DIR}/")
    print(f"http://localhost:{PORT}")

    webbrowser.open(f"http://localhost:{PORT}")
    HTTPServer(("", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
