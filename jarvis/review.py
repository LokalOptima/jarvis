#!/usr/bin/env python3
"""Enrollment web UI — record, upload, review clips, and build templates.

    uv run python -m jarvis.review

Opens a browser to record wake words, upload audio files, review/delete clips,
and build templates for the C++ runtime.
"""

import io
import json
import tempfile
import wave
import webbrowser
from contextlib import redirect_stdout
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from jarvis import CLIPS_DIR

PORT = 8457

# Lazy-loaded Whisper Turbo model (several GB, only needed for processing)
_labeler = None


def _get_labeler():
    global _labeler
    if _labeler is None:
        import whisper as whisper_asr
        print("Loading Whisper Turbo...")
        _labeler = whisper_asr.load_model("turbo")
        print("Whisper Turbo loaded.")
    return _labeler


HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Jarvis Enrollment</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 0 20px;
         background: #1a1a2e; color: #eee; }
  h1 { color: #e94560; margin-bottom: 4px; }
  .subtitle { color: #888; margin-top: 0; font-size: 0.95em; }

  .wake-word { display: flex; align-items: center; gap: 8px; margin: 20px 0 16px; }
  .wake-word input { background: #16213e; color: #eee; border: 1px solid #333;
                     padding: 6px 12px; border-radius: 4px; font-size: 1em; width: 180px; }

  /* Record */
  .record-row { display: flex; align-items: center; gap: 16px; margin: 12px 0; }
  .rec-btn { width: 52px; height: 52px; border-radius: 50%; border: 3px solid #e94560;
             background: none; cursor: pointer; display: flex; align-items: center;
             justify-content: center; transition: all 0.2s; flex-shrink: 0; }
  .rec-btn:hover { background: rgba(233,69,96,0.15); }
  .rec-btn .dot { width: 20px; height: 20px; border-radius: 50%; background: #e94560;
                  transition: all 0.15s; }
  .rec-btn.on .dot { border-radius: 3px; width: 16px; height: 16px; }
  .rec-btn.on { border-color: #ff4444; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(233,69,96,0.4); }
                     50% { box-shadow: 0 0 0 10px rgba(233,69,96,0); } }
  .rec-text { color: #888; font-size: 0.9em; }
  .rec-time { color: #e94560; font-weight: bold; font-size: 0.9em; }

  /* Drop zone */
  .drop { border: 2px dashed #333; border-radius: 8px; padding: 20px; text-align: center;
          color: #555; margin: 12px 0; cursor: pointer; transition: all 0.2s; }
  .drop:hover, .drop.over { border-color: #e94560; color: #999; background: rgba(233,69,96,0.04); }
  .drop input { display: none; }
  .drop label { color: #e94560; cursor: pointer; text-decoration: underline; }

  /* Status */
  .status { margin: 10px 0; padding: 8px 12px; border-radius: 4px; font-size: 0.9em; display: none; }
  .status.vis { display: block; }
  .status.ok { background: #1a3a2a; color: #6ddb8a; }
  .status.err { background: #3a1a1a; color: #e94560; }
  .status.wait { background: #16213e; color: #aaa; }

  hr { border: none; border-top: 1px solid #333; margin: 28px 0 20px; }

  /* Clips */
  .toolbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
  .count { color: #888; }
  .actions { display: flex; gap: 8px; }
  .btn { background: #0f3460; color: white; border: none; padding: 7px 14px; border-radius: 4px;
         cursor: pointer; font-size: 0.9em; }
  .btn:hover { background: #1a4a7a; }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .btn.primary { background: #e94560; }
  .btn.primary:hover:not(:disabled) { background: #c73650; }

  .clip { display: flex; align-items: center; gap: 12px; padding: 8px 12px; margin: 4px 0;
          background: #16213e; border-radius: 6px; }
  .clip:hover { background: #1a2744; }
  .clip .name { font-family: monospace; min-width: 130px; font-size: 0.9em; }
  .clip .dur { color: #888; min-width: 42px; font-size: 0.85em; }
  .clip audio { height: 32px; flex: 1; }
  .del { background: #e94560; color: white; border: none; padding: 3px 10px; border-radius: 4px;
         cursor: pointer; font-size: 0.85em; }
  .del:hover { background: #c73650; }
  .empty { color: #555; padding: 32px; text-align: center; }
  .session { margin-top: 14px; padding: 4px 0; color: #555; font-size: 0.8em; }
</style>
</head>
<body>
<h1>Jarvis</h1>
<p class="subtitle">Enrollment &amp; Clip Review</p>

<div class="wake-word">
  <span>Wake word:</span>
  <input type="text" id="ww" value="hey jarvis" spellcheck="false">
</div>

<div class="record-row">
  <button class="rec-btn" id="rec-btn" onclick="toggleRec()"><div class="dot"></div></button>
  <div>
    <div class="rec-text" id="rec-label">Click to record</div>
    <div class="rec-time" id="rec-time"></div>
  </div>
</div>

<div class="drop" id="drop">
  Drop an audio file here or <label for="file-in">browse</label>
  <input type="file" id="file-in" accept="audio/*,.wav,.mp3,.webm,.ogg,.m4a,.flac">
</div>

<div class="status" id="st"></div>

<hr>

<div class="toolbar">
  <span class="count" id="count"></span>
  <div class="actions">
    <button class="btn" onclick="playAll()">Play All</button>
    <button class="btn primary" id="build-btn" onclick="build()">Build Templates</button>
  </div>
</div>
<div id="clips"></div>
<div class="status" id="bst"></div>

<script>
const $ = id => document.getElementById(id);
let rec = null, chunks = [], timer = null, t0 = 0;

function st(el, msg, cls) { el.textContent = msg; el.className = 'status vis ' + cls; }

// ---- Record ----
async function toggleRec() {
  const btn = $('rec-btn'), label = $('rec-label'), time = $('rec-time');
  if (rec && rec.state === 'recording') {
    rec.stop(); btn.classList.remove('on');
    label.textContent = 'Processing...'; time.textContent = '';
    clearInterval(timer); return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia(
      {audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
    rec = new MediaRecorder(stream);
    chunks = [];
    rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      if (!chunks.length) { label.textContent = 'No audio'; return; }
      label.textContent = 'Click to record';
      await process(new Blob(chunks, {type: rec.mimeType}), 'recording.webm');
    };
    rec.start(1000);
    btn.classList.add('on');
    t0 = Date.now();
    const tick = () => { time.textContent = Math.round((Date.now()-t0)/1000) + 's'; };
    tick(); timer = setInterval(tick, 500);
    label.textContent = 'Say "' + $('ww').value + '" repeatedly \u2014 click stop when done';
  } catch(e) { label.textContent = 'Mic denied'; st($('st'), e.message, 'err'); }
}

// ---- Drop / file ----
const drop = $('drop'), fin = $('file-in');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('over');
  if (e.dataTransfer.files.length) process(e.dataTransfer.files[0], e.dataTransfer.files[0].name);
});
fin.addEventListener('change', () => { if (fin.files.length) process(fin.files[0], fin.files[0].name); });

// ---- Process audio ----
async function process(blob, name) {
  st($('st'), 'Processing audio (model loads on first use)\u2026', 'wait');
  try {
    const r = await fetch('/api/process', {
      method: 'POST', body: blob,
      headers: {'X-Filename': name, 'X-Wake-Words': $('ww').value},
    });
    const d = await r.json();
    if (d.error) st($('st'), d.error, 'err');
    else if (!d.n_clips) st($('st'), 'No wake words found in audio', 'err');
    else { st($('st'), 'Extracted ' + d.n_clips + ' clip(s)', 'ok'); load(); }
  } catch(e) { st($('st'), e.message, 'err'); }
}

// ---- Clips ----
async function load() {
  const clips = await (await fetch('/api/clips')).json();
  const el = $('clips');
  $('count').textContent = clips.length + ' clips';
  $('build-btn').disabled = !clips.length;
  if (!clips.length) { el.innerHTML = '<div class="empty">No clips yet. Record or upload audio above.</div>'; return; }
  let html = '', prev = 0, sn = 0;
  for (const c of clips) {
    if (!prev || c.mtime - prev > 10) {
      sn++; html += `<div class="session">Session ${sn} \u2014 ${new Date(c.mtime*1000).toLocaleString()}</div>`;
    }
    prev = c.mtime;
    html += `<div class="clip" id="r-${c.name}">
      <span class="name">${c.name}</span><span class="dur">${c.duration}s</span>
      <audio controls preload="metadata" src="/clip/${c.name}"></audio>
      <button class="del" onclick="del_('${c.name}')">delete</button></div>`;
  }
  el.innerHTML = html;
}
async function del_(name) {
  if (!confirm('Delete ' + name + '?')) return;
  await fetch('/api/delete?name=' + encodeURIComponent(name), {method:'POST'});
  const row = $('r-'+name); if (row) row.remove();
  const n = document.querySelectorAll('.clip').length;
  $('count').textContent = n + ' clips';
  $('build-btn').disabled = !n;
}
async function playAll() {
  for (const a of document.querySelectorAll('audio')) { a.play(); await new Promise(r => a.onended = r); }
}

// ---- Build ----
async function build() {
  $('build-btn').disabled = true;
  st($('bst'), 'Building templates\u2026', 'wait');
  try {
    const d = await (await fetch('/api/build', {method:'POST'})).json();
    if (d.ok) { const lines = d.output.trim().split('\n'); st($('bst'), lines[lines.length-1].trim(), 'ok'); }
    else st($('bst'), d.error || 'Build failed', 'err');
  } catch(e) { st($('bst'), e.message, 'err'); }
  $('build-btn').disabled = false;
}

load();
</script>
</body>
</html>"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("", "/"):
            self._html(HTML)
        elif p == "/api/clips":
            self._clips()
        elif p.startswith("/clip/"):
            self._serve_clip(p[6:])
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        p = parsed.path
        if p == "/api/process":
            self._process()
        elif p == "/api/delete":
            self._delete(parsed)
        elif p == "/api/build":
            self._build()
        else:
            self.send_error(404)

    # --- helpers ---

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    # --- GET handlers ---

    def _clips(self):
        clips = []
        if CLIPS_DIR.exists():
            for f in sorted(CLIPS_DIR.glob("*.wav")):
                with wave.open(str(f), "rb") as wf:
                    dur = round(wf.getnframes() / wf.getframerate(), 1)
                clips.append({"name": f.name, "duration": dur, "mtime": f.stat().st_mtime})
        self._json(clips)

    def _serve_clip(self, name):
        fpath = CLIPS_DIR / name
        if fpath.exists() and fpath.resolve().parent == CLIPS_DIR.resolve():
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            self.wfile.write(fpath.read_bytes())
        else:
            self.send_error(404)

    # --- POST handlers ---

    def _delete(self, parsed):
        qs = parse_qs(parsed.query)
        name = qs.get("name", [""])[0]
        fpath = CLIPS_DIR / name
        if fpath.exists() and fpath.resolve().parent == CLIPS_DIR.resolve() and name.endswith(".wav"):
            fpath.unlink()
            self._json({"ok": True})
        else:
            self.send_error(404)

    def _process(self):
        from jarvis.enroll import process_audio, save_clips

        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._json({"error": "No audio data"})
            return

        body = self.rfile.read(length)
        filename = self.headers.get("X-Filename", "recording.webm")
        ext = Path(filename).suffix or ".webm"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(body)
            tmp_path = tmp.name

        try:
            model = _get_labeler()
            wake_words = self.headers.get("X-Wake-Words", "hey jarvis").lower().split()

            clips = process_audio(tmp_path, model, wake_words)
            if not clips:
                self._json({"n_clips": 0})
                return

            new_paths = save_clips(clips)
            self._json({"n_clips": len(new_paths), "clips": [p.name for p in new_paths]})
        except Exception as e:
            self._json({"error": str(e)})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _build(self):
        from jarvis.enroll import build_templates

        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                build_templates()
            self._json({"ok": True, "output": buf.getvalue()})
        except SystemExit:
            self._json({"ok": False, "error": buf.getvalue()})
        except Exception as e:
            self._json({"ok": False, "error": str(e)})


def main():
    clips = list(CLIPS_DIR.glob("*.wav")) if CLIPS_DIR.exists() else []
    print(f"Jarvis Enrollment UI")
    print(f"  {len(clips)} clips in {CLIPS_DIR}/")
    print(f"  http://localhost:{PORT}")
    webbrowser.open(f"http://localhost:{PORT}")
    HTTPServer(("", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
