#!/usr/bin/env python3
"""Enrollment web UI — record, trim, review clips, and build templates.

    uv run enroll

Opens a browser to record wake words one at a time, visually trim them
with wavesurfer.js, review/delete clips, and build templates for the
C++ runtime.  Supports multiple keywords via tabs.
"""

import io
import json
import shutil
import struct
import sys
import wave
import webbrowser
from contextlib import redirect_stdout
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from jarvis import (
    CLIPS_DIR, TEMPLATES_DIR, MAX_TEMPLATE_FRAMES, ONSET_SKIP,
    keyword_name, list_keyword_dirs,
)
from jarvis.dtw import cmvn, dba
from jarvis.features import extract_features_batch

PORT = 8457


# ---- Template build pipeline ----

def next_clip_index(keyword_dir: Path) -> int:
    """Find the next available clip index (handles gaps from deletions)."""
    existing = sorted(keyword_dir.glob("clip_*.wav"))
    if not existing:
        return 0
    last = existing[-1].stem  # "clip_0023"
    return int(last.split("_")[1]) + 1


def _build_keyword(keyword_dir: Path) -> bool:
    """Build a single keyword's template from its clips. Returns True on success."""
    keyword = keyword_dir.name
    wav_files = sorted(keyword_dir.glob("*.wav"))
    if not wav_files:
        return False

    print(f"\n--- {keyword} ({len(wav_files)} clips) ---")
    print(f"Extracting features...")

    all_features = extract_features_batch(wav_files)
    raw_features = []
    for i, (wav_path, features) in enumerate(zip(wav_files, all_features)):
        features = features[ONSET_SKIP:MAX_TEMPLATE_FRAMES]
        raw_features.append(features)
        if (i + 1) % 5 == 0 or i == 0 or i == len(wav_files) - 1:
            print(f"  [{i + 1}/{len(wav_files)}] {wav_path.name} ({features.shape[0]} frames)")

    # CMVN per clip, then DBA to merge all into 1 representative template
    print(f"Applying CMVN + DBA ({len(raw_features)} clips -> 1 template)...")
    cmvn_features = [cmvn(f) for f in raw_features]
    template = dba(cmvn_features, n_iter=5)

    # L2-normalize each frame so C++ can skip template norm computation
    norms = np.linalg.norm(template, axis=1, keepdims=True)
    template = template / np.maximum(norms, 1e-10)

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    # Binary format: int32 n_templates, per template: int32 n_frames, float[n_frames * 384]
    bin_path = TEMPLATES_DIR / f"{keyword}.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack("i", 1))  # 1 DBA template
        f.write(struct.pack("i", template.shape[0]))
        f.write(template.astype(np.float32).tobytes())

    print(f"  1 DBA template, {template.shape[0]} frames (384-dim, CMVN + L2-normalized)")
    print(f"  {bin_path} ({bin_path.stat().st_size} bytes)")
    return True


def build_templates():
    """Read clips for all keywords, extract features, build templates."""
    keyword_dirs = list_keyword_dirs()

    if not keyword_dirs:
        print(f"No clips in {CLIPS_DIR}/")
        print("Run 'uv run enroll' to record clips first.")
        sys.exit(1)

    print(f"Found {len(keyword_dirs)} keyword(s): {', '.join(d.name for d in keyword_dirs)}")

    built = 0
    for keyword_dir in keyword_dirs:
        if _build_keyword(keyword_dir):
            built += 1

    if built == 0:
        print("\nNo templates built.")
        sys.exit(1)

    print(f"\nBuilt {built} keyword template(s) in {TEMPLATES_DIR}/")


# ---- Web UI ----

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Enrollment</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui; max-width: 860px; margin: 40px auto; padding: 0 20px;
         background: #1a1a2e; color: #eee; }
  h1 { color: #e94560; margin-bottom: 4px; }
  .subtitle { color: #888; margin-top: 0; font-size: 0.95em; margin-bottom: 20px; }

  /* Keyword tabs */
  .kw-bar { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 20px; }
  .kw-tab { background: #0f3460; color: #eee; padding: 6px 14px; border-radius: 6px;
            font-size: 0.9em; cursor: pointer; border: 2px solid transparent;
            transition: all 0.15s; user-select: none; }
  .kw-tab:hover { border-color: #e94560; }
  .kw-tab.active { border-color: #e94560; background: #1a4a7a; }
  .kw-tab .x { margin-left: 6px; opacity: 0.5; font-size: 0.85em; }
  .kw-tab .x:hover { opacity: 1; color: #e94560; }
  .kw-add { display: flex; gap: 4px; }
  .kw-add input { background: #16213e; color: #eee; border: 1px solid #333;
                  padding: 5px 10px; border-radius: 4px; font-size: 0.9em; width: 150px; }
  .kw-add input::placeholder { color: #555; }
  .kw-add button { background: #0f3460; color: #eee; border: none; padding: 5px 10px;
                   border-radius: 4px; cursor: pointer; font-size: 0.9em; }
  .kw-add button:hover { background: #1a4a7a; }

  /* Recording */
  .record-section { margin: 16px 0; }
  .record-row { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }
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

  /* Waveform */
  .wave-box { background: #16213e; border-radius: 8px; padding: 12px; margin: 12px 0;
              min-height: 100px; display: none; }
  .wave-box.vis { display: block; }
  #waveform { width: 100%; }
  .wave-actions { display: flex; gap: 8px; margin-top: 10px; }

  /* Buttons */
  .btn { background: #0f3460; color: white; border: none; padding: 7px 14px; border-radius: 4px;
         cursor: pointer; font-size: 0.9em; }
  .btn:hover { background: #1a4a7a; }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .btn.primary { background: #e94560; }
  .btn.primary:hover:not(:disabled) { background: #c73650; }

  /* Status */
  .status { margin: 10px 0; padding: 8px 12px; border-radius: 4px; font-size: 0.9em; display: none; }
  .status.vis { display: block; }
  .status.ok { background: #1a3a2a; color: #6ddb8a; }
  .status.err { background: #3a1a1a; color: #e94560; }
  .status.wait { background: #16213e; color: #aaa; }

  hr { border: none; border-top: 1px solid #333; margin: 24px 0 20px; }

  /* Clips list */
  .toolbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
  .count { color: #888; }
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
</style>
</head>
<body>
<h1>Enrollment</h1>
<p class="subtitle">Record, trim, build templates</p>

<!-- Keyword tabs -->
<div class="kw-bar" id="kw-bar"></div>
<div class="kw-add">
  <input type="text" id="kw-input" placeholder="new keyword..." spellcheck="false"
         onkeydown="if(event.key==='Enter')addKeyword()">
  <button onclick="addKeyword()">+ Add</button>
</div>

<!-- Record -->
<div class="record-section">
  <div class="record-row">
    <button class="rec-btn" id="rec-btn" onclick="toggleRec()"><div class="dot"></div></button>
    <div>
      <div class="rec-text" id="rec-label">Select a keyword, then click to record</div>
      <div class="rec-time" id="rec-time"></div>
    </div>
  </div>
</div>

<!-- Waveform + trim -->
<div class="wave-box" id="wave-box">
  <div id="waveform"></div>
  <div class="wave-actions">
    <button class="btn" onclick="playRegion()">Play region</button>
    <button class="btn primary" onclick="saveClip()">Save clip</button>
    <button class="btn" onclick="discardClip()">Discard</button>
  </div>
</div>

<div class="status" id="st"></div>

<hr>

<!-- Clips list -->
<div class="toolbar">
  <span class="count" id="count"></span>
  <div style="display:flex;gap:8px;">
    <button class="btn" onclick="playAll()">Play All</button>
    <button class="btn primary" id="build-btn" onclick="build()" disabled>Build All Templates</button>
  </div>
</div>
<div id="clips"></div>
<div class="status" id="bst"></div>

<script type="module">
import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js';
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js';

const $ = id => document.getElementById(id);

// ---- State ----
let currentKw = null;    // active keyword name
let micStream = null;    // persistent mic stream (kept alive to avoid warm-up)
let rec = null;          // MediaRecorder
let chunks = [];
let timer = null;
let ws = null;           // wavesurfer instance
let regions = null;      // regions plugin
let trimRegion = null;   // the active trim region
let recordedBlob = null; // original recording blob (native quality)
let audioBuf = null;     // decoded AudioBuffer (native sample rate)

const MIC_CONSTRAINTS = {audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}};

async function ensureMic() {
  if (!micStream || !micStream.active)
    micStream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
  return micStream;
}

// ---- Helpers ----
function st(el, msg, cls) { el.textContent = msg; el.className = 'status vis ' + cls; }
function clearSt(el) { el.className = 'status'; el.textContent = ''; }

// ---- Keyword tabs ----
async function loadKeywords() {
  const kws = await (await fetch('/api/keywords')).json();
  const bar = $('kw-bar');
  bar.innerHTML = '';
  for (const kw of kws) {
    const tab = document.createElement('span');
    tab.className = 'kw-tab' + (kw === currentKw ? ' active' : '');
    tab.innerHTML = kw + '<span class="x" title="Delete keyword">&times;</span>';
    tab.addEventListener('click', (e) => {
      if (e.target.classList.contains('x')) { deleteKeyword(kw); return; }
      selectKw(kw);
    });
    bar.appendChild(tab);
  }
  // Auto-select first if none selected
  if (!currentKw && kws.length) selectKw(kws[0]);
  if (currentKw && !kws.includes(currentKw)) { currentKw = kws[0] || null; }
  updateRecLabel();
}

function selectKw(kw) {
  currentKw = kw;
  loadKeywords();
  loadClips();
}

async function addKeyword() {
  const input = $('kw-input');
  const name = input.value.trim();
  if (!name) return;
  const r = await fetch('/api/add-keyword', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name}),
  });
  const d = await r.json();
  if (d.error) { st($('st'), d.error, 'err'); return; }
  input.value = '';
  currentKw = d.keyword;
  await loadKeywords();
  loadClips();
}

async function deleteKeyword(kw) {
  if (!confirm('Delete keyword "' + kw + '" and all its clips?')) return;
  await fetch('/api/delete-keyword', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: kw}),
  });
  if (currentKw === kw) currentKw = null;
  await loadKeywords();
  loadClips();
}

function updateRecLabel() {
  const label = $('rec-label');
  if (!currentKw) label.textContent = 'Add a keyword first';
  else label.textContent = 'Click to record "' + currentKw.replace(/_/g, ' ') + '"';
}

// ---- Resampling ----
async function resampleRegion(mono, fromRate, toRate) {
  // mono: Float32Array at fromRate → returns Float32Array at toRate
  const outLen = Math.round(mono.length * toRate / fromRate);
  const offCtx = new OfflineAudioContext(1, outLen, toRate);
  const buf = offCtx.createBuffer(1, mono.length, fromRate);
  buf.getChannelData(0).set(mono);
  const src = offCtx.createBufferSource();
  src.buffer = buf;
  src.connect(offCtx.destination);
  src.start();
  const rendered = await offCtx.startRendering();
  return rendered.getChannelData(0);
}

// ---- WAV encoding ----
function encodeWAV(samples, sampleRate) {
  // samples: Float32Array [-1..1], sampleRate: 16000
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);
  // RIFF header
  writeStr(view, 0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(view, 8, 'WAVE');
  // fmt chunk
  writeStr(view, 12, 'fmt ');
  view.setUint32(16, 16, true);         // chunk size
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, sampleRate, true);  // sample rate
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);          // block align
  view.setUint16(34, 16, true);         // bits per sample
  // data chunk
  writeStr(view, 36, 'data');
  view.setUint32(40, numSamples * 2, true);
  for (let i = 0; i < numSamples; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return buffer;
}
function writeStr(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

// ---- wavesurfer ----
function initWavesurfer() {
  destroyWavesurfer();

  regions = RegionsPlugin.create();
  ws = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#4a6fa5',
    progressColor: '#4a6fa5',
    cursorColor: '#e94560',
    height: 80,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    plugins: [regions],
  });

  // Load original blob directly — no resampling for display/playback
  ws.loadBlob(recordedBlob);

  ws.on('ready', () => {
    const dur = ws.getDuration();
    // Auto-create trim region spanning most of the waveform
    const pad = Math.min(0.05, dur * 0.05);
    trimRegion = regions.addRegion({
      start: pad,
      end: dur - pad,
      color: 'rgba(233, 69, 96, 0.15)',
      drag: true,
      resize: true,
    });
  });

  $('wave-box').classList.add('vis');
}

function destroyWavesurfer() {
  if (ws) { ws.destroy(); ws = null; }
  regions = null;
  trimRegion = null;
  $('wave-box').classList.remove('vis');
}

// ---- Recording ----
async function toggleRec() {
  if (!currentKw) { st($('st'), 'Add a keyword first', 'err'); return; }
  const btn = $('rec-btn'), label = $('rec-label'), time = $('rec-time');

  if (rec && rec.state === 'recording') {
    rec.stop(); btn.classList.remove('on');
    label.textContent = 'Processing...'; time.textContent = '';
    clearInterval(timer); return;
  }

  // Discard previous waveform
  destroyWavesurfer();
  clearSt($('st'));

  try {
    rec = new MediaRecorder(await ensureMic());
    chunks = [];
    rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = async () => {
      if (!chunks.length) { label.textContent = 'No audio'; return; }
      try {
        recordedBlob = new Blob(chunks, {type: rec.mimeType});
        const ctx = new AudioContext();
        audioBuf = await ctx.decodeAudioData(await recordedBlob.arrayBuffer());
        initWavesurfer();
        updateRecLabel();
      } catch(e) {
        st($('st'), 'Decode error: ' + e.message, 'err');
        updateRecLabel();
      }
    };
    rec.start();
    btn.classList.add('on');
    const t0 = Date.now();
    const tick = () => { time.textContent = ((Date.now()-t0)/1000).toFixed(1) + 's'; };
    tick(); timer = setInterval(tick, 200);
    label.textContent = 'Say "' + currentKw.replace(/_/g, ' ') + '" once — click stop when done';
  } catch(e) { label.textContent = 'Mic denied'; st($('st'), e.message, 'err'); }
}

// ---- Trim actions ----
function playRegion() {
  if (!ws || !trimRegion) return;
  const end = trimRegion.end;
  const onTick = (t) => {
    if (t >= end) {
      ws.pause();
      ws.setTime(end);
      ws.un('timeupdate', onTick);
    }
  };
  ws.on('timeupdate', onTick);
  ws.setTime(trimRegion.start);
  ws.play();
}

async function saveClip() {
  if (!ws || !trimRegion || !audioBuf || !currentKw) return;
  const start = trimRegion.start;
  const end = trimRegion.end;
  const dur = end - start;

  if (dur < 0.2) {
    st($('st'), 'Clip too short (< 0.2s)', 'err'); return;
  }

  // Extract the selected region from the native-rate buffer
  const nativeSR = audioBuf.sampleRate;
  const startSample = Math.round(start * nativeSR);
  const endSample = Math.round(end * nativeSR);
  const regionLen = endSample - startSample;

  // Mix to mono
  const mono = new Float32Array(regionLen);
  for (let ch = 0; ch < audioBuf.numberOfChannels; ch++) {
    const chan = audioBuf.getChannelData(ch);
    for (let i = 0; i < regionLen; i++) mono[i] += chan[startSample + i];
  }
  if (audioBuf.numberOfChannels > 1) {
    for (let i = 0; i < regionLen; i++) mono[i] /= audioBuf.numberOfChannels;
  }

  // Resample to 16kHz
  const pcm16k = await resampleRegion(mono, nativeSR, 16000);
  const wavBuf = encodeWAV(pcm16k, 16000);
  st($('st'), 'Saving...', 'wait');
  try {
    const r = await fetch('/api/save-clip', {
      method: 'POST', body: wavBuf,
      headers: {'X-Keyword': currentKw, 'Content-Type': 'audio/wav'},
    });
    const d = await r.json();
    if (d.error) { st($('st'), d.error, 'err'); return; }
    st($('st'), 'Saved ' + d.name, 'ok');
    destroyWavesurfer();
    recordedBlob = null;
    audioBuf = null;
    loadClips();
  } catch(e) { st($('st'), e.message, 'err'); }
}

function discardClip() {
  destroyWavesurfer();
  recordedBlob = null;
  audioBuf = null;
  clearSt($('st'));
  updateRecLabel();
}

// ---- Clips list ----
async function loadClips() {
  if (!currentKw) {
    $('clips').innerHTML = '<div class="empty">No keyword selected.</div>';
    $('count').textContent = '';
    $('build-btn').disabled = true;
    return;
  }
  const clips = await (await fetch('/api/clips?keyword=' + encodeURIComponent(currentKw))).json();
  const el = $('clips');
  $('count').textContent = clips.length + ' clips (' + currentKw + ')';
  $('build-btn').disabled = !clips.length;
  if (!clips.length) {
    el.innerHTML = '<div class="empty">No clips yet. Record above.</div>';
    return;
  }
  let html = '';
  for (const c of clips) {
    html += '<div class="clip" id="r-' + c.name + '">' +
      '<span class="name">' + c.name + '</span>' +
      '<span class="dur">' + c.duration + 's</span>' +
      '<audio controls preload="metadata" src="/clip/' + currentKw + '/' + c.name + '"></audio>' +
      '<button class="del" onclick="window._del(\'' + c.name + '\')">delete</button></div>';
  }
  el.innerHTML = html;
}

window._del = async function(name) {
  if (!confirm('Delete ' + name + '?')) return;
  await fetch('/api/delete?keyword=' + encodeURIComponent(currentKw) + '&name=' + encodeURIComponent(name),
    {method: 'POST'});
  const row = $('r-' + name); if (row) row.remove();
  const n = document.querySelectorAll('.clip').length;
  $('count').textContent = n + ' clips (' + currentKw + ')';
  $('build-btn').disabled = !n;
};

async function playAll() {
  for (const a of document.querySelectorAll('.clip audio')) {
    a.play();
    await new Promise(r => a.onended = r);
  }
}

// ---- Build ----
async function build() {
  $('build-btn').disabled = true;
  st($('bst'), 'Building templates...', 'wait');
  try {
    const d = await (await fetch('/api/build', {method: 'POST'})).json();
    if (d.ok) {
      const lines = d.output.trim().split('\n');
      st($('bst'), lines[lines.length-1].trim(), 'ok');
    } else st($('bst'), d.error || 'Build failed', 'err');
  } catch(e) { st($('bst'), e.message, 'err'); }
  $('build-btn').disabled = false;
}

// Expose to onclick handlers
window.toggleRec = toggleRec;
window.playRegion = playRegion;
window.saveClip = saveClip;
window.discardClip = discardClip;
window.addKeyword = addKeyword;
window.playAll = playAll;
window.build = build;

// ---- Init ----
loadKeywords();
loadClips();
// Acquire mic early so it's warm by first recording
ensureMic().catch(() => {});
</script>
</body>
</html>"""


# ---- HTTP Server ----

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        p = parsed.path
        if p in ("", "/"):
            self._html(HTML)
        elif p == "/api/clips":
            self._clips(parsed)
        elif p == "/api/keywords":
            self._keywords()
        elif p.startswith("/clip/"):
            parts = p[6:].split("/", 1)
            if len(parts) == 2:
                self._serve_clip(parts[0], parts[1])
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        p = parsed.path
        if p == "/api/save-clip":
            self._save_clip()
        elif p == "/api/delete":
            self._delete(parsed)
        elif p == "/api/build":
            self._build()
        elif p == "/api/add-keyword":
            self._add_keyword()
        elif p == "/api/delete-keyword":
            self._delete_keyword()
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

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    # --- GET handlers ---

    @staticmethod
    def _safe_clip_path(kw: str, name: str) -> Path | None:
        """Resolve a clip path, returning None if it escapes CLIPS_DIR."""
        kw_dir = CLIPS_DIR / kw
        fpath = kw_dir / name
        if (fpath.exists()
                and fpath.resolve().parent == kw_dir.resolve()
                and kw_dir.resolve().parent == CLIPS_DIR.resolve()):
            return fpath
        return None

    def _keywords(self):
        # Include empty dirs (newly added keywords with no clips yet)
        if CLIPS_DIR.exists():
            kws = sorted(d.name for d in CLIPS_DIR.iterdir() if d.is_dir())
        else:
            kws = []
        self._json(kws)

    def _clips(self, parsed):
        qs = parse_qs(parsed.query)
        kw = qs.get("keyword", [""])[0]
        kw_dir = CLIPS_DIR / kw
        clips = []
        if kw and kw_dir.exists() and kw_dir.resolve().parent == CLIPS_DIR.resolve():
            for f in sorted(kw_dir.glob("*.wav")):
                with wave.open(str(f), "rb") as wf:
                    dur = round(wf.getnframes() / wf.getframerate(), 1)
                clips.append({"name": f.name, "duration": dur, "mtime": f.stat().st_mtime})
        self._json(clips)

    def _serve_clip(self, kw, name):
        fpath = self._safe_clip_path(kw, name)
        if fpath:
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            self.wfile.write(fpath.read_bytes())
        else:
            self.send_error(404)

    # --- POST handlers ---

    def _save_clip(self):
        """Save a trimmed WAV clip from the browser."""
        kw = self.headers.get("X-Keyword", "").strip()
        if not kw:
            self._json({"error": "No keyword specified"})
            return

        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._json({"error": "No audio data"})
            return

        wav_bytes = self.rfile.read(length)

        # Validate it's a WAV file
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    self._json({"error": f"Expected 16kHz mono 16-bit WAV, got {wf.getframerate()}Hz {wf.getnchannels()}ch {wf.getsampwidth()*8}bit"})
                    return
        except Exception as e:
            self._json({"error": f"Invalid WAV: {e}"})
            return

        kw_dir = CLIPS_DIR / kw
        kw_dir.mkdir(parents=True, exist_ok=True)

        # Validate path doesn't escape
        if kw_dir.resolve().parent != CLIPS_DIR.resolve():
            self._json({"error": "Invalid keyword"})
            return

        idx = next_clip_index(kw_dir)
        clip_path = kw_dir / f"clip_{idx:04d}.wav"
        clip_path.write_bytes(wav_bytes)

        total = len(list(kw_dir.glob("*.wav")))
        self._json({"ok": True, "name": clip_path.name, "total": total})

    def _delete(self, parsed):
        qs = parse_qs(parsed.query)
        kw = qs.get("keyword", [""])[0]
        name = qs.get("name", [""])[0]
        fpath = self._safe_clip_path(kw, name)
        if fpath and name.endswith(".wav"):
            fpath.unlink()
            self._json({"ok": True})
        else:
            self.send_error(404)

    def _add_keyword(self):
        data = self._read_json_body()
        name = data.get("name", "").strip()
        if not name:
            self._json({"error": "No name"})
            return
        try:
            kw = keyword_name(name)
        except ValueError as e:
            self._json({"error": str(e)})
            return
        kw_dir = CLIPS_DIR / kw
        if kw_dir.resolve().parent != CLIPS_DIR.resolve():
            self._json({"error": "Invalid name"})
            return
        kw_dir.mkdir(parents=True, exist_ok=True)
        self._json({"ok": True, "keyword": kw})

    def _delete_keyword(self):
        data = self._read_json_body()
        name = data.get("name", "").strip()
        if not name:
            self._json({"error": "No name"})
            return
        kw_dir = CLIPS_DIR / name
        if not kw_dir.exists() or kw_dir.resolve().parent != CLIPS_DIR.resolve():
            self._json({"error": "Not found"})
            return
        shutil.rmtree(kw_dir)
        # Also remove template if it exists
        tmpl = TEMPLATES_DIR / f"{name}.bin"
        if tmpl.exists():
            tmpl.unlink()
        self._json({"ok": True})

    def _build(self):
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
    total_clips = 0
    keywords = []
    for d in list_keyword_dirs():
        n = len(list(d.glob("*.wav")))
        keywords.append(f"{d.name} ({n})")
        total_clips += n
    print(f"Enrollment UI")
    if keywords:
        print(f"  {total_clips} clips: {', '.join(keywords)}")
    else:
        print(f"  No clips yet")
    print(f"  http://localhost:{PORT}")
    webbrowser.open(f"http://localhost:{PORT}")
    HTTPServer(("", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
