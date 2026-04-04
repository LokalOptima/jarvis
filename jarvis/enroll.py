#!/usr/bin/env python3
"""Enroll wake word templates. Two steps:

Step 1 — Record and extract clips:

    uv run python -m jarvis.enroll                      # live mic
    uv run python -m jarvis.enroll recording.wav         # from file

Step 2 — Build templates from clips:

    uv run python -m jarvis.enroll --build

    Reads data/clips/<keyword>/*.wav, extracts Whisper features,
    saves models/templates/<keyword>.bin for each keyword.

You can delete bad clips from data/clips/<keyword>/ between steps.
"""

import argparse
import struct
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import whisper as whisper_asr

from jarvis import (
    RATE, DATA_DIR, CLIPS_DIR, TEMPLATES_DIR,
    MAX_TEMPLATE_FRAMES, ONSET_SKIP, keyword_name, list_keyword_dirs,
)
from jarvis.dtw import cmvn, dba
from jarvis.features import extract_features

CLIP_PAD_SEC = 0.3   # padding around detected wake word boundaries
MIN_CLIP_SEC = 0.4   # minimum clip length to keep
SILENCE_MARGIN = int(RATE * 0.05)  # 50ms margin when trimming silence


def find_wake_words(
    audio: str | np.ndarray,
    model: whisper_asr.Whisper,
    wake_words: list[str],
) -> list[tuple[float, float]]:
    """Find wake word timestamps in audio. Returns (start, end) pairs.

    Args:
        audio: path to audio file, or float32 numpy array (mono, 16kHz).
    """
    result = model.transcribe(
        audio, word_timestamps=True, language="en", no_speech_threshold=0.5,
    )

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({
                "text": w["word"].strip().lower().strip(".,!?"),
                "start": w["start"],
                "end": w["end"],
            })

    detections = []
    n_wake = len(wake_words)
    for i in range(len(words) - n_wake + 1):
        window = [words[i + j]["text"] for j in range(n_wake)]
        if window == wake_words:
            detections.append((words[i]["start"], words[i + n_wake - 1]["end"]))

    return detections


def trim_silence(audio: np.ndarray, margin: int = SILENCE_MARGIN) -> np.ndarray:
    """Trim leading/trailing silence, keeping a small margin."""
    energy = np.abs(audio).astype(np.float32)
    win = int(RATE * 0.03)  # 30ms smoothing window
    if len(energy) <= win:
        return audio
    smoothed = np.convolve(energy, np.ones(win) / win, mode="same")
    threshold = max(smoothed.max() * 0.08, 50)
    above = np.where(smoothed > threshold)[0]
    if len(above) == 0:
        return audio
    start = max(0, above[0] - margin)
    end = min(len(audio), above[-1] + margin)
    return audio[start:end]


def extract_clips_from_audio(
    audio: np.ndarray, detections: list[tuple[float, float]],
) -> list[np.ndarray]:
    """Extract int16 audio clips around each detection, trimmed."""
    clips = []
    for start, end in detections:
        s = max(0, int((start - CLIP_PAD_SEC) * RATE))
        e = min(len(audio), int((end + CLIP_PAD_SEC) * RATE))
        clip = trim_silence(audio[s:e])
        if len(clip) >= int(MIN_CLIP_SEC * RATE):
            clips.append(clip)
    return clips


def save_wav(path: Path, audio: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio.tobytes())


def next_clip_index(keyword_dir: Path) -> int:
    """Find the next available clip index (handles gaps from deletions)."""
    existing = sorted(keyword_dir.glob("clip_*.wav"))
    if not existing:
        return 0
    last = existing[-1].stem  # "clip_0023"
    return int(last.split("_")[1]) + 1


def save_clips(clips: list[np.ndarray], keyword: str) -> list[Path]:
    """Save extracted clips to the keyword's clip directory. Returns paths of new clips."""
    kw_dir = CLIPS_DIR / keyword
    kw_dir.mkdir(parents=True, exist_ok=True)

    idx = next_clip_index(kw_dir)
    new_paths = []
    for i, clip in enumerate(clips):
        p = kw_dir / f"clip_{idx + i:04d}.wav"
        save_wav(p, clip)
        new_paths.append(p)

    total = len(list(kw_dir.glob("*.wav")))
    print(f"  {len(clips)} new clips saved to {kw_dir}/")
    print(f"  Total clips for '{keyword}': {total}")
    return new_paths


def _build_keyword(keyword_dir: Path) -> bool:
    """Build a single keyword's template from its clips. Returns True on success."""
    keyword = keyword_dir.name
    wav_files = sorted(keyword_dir.glob("*.wav"))
    if not wav_files:
        return False

    print(f"\n--- {keyword} ({len(wav_files)} clips) ---")
    print(f"Extracting features...")

    raw_features = []
    for i, wav_path in enumerate(wav_files):
        with wave.open(str(wav_path), "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        features = extract_features(audio)
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
    # Migrate old flat clips dir (data/clips/*.wav) to data/clips/hey_jarvis/
    old_clips = sorted(CLIPS_DIR.glob("*.wav")) if CLIPS_DIR.exists() else []
    if old_clips:
        dest = CLIPS_DIR / "hey_jarvis"
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Migrating {len(old_clips)} clips to {dest}/...")
        for f in old_clips:
            f.rename(dest / f.name)

    keyword_dirs = list_keyword_dirs()

    if not keyword_dirs:
        print(f"No clips in {CLIPS_DIR}/")
        print("Run 'uv run python -m jarvis.enroll' first to record and extract clips.")
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


# ---- Live recording + extraction ----

def live_enroll(wake_words: list[str], device: int | None = None):
    """Record from mic, show live detections, extract on Ctrl+C."""
    import sounddevice as sd

    keyword = keyword_name(" ".join(wake_words))
    print("Loading Whisper Turbo...")
    labeler = whisper_asr.load_model("turbo")

    phrase = " ".join(wake_words)
    print(f'\nSay "{phrase}" repeatedly. Ctrl+C when done.\n')

    chunk_sec = 3.0
    overlap_samples = int(RATE * 1.0)

    all_recorded: list[np.ndarray] = []
    recording = True
    audio_buffer = np.array([], dtype=np.int16)

    chunks_queue: list[np.ndarray] = []
    lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        if recording:
            with lock:
                chunks_queue.append(indata.copy().squeeze())

    stream = sd.InputStream(
        samplerate=RATE, channels=1, dtype="int16",
        device=device, callback=audio_callback, blocksize=int(RATE * 0.1),
    )
    stream.start()

    preview_count = 0
    try:
        while True:
            time.sleep(chunk_sec)

            with lock:
                if not chunks_queue:
                    continue
                new_audio = np.concatenate(chunks_queue)
                chunks_queue.clear()

            all_recorded.append(new_audio)
            audio_buffer = np.concatenate([audio_buffer[-overlap_samples:], new_audio])

            # Live feedback: write temp WAV for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                save_wav(Path(tmp_path), audio_buffer)

            detections = find_wake_words(tmp_path, labeler, wake_words)
            Path(tmp_path).unlink()

            for _ in detections:
                preview_count += 1
                print(f"  ~{preview_count} detected")

    except KeyboardInterrupt:
        pass
    finally:
        recording = False
        stream.stop()
        stream.close()

    if not all_recorded:
        print("\nNo audio recorded.")
        sys.exit(1)

    full_audio = np.concatenate(all_recorded)
    print(f"\nRecorded {len(full_audio)/RATE:.1f}s. Running full detection...")

    # Final detection on complete audio
    rec_path = DATA_DIR / "recording.wav"
    save_wav(rec_path, full_audio)

    detections = find_wake_words(str(rec_path), labeler, wake_words)
    print(f"  {len(detections)} wake words found:")
    for start, end in detections:
        print(f"    {start:.2f}s - {end:.2f}s")

    if not detections:
        print("\nNo wake words found in recording.")
        sys.exit(1)

    clips = extract_clips_from_audio(full_audio, detections)
    save_clips(clips, keyword)
    print(f"\nRun 'uv run python -m jarvis.enroll --build' to generate templates.")


def process_audio(
    audio_path: str,
    model: whisper_asr.Whisper,
    wake_words: list[str],
) -> list[np.ndarray]:
    """Detect wake words in an audio file and return extracted clips.

    Loads audio once and passes the float32 array to both transcription and
    clip extraction, avoiding a redundant ffmpeg decode.
    """
    audio = whisper_asr.load_audio(audio_path)
    detections = find_wake_words(audio, model, wake_words)
    if not detections:
        return []
    audio_int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
    return extract_clips_from_audio(audio_int16, detections)


# ---- File-based extraction ----

def file_enroll(audio_files: list[str], wake_words: list[str]):
    """Extract clips from pre-recorded audio files."""
    keyword = keyword_name(" ".join(wake_words))
    print("Loading Whisper Turbo...")
    labeler = whisper_asr.load_model("turbo")

    all_clips = []
    for audio_path in audio_files:
        p = Path(audio_path)
        if not p.exists():
            print(f"  Skipping: {audio_path}")
            continue

        print(f"\n  {p.name}:")
        clips = process_audio(str(p), labeler, wake_words)
        print(f"    {len(clips)} clips extracted")
        all_clips.extend(clips)

    if not all_clips:
        print("\nNo wake words found.")
        sys.exit(1)

    save_clips(all_clips, keyword)
    print(f"\nRun 'uv run python -m jarvis.enroll --build' to generate templates.")


def main():
    parser = argparse.ArgumentParser(description="Enroll wake word templates")
    parser.add_argument("audio", nargs="*", help="Audio file(s) — omit for live mic")
    parser.add_argument("--build", action="store_true", help="Build templates from clips")
    parser.add_argument(
        "--wake-word", type=str, default="hey jarvis",
        help="Wake word phrase (default: 'hey jarvis')",
    )
    parser.add_argument("--device", type=int, default=None, help="Audio input device")
    args = parser.parse_args()

    if args.build:
        build_templates()
        return

    wake_words = args.wake_word.lower().split()

    if args.audio:
        file_enroll(args.audio, wake_words)
    else:
        live_enroll(wake_words, device=args.device)


if __name__ == "__main__":
    main()
