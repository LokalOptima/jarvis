#!/usr/bin/env python3
"""Enroll wake word templates. Two steps:

Step 1 — Record and extract clips:

    uv run python -m jarvis.enroll                      # live mic
    uv run python -m jarvis.enroll recording.wav         # from file

    Saves to data/enrollment/recording.wav and data/enrollment/clips/

Step 2 — Build templates from clips:

    uv run python -m jarvis.enroll --build

    Reads data/enrollment/clips/*.wav, extracts Whisper features,
    saves models/templates.bin

You can delete bad clips from data/enrollment/clips/ between steps.
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

from jarvis.features import extract_features_from_audio

RATE = 16000
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
CLIPS_DIR = DATA_DIR / "clips"


def find_wake_words(
    audio_path: str,
    model: whisper_asr.Whisper,
    wake_words: list[str],
) -> list[tuple[float, float]]:
    """Find wake word timestamps in audio. Returns (start, end) pairs."""
    result = model.transcribe(
        audio_path, word_timestamps=True, language="en", no_speech_threshold=0.5,
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


def trim_silence(audio: np.ndarray, margin: int = 800) -> np.ndarray:
    """Trim leading/trailing silence, keeping a small margin (default 50ms)."""
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
        s = max(0, int((start - 0.3) * RATE))
        e = min(len(audio), int((end + 0.3) * RATE))
        clip = trim_silence(audio[s:e])
        if len(clip) >= int(0.4 * RATE):
            clips.append(clip)
    return clips


def save_wav(path: Path, audio: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio.tobytes())


def next_clip_index() -> int:
    """Find the next available clip index (handles gaps from deletions)."""
    existing = sorted(CLIPS_DIR.glob("clip_*.wav"))
    if not existing:
        return 0
    # Parse highest index and add 1
    last = existing[-1].stem  # "clip_0023"
    return int(last.split("_")[1]) + 1


def save_clips(clips: list[np.ndarray], full_audio: np.ndarray | None) -> list[Path]:
    """Save recording and extracted clips. Returns paths of new clips."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    if full_audio is not None and len(full_audio) > 0:
        rec_path = DATA_DIR / "recording.wav"
        save_wav(rec_path, full_audio)
        print(f"  Recording: {rec_path} ({len(full_audio)/RATE:.1f}s)")

    idx = next_clip_index()
    new_paths = []
    for i, clip in enumerate(clips):
        p = CLIPS_DIR / f"clip_{idx + i:04d}.wav"
        save_wav(p, clip)
        new_paths.append(p)

    total = len(list(CLIPS_DIR.glob("*.wav")))
    print(f"  {len(clips)} new clips saved to {CLIPS_DIR}/")
    print(f"  Total clips: {total}")
    return new_paths


def build_templates():
    """Read clips, extract full frame sequences, save templates."""
    wav_files = sorted(CLIPS_DIR.glob("*.wav"))
    if not wav_files:
        print(f"No clips in {CLIPS_DIR}/")
        print("Run 'python -m jarvis.enroll' first to record and extract clips.")
        sys.exit(1)

    print(f"Building templates from {len(wav_files)} clips...")

    templates = []  # list of [T_i, 384] arrays
    for i, wav_path in enumerate(wav_files):
        with wave.open(str(wav_path), "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        audio_f32 = audio.astype(np.float32) / 32768.0
        features = extract_features_from_audio(audio_f32)  # [T, 384]
        templates.append(features[:100])
        if (i + 1) % 5 == 0 or i == 0 or i == len(wav_files) - 1:
            print(f"  [{i + 1}/{len(wav_files)}] {wav_path.name} ({features.shape[0]} frames)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Binary format: int32 n_templates, then per template: int32 n_frames, float[n_frames * 384]
    bin_path = MODELS_DIR / "templates.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack("i", len(templates)))
        for feat in templates:
            f.write(struct.pack("i", feat.shape[0]))
            f.write(feat.astype(np.float32).tobytes())

    total_frames = sum(t.shape[0] for t in templates)
    print(f"\n  {len(templates)} templates, {total_frames} total frames (384-dim)")
    print(f"  {bin_path} ({bin_path.stat().st_size} bytes)")


# ---- Live recording + extraction ----

def live_enroll(wake_words: list[str], device: int | None = None):
    """Record from mic, show live detections, extract on Ctrl+C."""
    import sounddevice as sd

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
        device=device, callback=audio_callback, blocksize=1600,
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

            # Live feedback only
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                with wave.open(tmp_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(RATE)
                    wf.writeframes(audio_buffer.tobytes())

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
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(rec_path, full_audio)

    detections = find_wake_words(str(rec_path), labeler, wake_words)
    print(f"  {len(detections)} wake words found:")
    for start, end in detections:
        print(f"    {start:.2f}s - {end:.2f}s")

    if not detections:
        print("\nNo wake words found in recording.")
        sys.exit(1)

    clips = extract_clips_from_audio(full_audio, detections)
    save_clips(clips, full_audio)
    print(f"\nRun 'python -m jarvis.enroll --build' to generate templates.")


# ---- File-based extraction ----

def file_enroll(audio_files: list[str], wake_words: list[str]):
    """Extract clips from pre-recorded audio files."""
    print("Loading Whisper Turbo...")
    labeler = whisper_asr.load_model("turbo")

    all_clips = []
    for audio_path in audio_files:
        p = Path(audio_path)
        if not p.exists():
            print(f"  Skipping: {audio_path}")
            continue

        print(f"\n  {p.name}:")
        detections = find_wake_words(str(p), labeler, wake_words)
        print(f"    {len(detections)} detections")
        for start, end in detections:
            print(f"      {start:.2f}s - {end:.2f}s")

        audio = whisper_asr.load_audio(str(p))
        audio_int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
        clips = extract_clips_from_audio(audio_int16, detections)
        all_clips.extend(clips)

    if not all_clips:
        print("\nNo wake words found.")
        sys.exit(1)

    save_clips(all_clips, None)
    print(f"\nRun 'python -m jarvis.enroll --build' to generate templates.")


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
