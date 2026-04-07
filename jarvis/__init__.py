from pathlib import Path

RATE = 16000
DATA_DIR = Path(__file__).parent.parent / "data"
CLIPS_DIR = DATA_DIR / "clips"
CACHE_DIR = Path.home() / ".cache" / "jarvis"
TEMPLATES_DIR = CACHE_DIR / "templates"
MAX_TEMPLATE_FRAMES = 100  # ~2 seconds at 50 encoder frames/sec
ONSET_SKIP = 2             # skip first N encoder frames (Whisper "start of audio" artifact)
STEP_PENALTY = 0.1         # DTW non-diagonal transition penalty


def keyword_name(phrase: str) -> str:
    """Convert wake word phrase to a safe directory name: 'hey jarvis' -> 'hey_jarvis'"""
    import re
    name = re.sub(r"\s+", "_", phrase.lower().strip())
    name = re.sub(r"[^\w]", "", name)  # strip anything that isn't alphanumeric or _
    if not name:
        raise ValueError(f"Invalid wake word phrase: {phrase!r}")
    return name


def list_keyword_dirs() -> list[Path]:
    """List keyword subdirectories of CLIPS_DIR that contain .wav files."""
    if not CLIPS_DIR.exists():
        return []
    return sorted(
        d for d in CLIPS_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.wav"))
    )
