from pathlib import Path

RATE = 16000
DATA_DIR = Path(__file__).parent.parent / "data"
CLIPS_DIR = DATA_DIR / "clips"
MODELS_DIR = Path(__file__).parent.parent / "models"
MAX_TEMPLATE_FRAMES = 100  # ~2 seconds at 50 encoder frames/sec
