"""Global configuration for DEPTHMAP AI.

Everything that more than one module needs (paths, defaults, method names)
lives here so there is exactly one place to change it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "depthnet.pt"
RESULTS_DIR = ROOT / "results"

APP_NAME = "DEPTHMAP AI"
TAGLINE = "Beyond Pixels. Preserving Geometry."

SCALES = (2, 4, 8)

# Display name -> internal key
METHODS = {
    "Nearest neighbour": "nearest",
    "Bilinear": "bilinear",
    "Bicubic": "bicubic",
    "Guided filtering": "guided",
    "Joint bilateral filtering": "jbu",
    "Deep learning (DepthNet)": "deep",
    "Hybrid reconstruction": "hybrid",
}
METHOD_LABELS = {v: k for k, v in METHODS.items()}

# Upload limits (pixels) keep the hosted app responsive (rubric: ~10 s).
MAX_OUTPUT_PIXELS = 1280 * 960
MAX_UPLOAD_MB = 25

# Default depth encoding for integer depth files: raw value per metre.
# NYU / Kinect / Middlebury-style 16-bit PNGs are usually millimetres (1000).
DEFAULT_UNITS_PER_METRE = 1000.0
