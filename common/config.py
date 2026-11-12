"""Single source of truth. Override paths with env vars, never edit constants per person."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get(
    "DS6G_ROOT", "/kaggle/input/datasets/haythemkrid/deepsense6g-scenario-33/scenario33"))
CSV_NAME = "scenario33_dev.csv"
WORK = Path(os.environ.get("DS6G_WORK", "/kaggle/working/ds6g"))   # master.csv + cache/ live here
CACHE = WORK / "cache"
RESULTS = Path(os.environ.get("DS6G_RESULTS", str(ROOT / "results")))
SPLIT_FILE = ROOT / "splits" / "splits.json"                      # committed in repo, built once by M1

N_BEAMS = 64
IMG_HW = (256, 448)          # cached camera size (H, W); train crop = 224x400
IMG_CROP = (224, 400)
BEV_RANGE = 64.0             # metres, LiDAR BEV covers [-R, R]^2  (check against EDA x-y density plot)
BEV_SIZE = 128               # pixels
N_PTS = 4096                 # fixed point count for point-based LiDAR models
SEED = 0
