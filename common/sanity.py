"""python -m common.sanity   -> must pass before anybody trains."""
import json
from . import config as C
from .dataset import BeamDataset
from .metrics import baselines

sp = json.load(open(C.SPLIT_FILE))
assert not (set(sp["train"]) & set(sp["val"])) and not (set(sp["train"]) & set(sp["test"])) \
    and not (set(sp["val"]) & set(sp["test"])), "split overlap!"
for k in ("train", "val", "test"):
    print(k, len(sp[k]), "samples | seqs", sp[f"{k}_seqs"])
for mod in ("camera", "radar", "radar_ra", "lidar", "lidar_pts", "gps"):
    try:
        b = BeamDataset("val", [mod])[0]
        print(f"{mod:10s}", tuple(b[mod].shape), b[mod].dtype, "| label", int(b["label"]), "| idx", b["idx"])
    except FileNotFoundError:
        print(f"{mod:10s} cache missing -> run build_cache --stage {mod}")
b = BeamDataset("val", ["gps"], history=3, horizon=2)[0]
print("history/horizon OK:", tuple(b["gps"].shape))
print("baselines (test):", {k: {a: round(v, 3) for a, v in d.items()} for k, d in baselines().items()})
