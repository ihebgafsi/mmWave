"""Run ONCE (M1), then share cache as a Kaggle dataset. Everyone else just attaches it.
  python -m common.build_cache --stage meta        # master.csv + power.npy + splits (fast)
  python -m common.build_cache --stage camera|radar|radar_ra|lidar|lidar_pts
  python -m common.build_cache --stage all
"""
import argparse, json
import numpy as np, pandas as pd
from tqdm import tqdm
from . import config as C
from .splits import make_splits, save_splits


def read_ply(path):
    try:
        import open3d as o3d
        return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    except ImportError:
        from plyfile import PlyData
        v = PlyData.read(str(path))["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)


def load_master():
    return pd.read_csv(C.WORK / "master.csv")


# ---------------------------------------------------------------- meta
def stage_meta():
    C.WORK.mkdir(parents=True, exist_ok=True); C.CACHE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.DATA_ROOT / C.CSV_NAME, index_col=0).sort_index()
    lat0, lon0 = np.loadtxt(C.DATA_ROOT / df["unit1_loc"].iloc[0])
    R = 6371000
    g = np.array([np.loadtxt(C.DATA_ROOT / p) for p in df["unit2_loc"]])
    x = np.deg2rad(g[:, 1] - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(g[:, 0] - lat0) * R
    bearing = np.arctan2(x, y)
    t = pd.to_datetime(df["time_stamp"].str.replace("-", ".", n=1), format="%H:%M:%S.%f")

    power = np.stack([np.loadtxt(C.DATA_ROOT / p) for p in df["unit1_pwr_60ghz"]]).astype(np.float32)
    np.save(C.CACHE / "power.npy", power)                      # (N,64) may contain NaN
    arg = np.where(np.isnan(power), -np.inf, power).argmax(1) + 1

    m = pd.DataFrame({
        "idx": df.index.values, "seq": df["seq_index"].values,
        "beam": df["unit1_beam"].values, "beam0": df["unit1_beam"].values - 1,   # label in [0,63]
        "max_pwr": df["unit1_max_pwr"].values, "t_sec": (t - t.iloc[0]).dt.total_seconds().values,
        "x_m": x, "y_m": y, "dist_m": np.hypot(x, y),
        "bearing_deg": np.degrees(bearing), "sin_b": np.sin(bearing), "cos_b": np.cos(bearing),
        "speed_kmph": df["unit2_spd_over_grnd_kmph"].values,
        "interpolated_gps": df["unit2_interpolated_position"].values,
        "label_ne_argmax": (arg != df["unit1_beam"].values),     # ~1% rows: CSV label != argmax(power)
        "rgb": df["unit1_rgb"].values, "lidar": df["unit1_lidar"].values, "radar": df["unit1_radar"].values,
    })
    m["pos"] = m.groupby("seq").cumcount()
    # sequences must be contiguous in idx (EDA: 0 holes) -> row +/- k stays inside the sequence
    assert all((g_.idx.diff().dropna() == 1).all() for _, g_ in m.groupby("seq")), "hole inside a sequence"
    m.to_csv(C.WORK / "master.csv", index=False)

    C.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not C.SPLIT_FILE.exists():                               # never overwrite the committed split
        save_splits(make_splits(m, seed=C.SEED), C.SPLIT_FILE)
        print("splits written ->", C.SPLIT_FILE)
    sp = json.load(open(C.SPLIT_FILE))
    tr = m[m.idx.isin(sp["train"])]
    stats = {"gps_mean": tr[["x_m", "y_m", "dist_m"]].mean().tolist(),
             "gps_std": tr[["x_m", "y_m", "dist_m"]].std().tolist()}
    json.dump(stats, open(C.CACHE / "stats.json", "w"))
    print({k: len(sp[k]) for k in ("train", "val", "test")}, "seqs:",
          {k: sp[f"{k}_seqs"] for k in ("train", "val", "test")})


# ---------------------------------------------------------------- camera
def stage_camera():
    from PIL import Image
    m = load_master(); H, W = C.IMG_HW
    mm = np.lib.format.open_memmap(C.CACHE / "camera.npy", "w+", np.uint8, (len(m), H, W, 3))
    for r, p in enumerate(tqdm(m.rgb)):
        mm[r] = np.asarray(Image.open(C.DATA_ROOT / p).convert("RGB").resize((W, H), Image.BILINEAR))
    mm.flush()


# ---------------------------------------------------------------- radar
def radar_rd(r):    # (4,256,250) complex -> (4,256,250) dB, Doppler centred  [EDA cell 84]
    f = np.fft.fft(r, axis=1)
    d = np.fft.fftshift(np.fft.fft(f, axis=2), axes=2)
    return 20 * np.log10(np.abs(d) + 1e-6)


def radar_ra(r):    # -> (1,256,64) dB range-angle  [EDA cell 142]
    f = np.fft.fft(r, axis=1)
    a = np.fft.fftshift(np.abs(np.fft.fft(f, n=64, axis=0)).mean(axis=2), axes=0).T
    return (20 * np.log10(a + 1e-6))[None]


def stage_radar(kind):
    m = load_master(); fn, shp = (radar_rd, (4, 256, 250)) if kind == "radar" else (radar_ra, (1, 256, 64))
    mm = np.lib.format.open_memmap(C.CACHE / f"{kind}.npy", "w+", np.float16, (len(m),) + shp)
    for r, p in enumerate(tqdm(m.radar)):
        mm[r] = fn(np.load(C.DATA_ROOT / p)).astype(np.float16)
    mm.flush()
    sp = json.load(open(C.SPLIT_FILE)); rows = m.index[m.idx.isin(sp["train"])].to_numpy()
    sub = np.asarray(mm[np.sort(np.random.default_rng(0).choice(rows, 300, replace=False))], dtype=np.float32)
    st = json.load(open(C.CACHE / "stats.json"))
    st[f"{kind}_mean"] = sub.mean(axis=(0, 2, 3)).tolist(); st[f"{kind}_std"] = sub.std(axis=(0, 2, 3)).tolist()
    json.dump(st, open(C.CACHE / "stats.json", "w"))


# ---------------------------------------------------------------- lidar
def lidar_bev(pts, R=C.BEV_RANGE, S=C.BEV_SIZE):
    pts = pts[np.isfinite(pts).all(1)]
    p = pts[(np.abs(pts[:, 0]) < R) & (np.abs(pts[:, 1]) < R)]
    ix = ((p[:, 0] + R) / (2 * R) * S).astype(int).clip(0, S - 1)
    iy = ((p[:, 1] + R) / (2 * R) * S).astype(int).clip(0, S - 1)
    flat = ix * S + iy
    z = np.clip(p[:, 2], -4, 8) / 4
    cnt = np.bincount(flat, minlength=S * S).astype(np.float32)
    zmax = np.full(S * S, -np.inf, np.float32); np.maximum.at(zmax, flat, z); zmax[~np.isfinite(zmax)] = 0
    zmean = np.divide(np.bincount(flat, weights=z, minlength=S * S), cnt, out=np.zeros(S * S), where=cnt > 0)
    return np.stack([np.log1p(cnt), zmax, zmean]).reshape(3, S, S)       # ch: density, max z, mean z


def stage_lidar():
    m = load_master(); S = C.BEV_SIZE
    mm = np.lib.format.open_memmap(C.CACHE / f"lidar_bev_R{int(C.BEV_RANGE)}_S{S}.npy", "w+",
                                   np.float16, (len(m), 3, S, S))
    for r, p in enumerate(tqdm(m.lidar)):
        mm[r] = lidar_bev(read_ply(C.DATA_ROOT / p)).astype(np.float16)
    mm.flush()


def stage_lidar_pts():
    m = load_master(); rng = np.random.default_rng(0)
    out = np.zeros((len(m), C.N_PTS, 3), np.float16)
    for r, p in enumerate(tqdm(m.lidar)):
        pts = read_ply(C.DATA_ROOT / p); pts = pts[np.isfinite(pts).all(1)]
        out[r] = pts[rng.choice(len(pts), C.N_PTS, replace=len(pts) < C.N_PTS)]
    np.save(C.CACHE / f"lidar_pts_{C.N_PTS}.npy", out)       # metres; dataset clips/normalises


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    stages = {"meta": stage_meta, "camera": stage_camera, "radar": lambda: stage_radar("radar"),
              "radar_ra": lambda: stage_radar("radar_ra"), "lidar": stage_lidar, "lidar_pts": stage_lidar_pts}
    for name in (stages if a.stage == "all" else [a.stage]):
        print(">>", name); stages[name]()
