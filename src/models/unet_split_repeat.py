from __future__ import annotations
"""
Repeat the U-Net ablation on other spatial block partitions.

The U-Net results use one assignment of the 50 blocks to training, validation and test (seed 42). This script retrains the four experiments on other
partitions (seeds given on the command line) with train_unet.py's code and hyperparameters, and scores every model on ALL valid pixels of that
partition's eight test blocks, the protocol of compare_models_on_test_blocks.py: positive = classes 1-3; class 4 and no-data excluded; scored where the
XGBoost core features are finite; threshold 0.5; probabilities from overlapping 256-px tiles at stride 128 with cosine blending (predict_unet.py),
restricted to the tiles that touch a test block. The seed-42 models already on disk are scored with the same function, as a check that it reproduces
the published comparison.

Models and per-seed tables go to outputs/models/split_repeat/ and outputs/tables/split_repeat/; the main models are not touched.

Usage:  python -m src.models.unet_split_repeat --seeds 1 2 3 [--experiments full sar_only optical_only terrain_only]
Output: outputs/tables/unet_split_repeat.csv (one row per seed and experiment; appended as each model finishes)
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window

import src.models.train_unet as tu
from src.models.predict_unet import STRIDE, TILE_SIZE, _cosine_window, get_device
from src.models.train_random_forest import IMPUTE_PREFIXES, add_spatial_blocks, spatial_train_test_split
from src.models.unet_model import UNet
from src.utils.paths import get_output_path, get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
CORE_IDX = [i for i, n in enumerate(tu.FEATURE_NAMES) if not any(n.startswith(p) for p in IMPUTE_PREFIXES)]


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12), tp / max(tp + fp + fn, 1)


@torch.no_grad()
def score_bundle(bundle_path, test_blocks, feat_path, label_path, device, thr=0.5):
    """All-pixel scores of one U-Net bundle on the given test blocks."""
    b = torch.load(bundle_path, map_location="cpu", weights_only=False)
    idx = b["band_indices"]; m = UNet(in_channels=len(idx), base_filters=b.get("base_filters", 16))
    m.load_state_dict(b["model_state"]); m.eval().to(device)
    mean = b["stats_mean"][idx].reshape(-1, 1, 1).astype(np.float32); std = b["stats_std"][idx].reshape(-1, 1, 1).astype(np.float32)
    P, S = TILE_SIZE, STRIDE; win_w = _cosine_window(P)
    with rasterio.open(feat_path) as fs, rasterio.open(label_path) as ls:
        H, W = ls.height, ls.width
        rs = list(range(0, H - P + 1, S)); cs = list(range(0, W - P + 1, S))
        if rs[-1] + P < H: rs.append(H - P)
        if cs[-1] + P < W: cs.append(W - P)
        acc = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "blk": []}
        for blk in sorted(test_blocks):
            r, c = map(int, blk.split("_"))
            br0, bc0 = r * tu.BLOCK_PX, c * tu.BLOCK_PX
            br1, bc1 = min(br0 + tu.BLOCK_PX, H), min(bc0 + tu.BLOCK_PX, W)
            tiles = [(r0, c0) for r0 in rs for c0 in cs if r0 < br1 and r0 + P > br0 and c0 < bc1 and c0 + P > bc0]
            pa = np.zeros((br1 - br0, bc1 - bc0), np.float64); wa = np.zeros_like(pa)
            for r0, c0 in tiles:
                feat = fs.read(window=Window(c0, r0, P, P)).astype(np.float32)[idx]
                feat = np.nan_to_num((feat - mean) / (std + 1e-8), nan=0.0, posinf=0.0, neginf=0.0)
                prob = torch.sigmoid(m(torch.from_numpy(feat[None]).to(device)))[0, 0].cpu().numpy()
                a0, a1 = max(r0, br0), min(r0 + P, br1); b0, b1 = max(c0, bc0), min(c0 + P, bc1)
                pa[a0 - br0:a1 - br0, b0 - bc0:b1 - bc0] += (prob * win_w)[a0 - r0:a1 - r0, b0 - c0:b1 - c0]
                wa[a0 - br0:a1 - br0, b0 - bc0:b1 - bc0] += win_w[a0 - r0:a1 - r0, b0 - c0:b1 - c0]
            prob = np.clip(np.where(wa > 0, pa / np.maximum(wa, 1e-12), 0.0), 0, 1)
            blkwin = Window(bc0, br0, bc1 - bc0, br1 - br0)
            lbl = ls.read(1, window=blkwin)
            core = np.ones(lbl.shape, bool)
            f = fs.read(window=blkwin).astype(np.float32)
            for i in CORE_IDX:
                core &= np.isfinite(f[i])
            valid = (lbl != 9) & (lbl != 4) & core; pos = (lbl >= 1) & (lbl <= 3); pred = prob >= thr
            tp = int((pos & pred & valid).sum()); fp = int((~pos & pred & valid).sum()); fn = int((pos & ~pred & valid).sum()); tn = int((~pos & ~pred & valid).sum())
            for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
                acc[k] += v
            if tp + fn > 0:
                acc["blk"].append(prf(tp, fp, fn)[2])
    p, r, f1, iou = prf(acc["tp"], acc["fp"], acc["fn"]); n = acc["tp"] + acc["fp"] + acc["fn"] + acc["tn"]
    return {"n_pixels": n, "flood_pct": 100 * (acc["tp"] + acc["fn"]) / n, "precision": p, "recall": r, "f1": f1, "iou": iou,
            "predicted_flood_km2": (acc["tp"] + acc["fp"]) / 1e4, "reference_flood_km2": (acc["tp"] + acc["fn"]) / 1e4,
            "block_f1_mean": float(np.mean(acc["blk"])), "blocks_with_flood": len(acc["blk"])}


def main(seeds, experiments) -> None:
    root = get_project_root()
    feat = root / "data/processed/features/flood_model_feature_stack.tif"
    label = root / "data/processed/flood_extent/fused_flood_confidence.tif"
    parquet = root / "data/processed/features/model_training_samples.parquet"
    out_csv = get_output_path("tables/unet_split_repeat.csv")
    device = get_device()
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    df = add_spatial_blocks(pd.read_parquet(parquet), tu.BLOCK_PX)

    def record(seed, name, scores, extra):
        nonlocal done
        done = pd.concat([done, pd.DataFrame([{"seed": seed, "experiment": name, **scores, **extra}])], ignore_index=True)
        done.to_csv(out_csv, index=False)
        log.info("seed %d %-13s all-pixel F1 %.4f (P %.3f R %.3f) | flood %.2f%%", seed, name, scores["f1"], scores["precision"], scores["recall"], scores["flood_pct"])

    # seed 42: score the models already on disk with the same function
    for name in tu.EXPERIMENTS:
        if ((done.seed == 42) & (done.experiment == name)).any() if len(done) else False:
            continue
        tu.RANDOM_SEED = 42
        _, _, te = tu.get_spatial_splits(parquet)
        record(42, name, score_bundle(root / f"outputs/models/unet_{name}.pt", te, feat, label, device), {"trained_here": False})

    for seed in seeds:
        tu.RANDOM_SEED = seed
        tr, va, te = tu.get_spatial_splits(parquet)
        _, _, xte = spatial_train_test_split(df, 0.70, 0.15, seed)
        assert set(xte.block_id.unique()) == set(te), f"seed {seed}: XGBoost and U-Net test blocks differ"
        mdir = root / f"outputs/models/split_repeat/seed{seed}"; tdir = root / f"outputs/tables/split_repeat/seed{seed}"
        mdir.mkdir(parents=True, exist_ok=True); tdir.mkdir(parents=True, exist_ok=True)
        for name in experiments:
            if len(done) and ((done.seed == seed) & (done.experiment == name)).any():
                continue
            np.random.seed(seed); torch.manual_seed(seed)
            t0 = time.time()
            if not (mdir / f"unet_{name}.pt").exists():
                res = tu.run_experiment(name, tu.EXPERIMENTS[name], feat, label, parquet, mdir, tdir, device)
            else:
                res = {}
            sc = score_bundle(mdir / f"unet_{name}.pt", te, feat, label, device)
            record(seed, name, sc, {"trained_here": True, "train_minutes": round((time.time() - t0) / 60, 1), "n_test_blocks": len(te),
                                    **{f"patch_{k}": v for k, v in res.items() if k.startswith("test_")}, "best_epoch": res.get("best_epoch")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--experiments", nargs="+", default=list(tu.EXPERIMENTS), choices=list(tu.EXPERIMENTS))
    a = ap.parse_args()
    main(a.seeds, a.experiments)
