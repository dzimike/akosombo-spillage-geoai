from __future__ import annotations
"""
Deep-learning paper — U-Net flood segmentation training

Trains four ablation variants on the same spatial-block 70/15/15 split
used by XGBoost (seed=42) to allow direct performance comparison:

  full         – all 28 input bands
  sar_only     – 5 SAR-derived bands
  optical_only – 7 optical spectral-index bands
  terrain_only – 19 pre-event terrain + settlement bands (forecast mode)

Outputs
-------
outputs/models/unet_full.pt
outputs/models/unet_sar_only.pt
outputs/models/unet_optical_only.pt
outputs/models/unet_terrain_only.pt
outputs/tables/unet_metrics.csv
outputs/tables/unet_<name>_spatial_cv.csv  (per-block metrics for each variant)
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window

from src.models.unet_model import UNet, DiceBCELoss
from src.utils.paths import (load_config, get_project_root, get_output_path,
                              get_processed_path, ensure_dir)
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Band index definitions (0-indexed positions in the 28-band feature stack)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "elevation", "slope", "aspect_sin", "aspect_cos", "twi",
    "flow_accumulation_log", "relative_elevation", "hand_proxy",
    "dist_to_waterway", "dist_to_dam", "dist_to_volta",
    "ndvi_event", "ndwi_event", "mndwi_event", "ndbi_event",
    "ndvi_pre", "mndwi_pre", "ndwi_pre",
    "landcover_class",
    "building_density_500m", "road_density_1km", "dist_to_road",
    "population_density",
    "sar_vv_pre", "sar_vv_event", "sar_vv_change",
    "sar_vh_pre", "sar_vh_event",
]

EXPERIMENTS = {
    "full":         list(range(28)),
    "sar_only":     [23, 24, 25, 26, 27],
    "optical_only": [11, 12, 13, 14, 15, 16, 17],
    "terrain_only": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26],
}

# Bands whose physical meaning is orientation-dependent — excluded from
# rotation/flip augmentation to avoid contradictory training signal.
# Indices are positions in the global 28-band FEATURE_NAMES list.
DIRECTIONAL_BANDS = frozenset([
    2,   # aspect_sin
    3,   # aspect_cos
    5,   # flow_accumulation_log  (accumulates downhill)
    8,   # dist_to_waterway       (toward a specific geographic target)
    9,   # dist_to_dam
    10,  # dist_to_volta
])

PATCH_SIZE   = 256
BLOCK_PX     = 1000
RANDOM_SEED  = 42
BATCH_SIZE   = 8
MAX_EPOCHS   = 60
PATIENCE     = 10
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DICE_WEIGHT  = 0.6
POS_WEIGHT   = 7.0
MIN_FLOOD_PX = 64
WORKERS      = 0


def get_device():
    if torch.backends.mps.is_available():
        log.info("Device: Apple MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        log.info("Device: CUDA GPU")
        return torch.device("cuda")
    log.info("Device: CPU")
    return torch.device("cpu")


def get_spatial_splits(parquet_path):
    df = pd.read_parquet(parquet_path)
    df["block_id"] = (
        (df["pixel_row"] // BLOCK_PX).astype(str) + "_" +
        (df["pixel_col"] // BLOCK_PX).astype(str)
    )
    rng    = np.random.default_rng(RANDOM_SEED)
    blocks = df["block_id"].unique()
    rng.shuffle(blocks)
    n       = len(blocks)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    return (
        set(blocks[:n_train]),
        set(blocks[n_train:n_train + n_val]),
        set(blocks[n_train + n_val:]),
    )


class FloodPatchDataset(Dataset):
    def __init__(self, feat_path, label_path, block_set,
                 band_indices, stats=None, augment=False):
        self.feat_path    = feat_path
        self.label_path   = label_path
        self.band_indices = band_indices
        self.stats        = stats
        self.augment      = augment
        self.patches      = []
        self._build(block_set)

    def _build(self, block_set):
        with rasterio.open(self.label_path) as src:
            H, W = src.height, src.width
        for row0 in range(0, H - PATCH_SIZE + 1, PATCH_SIZE):
            for col0 in range(0, W - PATCH_SIZE + 1, PATCH_SIZE):
                cr  = (row0 + PATCH_SIZE // 2) // BLOCK_PX
                cc  = (col0 + PATCH_SIZE // 2) // BLOCK_PX
                bid = f"{cr}_{cc}"
                if bid not in block_set:
                    continue
                win = Window(col0, row0, PATCH_SIZE, PATCH_SIZE)
                with rasterio.open(self.label_path) as src:
                    lbl = src.read(1, window=win)
                if int(np.sum((lbl >= 1) & (lbl <= 3))) >= MIN_FLOOD_PX:
                    self.patches.append((row0, col0))
        log.info("    %d flood-positive patches", len(self.patches))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        row0, col0 = self.patches[idx]
        win = Window(col0, row0, PATCH_SIZE, PATCH_SIZE)
        with rasterio.open(self.feat_path) as src:
            feat = src.read(window=win).astype(np.float32)[self.band_indices]
        with rasterio.open(self.label_path) as src:
            lbl = src.read(1, window=win).astype(np.float32)
        # target: flood classes 1-3; valid: permanent water (class 4) is ignored
        # in loss and metrics, as it is excluded from the XGBoost sample
        lbl = np.stack([((lbl >= 1) & (lbl <= 3)).astype(np.float32),
                        (lbl != 4).astype(np.float32)], axis=-1)
        if self.stats is not None:
            m = self.stats[0][self.band_indices, None, None]
            s = self.stats[1][self.band_indices, None, None]
            feat = (feat - m) / (s + 1e-8)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        if self.augment:
            # Augmentation mask: True for bands whose values are rotation-invariant.
            # Directional bands (aspect, flow accumulation, proximity distances)
            # are held fixed — rotating them would create physically inconsistent
            # training examples and destabilise gradient updates.
            aug = np.array([b not in DIRECTIONAL_BANDS
                            for b in self.band_indices], dtype=bool)
            k = np.random.randint(0, 4)
            feat[aug] = np.rot90(feat[aug], k, axes=(1, 2)).copy()
            lbl       = np.rot90(lbl, k, axes=(0, 1)).copy()
            if np.random.rand() > 0.5:
                feat[aug] = feat[aug][:, :, ::-1].copy()
                lbl       = lbl[:, ::-1].copy()
            if np.random.rand() > 0.5:
                feat[aug] = feat[aug][:, ::-1, :].copy()
                lbl       = lbl[::-1, :].copy()
        return torch.from_numpy(feat), torch.from_numpy(np.ascontiguousarray(lbl.transpose(2, 0, 1)))


def compute_stats(feat_path, label_path, train_blocks):
    log.info("  Computing normalisation statistics…")
    with rasterio.open(feat_path) as src:
        H, W = src.height, src.width
    candidates = []
    for row0 in range(0, H - PATCH_SIZE + 1, PATCH_SIZE):
        for col0 in range(0, W - PATCH_SIZE + 1, PATCH_SIZE):
            cr = (row0 + PATCH_SIZE // 2) // BLOCK_PX
            cc = (col0 + PATCH_SIZE // 2) // BLOCK_PX
            if f"{cr}_{cc}" in train_blocks:
                candidates.append((row0, col0))
    np.random.default_rng(RANDOM_SEED).shuffle(candidates)
    selected = candidates[:min(150, len(candidates))]
    s1 = np.zeros(28, np.float64); s2 = np.zeros(28, np.float64); cnt = 0
    for r0, c0 in selected:
        win = Window(c0, r0, PATCH_SIZE, PATCH_SIZE)
        with rasterio.open(feat_path) as src:
            p = src.read(window=win).astype(np.float64)
        valid = np.isfinite(p); n = valid[0].sum()
        if n == 0: continue
        for b in range(28):
            v = p[b][valid[b]]; s1[b] += v.sum(); s2[b] += (v**2).sum()
        cnt += n
    mean = (s1 / cnt).astype(np.float32)
    std  = np.sqrt(np.maximum(s2 / cnt - mean**2, 0)).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)
    return mean, std


def train_epoch(model, loader, criterion, opt, device):
    model.train(); total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, thr=0.5):
    model.eval(); total = 0.0; tp=fp=fn=tn=0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x); total += criterion(logits, y).item()
        pred = (torch.sigmoid(logits) >= thr).bool()
        v = y[:, 1:].bool() if y.shape[1] == 2 else torch.ones_like(pred)
        y = y[:, :1].bool()
        tp += (pred & y & v).sum().item();  fp += (pred & ~y & v).sum().item()
        fn += (~pred & y & v).sum().item(); tn += (~pred & ~y & v).sum().item()
    prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
    f1   = 2*prec*rec/(prec+rec+1e-8)
    iou  = tp/(tp+fp+fn+1e-8); oa = (tp+tn)/(tp+fp+fn+tn+1e-8)
    return {"val_loss": total/len(loader), "precision": prec,
            "recall": rec, "f1": f1, "iou": iou, "overall_accuracy": oa}


@torch.no_grad()
def eval_blocks(model, feat_path, label_path, test_blocks,
                band_indices, stats, device, thr=0.5):
    model.eval(); records = []
    with rasterio.open(feat_path) as src: H, W = src.height, src.width
    for r0 in range(0, H - PATCH_SIZE + 1, PATCH_SIZE):
        for c0 in range(0, W - PATCH_SIZE + 1, PATCH_SIZE):
            cr = (r0+PATCH_SIZE//2)//BLOCK_PX; cc = (c0+PATCH_SIZE//2)//BLOCK_PX
            bid = f"{cr}_{cc}"
            if bid not in test_blocks: continue
            win = Window(c0, r0, PATCH_SIZE, PATCH_SIZE)
            with rasterio.open(feat_path) as src:
                feat = src.read(window=win).astype(np.float32)[band_indices]
            with rasterio.open(label_path) as src:
                lbl = src.read(1, window=win).astype(np.float32)
            lbl_b = ((lbl>=1)&(lbl<=3)).astype(np.float32)
            vmask = (lbl != 4).astype(np.float32)
            if lbl_b.sum() < MIN_FLOOD_PX: continue
            m = stats[0][band_indices,None,None]; s = stats[1][band_indices,None,None]
            feat = np.nan_to_num((feat-m)/(s+1e-8), nan=0.0, posinf=0.0, neginf=0.0)
            pred = (torch.sigmoid(model(torch.from_numpy(feat[None]).to(device)))
                    [0,0] >= thr).cpu().numpy().astype(np.float32)
            tp=float((pred*lbl_b*vmask).sum()); fp=float((pred*(1-lbl_b)*vmask).sum())
            fn=float(((1-pred)*lbl_b*vmask).sum()); tn=float(((1-pred)*(1-lbl_b)*vmask).sum())
            prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8)
            records.append({"block_id": bid,
                            "n_flood_px": int(lbl_b.sum()),
                            "precision": round(prec,4),
                            "recall":    round(rec,4),
                            "f1":        round(2*prec*rec/(prec+rec+1e-8),4),
                            "iou":       round(tp/(tp+fp+fn+1e-8),4)})
    return pd.DataFrame(records)


def run_experiment(name, band_indices, feat_path, label_path,
                   parquet_path, models_dir, tables_dir, device):
    log.info("=" * 60)
    log.info("Experiment: %s  (%d bands)", name, len(band_indices))
    log.info("=" * 60)
    train_blocks, val_blocks, test_blocks = get_spatial_splits(parquet_path)
    log.info("Blocks  train=%d  val=%d  test=%d",
             len(train_blocks), len(val_blocks), len(test_blocks))

    stats = compute_stats(feat_path, label_path, train_blocks)

    log.info("  Building datasets…")
    train_ds = FloodPatchDataset(feat_path, label_path, train_blocks,
                                 band_indices, stats, augment=True)
    val_ds   = FloodPatchDataset(feat_path, label_path, val_blocks,
                                 band_indices, stats, augment=False)
    if len(train_ds) == 0:
        log.error("No training patches — skipping %s", name); return {}

    train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=WORKERS)
    val_ld   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    model     = UNet(in_channels=len(band_indices), base_filters=16).to(device)
    log.info("  Parameters: %s", f"{UNet.param_count(model):,}")
    criterion = DiceBCELoss(dice_weight=DICE_WEIGHT, pos_weight=POS_WEIGHT)
    criterion.bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT]).to(device), reduction="none")
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    best_loss = float("inf"); best_state = None; pat = 0; history = []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS+1):
        tl = train_epoch(model, train_ld, criterion, opt, device)
        vm = evaluate(model, val_ld, criterion, device)
        sched.step(vm["val_loss"])
        history.append({"epoch": ep, "train_loss": tl, **vm})
        log.info("  Ep %3d  train=%.4f  val=%.4f  F1=%.4f  IoU=%.4f",
                 ep, tl, vm["val_loss"], vm["f1"], vm["iou"])
        if vm["val_loss"] < best_loss - 1e-5:
            best_loss = vm["val_loss"]
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                log.info("  Early stop at epoch %d", ep); break
    elapsed = time.time() - t0

    if best_state:
        model.load_state_dict(best_state)
    bundle = {"model_state": best_state, "band_indices": band_indices,
              "band_names": [FEATURE_NAMES[i] for i in band_indices],
              "stats_mean": stats[0], "stats_std": stats[1],
              "patch_size": PATCH_SIZE, "base_filters": 16,
              "experiment": name, "history": history}
    torch.save(bundle, models_dir / f"unet_{name}.pt")
    log.info("  Saved: %s", models_dir / f"unet_{name}.pt")

    log.info("  Per-block test evaluation…")
    block_df = eval_blocks(model, feat_path, label_path, test_blocks,
                           band_indices, stats, device)
    if not block_df.empty:
        log.info("  Block F1  mean=%.4f  min=%.4f  max=%.4f",
                 block_df["f1"].mean(), block_df["f1"].min(), block_df["f1"].max())
        block_df["experiment"] = name
        block_df.to_csv(tables_dir / f"unet_{name}_spatial_cv.csv", index=False)

    test_ds = FloodPatchDataset(feat_path, label_path, test_blocks,
                                band_indices, stats, augment=False)
    test_ld = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    tm = evaluate(model, test_ld, criterion, device)
    log.info("  Test  F1=%.4f  IoU=%.4f  Prec=%.4f  Rec=%.4f",
             tm["f1"], tm["iou"], tm["precision"], tm["recall"])

    result = {"experiment": name, "n_bands": len(band_indices),
              "n_params": UNet.param_count(model),
              "best_epoch": len(history)-pat,
              "train_time_s": round(elapsed, 1),
              **{f"test_{k}": round(v,4) for k,v in tm.items()}}
    if not block_df.empty:
        result["spatial_cv_mean_f1"] = round(block_df["f1"].mean(), 4)
        result["spatial_cv_min_f1"]  = round(block_df["f1"].min(),  4)
    return result


def main(config_path=None, experiments=None):
    proj_root    = Path(__file__).resolve().parents[2]
    config       = load_config(config_path)
    models_dir   = get_output_path("models", config)
    tables_dir   = get_output_path("tables", config)
    ensure_dir(models_dir); ensure_dir(tables_dir)

    feat_path    = proj_root / "data/processed/features/flood_model_feature_stack.tif"
    label_path   = proj_root / "data/processed/flood_extent/fused_flood_confidence.tif"
    parquet_path = proj_root / "data/processed/features/model_training_samples.parquet"

    for p, n in [(feat_path, "feature stack"),
                 (label_path, "label raster"),
                 (parquet_path, "training parquet")]:
        if not p.exists():
            log.error("Missing: %s (%s)", p, n); return

    device = get_device()
    exps   = experiments or list(EXPERIMENTS.keys())

    results = []
    with Timer() as t:
        for name in exps:
            if name not in EXPERIMENTS:
                log.warning("Unknown experiment '%s'; skipping", name); continue
            r = run_experiment(name, EXPERIMENTS[name], feat_path, label_path,
                               parquet_path, models_dir, tables_dir, device)
            if r: results.append(r)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(tables_dir / "unet_metrics.csv", index=False)
        log.info("Metrics saved.\n%s", df.to_string(index=False))

    log.info("All experiments done in %.1f s", t.elapsed)
    log_run(script=__file__, input_paths=[str(feat_path)],
            output_paths=[str(models_dir/f"unet_{n}.pt") for n in exps],
            parameters={"experiments": exps, "device": str(device)},
            status="success", runtime_seconds=t.elapsed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--experiments", nargs="+", default=None,
                   choices=list(EXPERIMENTS.keys()))
    a = p.parse_args()
    main(a.config, a.experiments)
