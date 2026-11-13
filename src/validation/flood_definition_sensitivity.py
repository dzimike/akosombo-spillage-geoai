from __future__ import annotations
"""
How the flood-extent definition changes the mapped area, and how well the two
independent sensors (Sentinel-1 backscatter, Sentinel-2 NDWI+MNDWI) agree.

Rows
----
v1_original            original fused map (sparse-SAR, sea not masked) from archive/v3_sea_in_labels
v1_original_land_only  same, restricted to land (GLO-30 sea mask removed)
primary                current fused map: event water outside JRC / open sea / pre-event optical water
primary_new_only       the part of the primary extent that was NOT SAR-dark in the pre-event window
Plus SAR-vs-optical agreement (precision/recall/F1/IoU/kappa of the SAR mask against the
optical mask, on land outside permanent water).

Output: outputs/tables/flood_definition_sensitivity.csv
"""
import logging

import numpy as np
import pandas as pd
import rasterio

from src.utils.paths import get_project_root, get_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def rd(p):
    with rasterio.open(p) as s:
        return s.read(1)


def km2(m):
    return float(m.sum()) / 1e4


def main() -> None:
    root = get_project_root()
    F = root / "data/processed/flood_extent"
    old = rd(root / "archive/v3_sea_in_labels/flood_extent/fused_flood_confidence.tif")
    new = rd(F / "fused_flood_confidence.tif")
    sea = rd(F / "sea_mask.tif") == 1
    sar_pre = rd(F / "sar_pre_water_mask.tif") == 1
    sar = rd(F / "sar_flood_mask_binary.tif") == 1
    opt = rd(F / "optical_water_mask.tif") == 1
    flood_old, flood_new = np.isin(old, [1, 2, 3]), np.isin(new, [1, 2, 3])
    rows = [
        {"definition": "v1_original (sparse SAR, sea unmasked)", "flood_km2": km2(flood_old),
         "of_which_sea_km2": km2(flood_old & sea)},
        {"definition": "v1_original, land only", "flood_km2": km2(flood_old & ~sea), "of_which_sea_km2": 0.0},
        {"definition": "primary (event water outside permanent water)", "flood_km2": km2(flood_new),
         "of_which_sea_km2": km2(flood_new & sea), "sar_only_km2": km2(new == 1),
         "optical_only_km2": km2(new == 2), "agreement_km2": km2(new == 3),
         "permanent_water_km2": km2(new == 4)},
        {"definition": "primary, new-only (not SAR-dark in pre-event window)",
         "flood_km2": km2(flood_new & ~sar_pre), "of_which_sea_km2": 0.0,
         "sar_only_km2": km2((new == 1) & ~sar_pre), "optical_only_km2": km2((new == 2) & ~sar_pre),
         "agreement_km2": km2((new == 3) & ~sar_pre)},
    ]
    dom = (new != 4) & ~sea
    s, o = sar & dom, opt & dom
    tp, fp, fn = (s & o).sum(), (s & ~o).sum(), (~s & o).sum()
    tn = (dom & ~s & ~o).sum()
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    agree = {"sar_km2": km2(s), "optical_km2": km2(o), "overlap_km2": float(tp) / 1e4,
             "precision_sar_vs_optical": tp / max(tp + fp, 1), "recall_sar_vs_optical": tp / max(tp + fn, 1),
             "f1": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1),
             "kappa": (po - pe) / (1 - pe)}
    pd.DataFrame(rows).to_csv(get_output_path("tables/flood_definition_sensitivity.csv"), index=False)
    pd.DataFrame([agree]).to_csv(get_output_path("tables/sar_optical_agreement.csv"), index=False)
    log.info("\n%s", pd.DataFrame(rows).round(2).to_string(index=False))
    log.info("SAR vs optical: %s", {k: round(v, 3) for k, v in agree.items()})


if __name__ == "__main__":
    main()
