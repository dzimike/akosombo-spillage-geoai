from __future__ import annotations
"""
Can the satellite layers see the buildings that the Charter 844 WorldView-2 map (18 Oct 2023) marks as flooded?

For the 4,215 OSM buildings in the Mepe map frame (179 flagged, charter_mepe_buildings.py), the ROC AUC of each
layer (flagged vs not flagged, building centroid) and the recall of the fused flood map are written to
outputs/tables/charter_mepe_detectability.csv. The orange flags are analyst interpretation, not field-validated.
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from src.utils.paths import get_project_root, get_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    root = get_project_root()
    sat, fl = root / "data/processed/satellite", root / "data/processed/flood_extent"
    rd = lambda p: rasterio.open(p).read(1)
    with rasterio.open(sat / "s1_pre_event_vv.tif") as s:
        tr, crs = s.transform, s.crs
    mb = gpd.read_file(root / "data/processed/validation/charter_mepe_flooded_buildings.gpkg").to_crs(crs)
    c = mb.geometry.centroid
    rr, cc = rasterio.transform.rowcol(tr, c.x.values, c.y.values); rr, cc = np.array(rr), np.array(cc)
    f = mb.charter_affected.values
    layers = {
        "VV 23 Oct (low = water)": (rd(sat / "s1_scene_20231023_vv.tif"), -1),
        "VH 23 Oct (low = water)": (rd(sat / "s1_scene_20231023_vh.tif"), -1),
        "dVV 23 Oct - pre-event median (negative = drop)": (rd(sat / "s1_scene_20231023_vv.tif") - rd(sat / "s1_pre_event_vv.tif"), -1),
        "dVH 23 Oct - pre-event median": (rd(sat / "s1_scene_20231023_vh.tif") - rd(sat / "s1_pre_event_vh.tif"), -1),
        "event-composite VV": (rd(sat / "s1_event_vv.tif"), -1),
        "event-composite VH": (rd(sat / "s1_event_vh.tif"), -1),
        "Sentinel-2 event NDWI": (rd(sat / "s2_ndwi_event.tif"), 1),
        "Sentinel-2 event MNDWI": (rd(sat / "s2_mndwi_event.tif"), 1),
    }
    rows = []
    for n, (a, sign) in layers.items():
        x = a[rr, cc]; ok = np.isfinite(x)
        rows.append({"layer": n, "metric": "ROC AUC (flagged vs other buildings)", "value": roc_auc_score(f[ok], sign * x[ok])})
    # model probabilities (XGBoost susceptibility and U-Nets), when present
    maps = root / "outputs/maps"
    for n, pth in (("XGBoost flood probability", "flood_susceptibility_xgb.tif"),
                   ("U-Net full probability", "unet_full_flood_prob.tif"),
                   ("U-Net SAR-only probability", "unet_sar_only_flood_prob.tif"),
                   ("U-Net optical-only probability", "unet_optical_only_flood_prob.tif"),
                   ("U-Net terrain-only probability", "unet_terrain_only_flood_prob.tif")):
        if (maps / pth).exists():
            x = rd(maps / pth)[rr, cc]; ok = np.isfinite(x)
            rows.append({"layer": n, "metric": "ROC AUC (flagged vs other buildings)", "value": roc_auc_score(f[ok], x[ok])})
    fu = rd(fl / "fused_flood_confidence.tif")
    for n, m in (("fused flood (classes 1-3)", np.isin(fu, [1, 2, 3])), ("pre-existing water (class 4)", fu == 4)):
        near = ndimage.binary_dilation(m, iterations=3)
        rows.append({"layer": n, "metric": "share of flagged buildings, centroid in class", "value": m[rr, cc][f].mean()})
        rows.append({"layer": n, "metric": "share of other buildings, centroid in class", "value": m[rr, cc][~f].mean()})
        rows.append({"layer": n, "metric": "share of flagged buildings, class within 30 m", "value": near[rr, cc][f].mean()})
        rows.append({"layer": n, "metric": "share of other buildings, class within 30 m", "value": near[rr, cc][~f].mean()})
    rows.append({"layer": "buildings", "metric": "flagged / total in map frame", "value": f"{int(f.sum())} / {len(f)}"})
    df = pd.DataFrame(rows)
    df.to_csv(get_output_path("tables/charter_mepe_detectability.csv"), index=False)
    log.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
