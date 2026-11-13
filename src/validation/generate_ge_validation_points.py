"""
Google Earth validation point generator
========================================
Draws a stratified random sample from the five-class fused flood confidence
raster and exports two files:

  outputs/validation/ge_validation_points.kml
      Ready to open in Google Earth Pro. Each placemark shows the class,
      a colour-coded pin, and an empty "interpreter_label" field to fill in.

  outputs/validation/ge_validation_points.csv
      One row per point: point_id, lon, lat, map_class, map_class_name.
      After labelling in Google Earth, add an "interpreter_label" column
      (0 = not flooded, 1 = flooded) and save as ge_validation_points_labelled.csv,
      then run src/validation/score_ge_validation.py.

Sampling design
---------------
Class 0 – not flooded        : 60 points
Class 1 – SAR only           : 40 points
Class 2 – optical only       : 40 points
Class 3 – SAR + optical      : 20 points  (class is rare; sample exhaustively)
Class 4 – permanent water    : 40 points
Total                        : 200 points

Points are drawn at pixel centres in EPSG:32630, then reprojected to WGS 84
(lon/lat) for Google Earth. A minimum inter-point spacing of 500 m is enforced
within each class to avoid spatial clustering.

Usage
-----
    python -m src.validation.generate_ge_validation_points
    python -m src.validation.generate_ge_validation_points --seed 99 --spacing 1000
"""

from __future__ import annotations
import argparse
import csv
import logging
import textwrap
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer

from src.utils.paths import get_project_root, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIDENCE_RASTER = (
    "data/processed/flood_extent/fused_flood_confidence.tif"
)

CLASS_CONFIG = {
    0: {"name": "Not flooded",       "n": 60,  "icon": "http://maps.google.com/mapfiles/kml/paddle/grn-blank.png"},
    1: {"name": "SAR only",          "n": 40,  "icon": "http://maps.google.com/mapfiles/kml/paddle/blu-blank.png"},
    2: {"name": "Optical only",      "n": 40,  "icon": "http://maps.google.com/mapfiles/kml/paddle/ylw-blank.png"},
    3: {"name": "SAR + optical",     "n": 20,  "icon": "http://maps.google.com/mapfiles/kml/paddle/red-blank.png"},
    4: {"name": "Permanent water",   "n": 40,  "icon": "http://maps.google.com/mapfiles/kml/paddle/wht-blank.png"},
}

KML_TEMPLATE = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Akosombo Flood Validation Points (n={total})</name>
  <description>
    Stratified random sample from the five-class fused flood confidence map.
    For each placemark: visit in Google Earth, use historical imagery (Sep-Nov 2023),
    and record whether the location was flooded (1) or not flooded (0)
    in the Description field labelled interpreter_label.
  </description>
{styles}
{placemarks}
</Document>
</kml>""")

STYLE_TEMPLATE = textwrap.dedent("""\
  <Style id="class{cls_id}">
    <IconStyle>
      <Icon><href>{icon}</href></Icon>
      <scale>0.9</scale>
    </IconStyle>
    <LabelStyle><scale>0.7</scale></LabelStyle>
  </Style>""")

PLACEMARK_TEMPLATE = textwrap.dedent("""\
  <Placemark>
    <name>{point_id}</name>
    <styleUrl>#class{cls_id}</styleUrl>
    <description><![CDATA[
      <b>Point ID:</b> {point_id}<br/>
      <b>Map class:</b> {cls_id} – {cls_name}<br/>
      <b>Lon:</b> {lon:.6f}<br/>
      <b>Lat:</b> {lat:.6f}<br/>
      <hr/>
      <b>interpreter_label:</b> ___  (enter 0 = not flooded, 1 = flooded)<br/>
      <b>Notes:</b> ___<br/>
      <br/>
      Use the historical imagery slider to view Sep–Nov 2023 imagery.
    ]]></description>
    <Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>
  </Placemark>""")


# ---------------------------------------------------------------------------
# Spatial utilities
# ---------------------------------------------------------------------------

def pixel_centres(raster_path: Path, cls_id: int) -> np.ndarray:
    """Return (N, 2) array of (col_x, row_y) pixel-centre coords for class."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        rows, cols = np.where(data == cls_id)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    return np.column_stack([xs, ys])


def spatially_thin(coords: np.ndarray, n: int, min_spacing: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Draw up to n points from coords with a minimum inter-point distance of
    min_spacing metres using a greedy random thinning approach.
    """
    idx = rng.permutation(len(coords))
    selected = []
    for i in idx:
        cand = coords[i]
        if not selected:
            selected.append(cand)
            continue
        sel_arr = np.array(selected)
        dists = np.sqrt(((sel_arr - cand) ** 2).sum(axis=1))
        if dists.min() >= min_spacing:
            selected.append(cand)
        if len(selected) >= n:
            break
    return np.array(selected)


# ---------------------------------------------------------------------------
# KML / CSV writers
# ---------------------------------------------------------------------------

def build_kml(rows: list[dict]) -> str:
    styles = "\n".join(
        STYLE_TEMPLATE.format(cls_id=c, icon=cfg["icon"])
        for c, cfg in CLASS_CONFIG.items()
    )
    placemarks = "\n".join(
        PLACEMARK_TEMPLATE.format(**r)
        for r in rows
    )
    return KML_TEMPLATE.format(
        total=len(rows),
        styles=styles,
        placemarks=placemarks,
    )


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["point_id", "lon", "lat", "map_class", "map_class_name"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "point_id":      r["point_id"],
                "lon":           round(r["lon"], 6),
                "lat":           round(r["lat"], 6),
                "map_class":     r["cls_id"],
                "map_class_name": r["cls_name"],
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 42, min_spacing: float = 500.0) -> None:
    proj_root = get_project_root()
    raster_path = proj_root / CONFIDENCE_RASTER
    if not raster_path.exists():
        log.error("Flood confidence raster not found: %s", raster_path)
        return

    out_dir = proj_root / "outputs/validation"
    ensure_dir(out_dir)

    with rasterio.open(raster_path) as src:
        utm_crs = src.crs

    transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    point_counter = 1

    for cls_id, cfg in CLASS_CONFIG.items():
        log.info("Class %d (%s): extracting pixel centres…", cls_id, cfg["name"])
        coords = pixel_centres(raster_path, cls_id)

        if len(coords) == 0:
            log.warning("  No pixels for class %d — skipping", cls_id)
            continue

        n_req = cfg["n"]
        if len(coords) <= n_req:
            log.info("  Only %d pixels available (requested %d); taking all", len(coords), n_req)
            selected = coords
        else:
            selected = spatially_thin(coords, n_req, min_spacing, rng)

        log.info("  Selected %d / %d requested points (spacing ≥ %.0f m)",
                 len(selected), n_req, min_spacing)

        for x_utm, y_utm in selected:
            lon, lat = transformer.transform(x_utm, y_utm)
            rows.append({
                "point_id": f"V{point_counter:04d}",
                "cls_id":   cls_id,
                "cls_name": cfg["name"],
                "lon":      lon,
                "lat":      lat,
            })
            point_counter += 1

    # Shuffle final list so class clusters don't bias the interpreter
    rng.shuffle(rows)

    kml_path = out_dir / "ge_validation_points.kml"
    kml_path.write_text(build_kml(rows), encoding="utf-8")
    log.info("KML written: %s  (%d points)", kml_path, len(rows))

    csv_path = out_dir / "ge_validation_points.csv"
    write_csv(rows, csv_path)
    log.info("CSV written: %s", csv_path)

    log.info("")
    log.info("Next steps:")
    log.info("  1. Open %s in Google Earth Pro", kml_path.name)
    log.info("  2. Enable historical imagery (View > Historical Imagery)")
    log.info("  3. Navigate to Sep–Nov 2023 and label each placemark:")
    log.info("     0 = not flooded,  1 = flooded")
    log.info("  4. Export the labelled data as ge_validation_points_labelled.csv")
    log.info("     (add an 'interpreter_label' column to the CSV copy)")
    log.info("  5. Run: python -m src.validation.score_ge_validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Google Earth validation points from flood confidence raster"
    )
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--spacing", type=float, default=500.0,
                        help="Minimum inter-point distance in metres (default 500)")
    args = parser.parse_args()
    main(args.seed, args.spacing)
