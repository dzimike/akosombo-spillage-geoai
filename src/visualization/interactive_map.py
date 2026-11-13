from __future__ import annotations
"""
Phase 10 – Task 10.2: Interactive Web Map

Builds a Folium HTML map with togglable layers:
  • Flood extent (colour-coded by confidence class)
  • Flood susceptibility (XGBoost probability — downsampled image overlay)
  • Exposed buildings (red dots, with popup)
  • Exposed roads (coloured by class)
  • Affected community markers (with exposure popup)
  • NDVI recovery heatmap (downsampled image overlay)

The map is self-contained HTML — open directly in any browser.

Outputs
-------
outputs/maps/interactive_flood_map.html
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
import folium
from folium import plugins

from src.utils.paths import load_config, get_project_root, get_output_path, \
    get_processed_path, ensure_dir
from src.utils.logging_utils import Timer, log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJ_CRS   = "EPSG:32630"
GEO_CRS    = "EPSG:4326"
DOWNSAMPLE = 512   # max pixel dimension for raster image overlays


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def raster_to_png_overlay(tif_path: Path, cmap_name: str,
                           vmin: float, vmax: float,
                           alpha: float = 0.65) -> tuple[str, list]:
    """
    Downsample a float32 GeoTIFF → base64-encoded PNG + geographic bounds.
    Reads in TILE_ROWS-row strips to avoid LZW decompression buffer issues,
    then uses scipy/numpy to resize to DOWNSAMPLE px on the longest axis.
    """
    import base64, tempfile, os
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from PIL import Image
    from rasterio.warp import transform_bounds

    TILE_ROWS = 500

    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        nd   = src.nodata
        bounds_wgs = transform_bounds(src.crs, GEO_CRS, *src.bounds)

        # Read in strips and assemble
        strips = []
        for row_off in range(0, H, TILE_ROWS):
            row_count = min(TILE_ROWS, H - row_off)
            from rasterio.windows import Window
            win  = Window(0, row_off, W, row_count)
            tile = src.read(1, window=win).astype(np.float32)
            if nd is not None:
                tile = np.where(tile == nd, np.nan, tile)
            strips.append(tile)
        data = np.concatenate(strips, axis=0)   # (H, W)

    # Downsample via PIL resize (avoids scipy dependency)
    scale = DOWNSAMPLE / max(H, W)
    oh    = max(1, int(H * scale))
    ow    = max(1, int(W * scale))

    # Replace NaN with a sentinel for resize, then restore
    sentinel  = vmin - 1.0
    data_fill = np.where(np.isfinite(data), data, sentinel)
    img_pil   = Image.fromarray(data_fill.astype(np.float32), mode="F")
    img_small = np.array(img_pil.resize((ow, oh), Image.BILINEAR), dtype=np.float32)
    valid     = img_small > sentinel + 0.5   # mask for originally-valid pixels

    # Apply colormap
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap  = plt.colormaps[cmap_name]
    rgba  = cmap(norm(img_small))           # (oh, ow, 4)
    rgba[~valid] = [0, 0, 0, 0]
    rgba[..., 3] = np.where(valid, alpha, 0.0)
    img_uint8 = (rgba * 255).astype(np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    Image.fromarray(img_uint8, mode="RGBA").save(tmp_path, format="PNG")
    with open(tmp_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    os.unlink(tmp_path)

    s, w, n, e    = bounds_wgs[1], bounds_wgs[0], bounds_wgs[3], bounds_wgs[2]
    return f"data:image/png;base64,{b64}", [[s, w], [n, e]]


def gdf_to_geojson_str(gdf: gpd.GeoDataFrame) -> str:
    return gdf.to_crs(GEO_CRS).to_json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    proj_root = get_project_root()
    config    = load_config(config_path)
    out_dir   = get_output_path("maps", config)
    ensure_dir(out_dir)

    # Input paths
    flood_conf_p  = get_processed_path("flood_extent/fused_flood_confidence.tif")
    flood_ext_p   = get_processed_path("flood_extent/fused_flood_extent.gpkg")
    susc_p        = proj_root / "outputs/maps/flood_susceptibility_xgb.tif"
    buildings_p   = get_processed_path("exposure/exposed_buildings.gpkg")
    roads_p       = get_processed_path("exposure/exposed_roads.gpkg")
    comm_p        = get_processed_path("aoi/affected_communities.gpkg")
    ndvi_rec_p    = get_processed_path("recovery/ndvi_recovery.tif")
    rec_idx_p     = proj_root / "outputs/tables/community_recovery_index.csv"
    comm_summary_p = get_processed_path("exposure/community_exposure_summary.gpkg")

    with Timer() as t:
        # Centre map on AOI centroid
        comms = gpd.read_file(comm_p).to_crs(GEO_CRS)
        center = [comms.geometry.y.mean(), comms.geometry.x.mean()]
        log.info("Map centre: %.4f, %.4f", *center)

        m = folium.Map(location=center, zoom_start=10,
                       tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                       attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
                       control_scale=True)

        # ---- Layer 1: Flood extent polygons ----------------------------
        log.info("Adding flood extent layer…")
        if flood_ext_p.exists():
            flood_gdf = gpd.read_file(flood_ext_p)
            flood_layer = folium.FeatureGroup(name="Flood Extent", show=True)
            folium.GeoJson(
                gdf_to_geojson_str(flood_gdf),
                style_function=lambda f: {
                    "fillColor": "#1565C0",
                    "color":     "#0D47A1",
                    "weight":    0.5,
                    "fillOpacity": 0.45,
                },
                **({"tooltip": folium.GeoJsonTooltip(
                    fields=["flood_class_label"],
                    aliases=["Class:"],
                )} if "flood_class_label" in flood_gdf.columns else {}),
            ).add_to(flood_layer)
            flood_layer.add_to(m)

        # ---- Layer 2: Susceptibility raster overlay --------------------
        log.info("Adding susceptibility overlay…")
        if susc_p.exists():
            try:
                from PIL import Image  # noqa – verify PIL available
                img_uri, bounds = raster_to_png_overlay(
                    susc_p, "RdYlGn_r", vmin=0.0, vmax=1.0, alpha=0.6)
                susc_layer = folium.FeatureGroup(name="Flood Susceptibility (XGBoost)",
                                                 show=False)
                folium.raster_layers.ImageOverlay(
                    image=img_uri, bounds=bounds, name="Susceptibility",
                    opacity=1.0,
                ).add_to(susc_layer)
                susc_layer.add_to(m)
                log.info("  Susceptibility overlay added")
            except Exception as e:
                log.warning("  Could not add susceptibility overlay: %s", e)

        # ---- Layer 3: NDVI Recovery overlay ----------------------------
        log.info("Adding NDVI recovery overlay…")
        if ndvi_rec_p.exists():
            try:
                from PIL import Image  # noqa
                img_uri2, bounds2 = raster_to_png_overlay(
                    ndvi_rec_p, "RdYlGn", vmin=0.0, vmax=1.0, alpha=0.65)
                ndvi_layer = folium.FeatureGroup(name="NDVI Recovery", show=False)
                folium.raster_layers.ImageOverlay(
                    image=img_uri2, bounds=bounds2, name="NDVI Recovery",
                    opacity=1.0,
                ).add_to(ndvi_layer)
                ndvi_layer.add_to(m)
            except Exception as e:
                log.warning("  Could not add NDVI overlay: %s", e)

        # ---- Layer 4: Exposed buildings --------------------------------
        log.info("Adding exposed buildings…")
        if buildings_p.exists():
            bldg       = gpd.read_file(buildings_p)
            bldg_proj  = bldg.to_crs(PROJ_CRS)
            bldg["cx"] = bldg_proj.geometry.centroid.to_crs(GEO_CRS).x
            bldg["cy"] = bldg_proj.geometry.centroid.to_crs(GEO_CRS).y
            bldg_layer = folium.FeatureGroup(name="Exposed Buildings", show=False)
            for _, row in bldg.head(5000).iterrows():   # cap at 5 000 for perf
                folium.CircleMarker(
                    location=[row.cy, row.cx],
                    radius=2, color="#E53935", fill=True,
                    fill_opacity=0.6, weight=0,
                    tooltip=row.get("building_type", "Building"),
                ).add_to(bldg_layer)
            bldg_layer.add_to(m)
            log.info("  Buildings: %d markers", min(len(bldg), 5000))

        # ---- Layer 5: Exposed roads ------------------------------------
        log.info("Adding exposed roads…")
        if roads_p.exists():
            roads = gpd.read_file(roads_p).to_crs(GEO_CRS)
            road_colors = {
                "primary":      "#E53935",
                "secondary":    "#FF7043",
                "local":        "#FFA726",
                "tracks_paths": "#FFCA28",
                "other":        "#B0BEC5",
            }
            road_layer = folium.FeatureGroup(name="Exposed Roads", show=True)
            for rc, grp in roads.groupby("road_class"):
                color = road_colors.get(rc, "#B0BEC5")
                folium.GeoJson(
                    grp.to_json(),
                    name=f"Roads — {rc}",
                    style_function=lambda f, c=color: {
                        "color": c, "weight": 2, "opacity": 0.75},
                    tooltip=folium.GeoJsonTooltip(
                        fields=["road_class", "flooded_length_km"],
                        aliases=["Class:", "Flooded km:"],
                    ),
                ).add_to(road_layer)
            road_layer.add_to(m)

        # ---- Layer 6: Community markers --------------------------------
        log.info("Adding community markers…")
        rec_idx = (pd.read_csv(rec_idx_p).set_index("community_name")
                   if rec_idx_p.exists() else pd.DataFrame())
        exp_summary = (gpd.read_file(comm_summary_p)
                       if comm_summary_p.exists() else gpd.GeoDataFrame())

        comm_layer = folium.FeatureGroup(name="Affected Communities", show=True)
        for _, row in comms.iterrows():
            name  = row.community_name
            ri    = rec_idx.loc[name, "recovery_index"] \
                    if name in rec_idx.index else "N/A"
            ri_rank = rec_idx.loc[name, "recovery_rank"] \
                      if name in rec_idx.index else "N/A"

            exp_row = (exp_summary[exp_summary.community_name == name].iloc[0]
                       if len(exp_summary) and name in
                       exp_summary.community_name.values else None)

            popup_lines = [
                f"<b>{name}</b>",
                f"Recovery Index: <b>{ri:.3f}</b> (rank {ri_rank})" if isinstance(ri, float) else "",
            ]
            if exp_row is not None:
                popup_lines += [
                    f"Exposed population: {int(exp_row.exposed_pop_estimate):,}",
                    f"Exposed buildings: {int(exp_row.exposed_buildings):,}",
                    f"Flooded roads: {exp_row.flooded_road_km:.1f} km",
                    f"Flood area: {exp_row.flood_area_km2:.1f} km²",
                ]
            popup_html = "<br>".join(l for l in popup_lines if l)

            # Colour star by recovery index
            ri_val     = ri if isinstance(ri, float) else 0.5
            star_color = ("green" if ri_val >= 0.7
                          else "orange" if ri_val >= 0.4 else "red")
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=name,
                icon=folium.Icon(color=star_color, icon="star", prefix="fa"),
            ).add_to(comm_layer)
        comm_layer.add_to(m)

        # ---- Mini-map & layer control ----------------------------------
        plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

        # ---- Title box -------------------------------------------------
        title_html = """
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                    z-index:1000;background:white;padding:8px 16px;
                    border-radius:6px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);
                    font-family:Arial;font-size:14px;font-weight:bold;
                    text-align:center;">
            Akosombo Dam Spillage 2023 — Flood Inundation &amp; Recovery
            <br><span style="font-size:11px;font-weight:normal;color:#555;">
            Lower Volta Basin, Ghana &nbsp;|&nbsp; GeoAI Analysis
            </span>
        </div>"""
        m.get_root().html.add_child(folium.Element(title_html))

        out_path = out_dir / "interactive_flood_map.html"
        m.save(str(out_path))

    file_mb = out_path.stat().st_size / 1e6
    log.info("Interactive map saved: %s  (%.1f MB)  in %.1f s",
             out_path, file_mb, t.elapsed)
    log_run(
        script=__file__,
        input_paths=[],
        output_paths=[str(out_path)],
        parameters={"file_mb": round(file_mb, 1)},
        status="success", runtime_seconds=t.elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Folium web map")
    parser.add_argument("--config", default=None)
    main(parser.parse_args().config)
