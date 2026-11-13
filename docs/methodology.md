# Methodology

## Overview

This project implements a reproducible GeoAI pipeline to model flood inundation,
exposure, and recovery from the 2023 Akosombo Dam spillage in Ghana's Lower Volta Basin.

## Phase 1: Area of Interest Construction

Community seed points were used to define a buffered corridor AOI along the Lower Volta River
downstream of Akosombo and Kpong dams. A 20 km buffer was used as the default, with 10 km and
30 km buffers tested for sensitivity.

TODO: Replace placeholder AOI with river corridor geometry from HydroSHEDS or MERIT Hydro.

## Phase 2: Data Ingestion

- Sentinel-1 GRD SAR imagery accessed via Google Earth Engine.
- Sentinel-2 SR imagery accessed via Google Earth Engine.
- Copernicus GLO-30 DEM (void-free) from GEE (SRTM was replaced: its voids over water and wetland removed terrain from most of the flood area).
- OpenStreetMap features (roads, buildings, facilities) via OSMnx.
- WorldPop population grid.
- GADM administrative boundaries.
- JRC Global Surface Water occurrence (>= 50 %), downloaded by src/preprocessing/download_jrc_water.py, plus an open-sea mask from the DEM (src/flood_mapping/build_sea_mask.py), for permanent water masking.

## Phase 3: Remote Sensing Preprocessing

Sentinel-1 preprocessing (src/preprocessing/build_s1_composites.py): COPERNICUS/S1_GRD IW VV+VH, which is already
calibrated and in dB (do NOT apply 10*log10 again: the original export did, which left only positive-dB pixels, ~1 % of the
domain, as 'composites'), 7×7 focal mean on the dB values, per-window median over both relative orbits, windows from
config event_windows. Full coverage of the domain.

Sentinel-2 preprocessing: SCL-based cloud masking, NDVI/NDWI/MNDWI/NDBI computation, monthly compositing.

DEM derivatives: slope, aspect, flow accumulation, TWI, relative elevation.

## Phase 4: Flood Extent Mapping

SAR water detection: VH and VV event-window backscatter thresholds chosen by Youden's J between long-term JRC water and
land (the pre/event difference is unimodal and not used), permanent water removal (JRC, open sea, pre-event optical water),
steep-slope mask (>5°), minimum mapping unit filter (10 pixels). The stricter 'new water only' extent (not SAR-dark in the
pre-event window) is written separately as a sensitivity (sar_new_water_mask.tif).

Optical detection: NDWI ≥ 0.0 and MNDWI ≥ 0.0 thresholds where cloud-free.

Fusion: 5-class confidence map (SAR-only, optical-only, agreement, permanent water, uncertain).

## Phase 5: Exposure Analysis

Spatial intersection of flood extent with buildings, roads, population grid, cropland, and facilities.
All calculations in EPSG:32630 (metric). Area in km²/ha; road length in km; population as estimated count.

## Phase 6: Feature Engineering

28 predictor bands covering terrain, hydrology, optical indices, land cover, settlement, and SAR.
Aligned to common 10 m grid, clipped to AOI.

## Phase 7: GeoAI Modeling

Random Forest and XGBoost flood susceptibility models trained on spatially blocked samples.
Spatial block cross-validation to prevent spatial autocorrelation leakage.
U-Net segmentation model optional (requires ≥500 labelled patches).

## Phase 8: Recovery Analysis

NDVI recovery ratio, water persistence, road accessibility recovery ratio, and composite
weighted recovery index. All recovery products are proxies unless field-validated.

## Phase 9: Validation and Uncertainty

Stratified random validation points (minimum 300) from the flood confidence map.
Accuracy assessment with overall accuracy, precision, recall, F1, IoU.
Threshold sensitivity analysis and bootstrap confidence intervals for exposure estimates.

## Limitations

See Section 28 of CLAUDE.md for full list of explicitly stated limitations.

## CRS Policy

- Storage: EPSG:4326 (WGS 84)
- Analysis: EPSG:32630 (WGS 84 / UTM Zone 30N)
- All distance, area, and length calculations use projected CRS.
