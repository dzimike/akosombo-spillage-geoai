# Akosombo spillage GeoAI

Code, derived tables and figure scripts for studies of the September-October 2023 Akosombo/Kpong Dam spillage on Ghana's Lower Volta River:
a change-based Sentinel-1/Sentinel-2 flood map, its comparison with independent analyst products, the exposure of buildings, roads, people,
cropland and facilities, and how all of that responds to the flood definition.

The study is described in three manuscripts (not in this repository):

1. Main paper: change-based flood mapping and exposure, comparison with UNOSAT, International Charter TerraSAR-X/TanDEM-X and WorldView-2 products, and sensitivity to the flood definition.
2. Uncertainty paper: propagation of the classifier threshold and the flood definition into exposure.
3. Deep-learning paper: U-Net and XGBoost classifiers trained on the labels, with spatial block cross-validation.

Everything here is a floodplain-scale analysis from 10 m satellite data. The independent products are analyst-derived and not field data. Nothing was validated
in the field, and the exposure counts are lower bounds.

## What is and is not in this repository

| In the repository | Not in the repository |
|---|---|
| Source code (`src/`), tests, configuration | Satellite imagery and other rasters (`data/`, about 16 GB; rebuilt by the scripts) |
| Result tables (`outputs/tables/`) and figures (`outputs/figures/`) | Trained model weights |
| The fused flood map, [`release_data/fused_flood_confidence.tif`](release_data/README.md), with class codes | UNOSAT and International Charter products and our digitisations of them (see below) |
| Method notes and logs (`docs/`) | Manuscript drafts; building and road layers (OpenStreetMap, rebuilt by the scripts) |

## Data sources and licences

| Source | Used for | Access and terms |
|---|---|---|
| Sentinel-1 GRD, Sentinel-2 L2A | change detection, water indices | Copernicus, free and open; accessed through Google Earth Engine |
| Copernicus GLO-30, JRC Global Surface Water, ESA WorldCover 2021 | terrain, pre-existing water, land use | open; accessed through Google Earth Engine or downloaded by `src/preprocessing/download_*.py` |
| OpenStreetMap | buildings, roads, facilities | ODbL; `src/data_ingestion/download_osm.py` |
| WorldPop 2020, GADM 4.1 | population, districts | open; see worldpop.org and gadm.org |
| UNOSAT FL20231018GHA flood extent (23 Oct 2023) | independent comparison | third-party product; obtain it from UNOSAT or the Humanitarian Data Exchange and check its licence. Not redistributed here. |
| International Charter activation 844 maps (TerraSAR-X, TanDEM-X, WorldView-2) | independent comparison | third-party products held by the Charter and its member agencies. Our digitisations of them are not redistributed, and Figure 8 of the main paper, which contains a WorldView-2 crop, is not included. Ask the Charter Executive Secretariat about reuse. |

Code is released under the MIT licence ([`LICENSE`](LICENSE)); the data product in `release_data/` is CC BY 4.0.

## Running it

```bash
conda env create -f environment.yml
conda activate akosombo-geoai
pytest tests/ -v
```

The main stages, in order. Each script is run as `python -m src.<package>.<module>` from the repository root; the earlier stages read the YAML files in `config/`.
Earth Engine needs an account and `earthengine authenticate`; set your own project ID in `config/gee_config.yaml`.

| Stage | Scripts |
|---|---|
| Study area and data | `src/aoi/build_aoi.py`; `src/data_ingestion/` (`download_gee_imagery.py`, `download_osm.py`, `load_population.py`, `load_admin_boundaries.py`) |
| Sentinel-1 and Sentinel-2 | `src/preprocessing/build_s1_composites.py`, `s1_scene_20231023.py`, `s1_change_by_date.py`, `preprocess_sentinel2.py` |
| Pre-existing water | `src/flood_mapping/water_masks.py`, `build_sea_mask.py`, `permanent_water_mask.py` |
| Flood extent | `src/flood_mapping/change_counts.py` (release-period counting), `sar_change_detection.py`, `water_index_mapping.py`, `flood_extent_fusion.py` |
| Exposure | `src/exposure/` (`building_exposure.py`, `road_exposure.py`, `population_exposure.py`, `agriculture_exposure.py`, `facility_exposure.py`, `district_exposure.py`) |
| Sensitivity and diagnostics | `src/validation/flood_definition_sensitivity.py`, `s1_change_timeseries.py`, `flood_patch_diagnostics.py`, `release_window_variant.py`, `patch_landuse_check.py` |
| Independent products | `src/validation/compare_unosat.py`, `charter_terrasarx_extent.py`, `charter_tandemx_extent.py`, `charter_mepe_detectability.py`, `definition_reference_matrix.py` |
| Classifiers | `src/models/train_xgboost.py`, `train_unet.py`, `ablation_label_bands.py`, `ablation_pairwise.py`; `src/validation/prior_correct_probability.py` |
| Recovery index | `src/recovery/recovery_index_v2.py` |
| Figures | `src/visualization/` (all share the colours in `palette.py`); `src/figures/build_fig1_matplotlib.py`; `regenerate_24_figures.py` |

`docs/change_based_rebuild_2026-10-13.md` is the running log of the rebuild, with every number and the script that produced it.

## Ethics and data governance

Results are aggregated to districts. No household-level coordinates are published (see `docs/ethics_and_data_governance.md`). The words "exposed" and "proxy" are used on purpose:
the flood map shows where satellite change or water was detected, not damage or measured depth.

## Citation

[To be added when the papers are published.]

## Contact

dzimike.md@gmail.com
