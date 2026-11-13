# Change-based rebuild of the flood labels (2026-10-13)

Decision: after the UNOSAT comparison (`unosat_comparison_2026-10-13.md`) the primary flood definition was changed from
event-window open water (VH < -26.7 dB and VV < -15.2 dB; 72 km2, F1 0.13 against UNOSAT) to SAR backscatter change against
an orbit-matched pre-event median fused with optical water. The pre-change state is archived in `archive/v5_open_water_labels/`.

## Definition (fixed before comparing with any reference product)

* Reference per relative orbit (74, 147): median of 7x7 focal-mean dB scenes, 1 Aug - 10 Sep 2023 (6 scenes).
* Each event-window acquisition (15 Sep - 15 Nov, 4-9 per pixel) is differenced against the reference of its own orbit.
* Changed on a date: dVV < -K dB and dVH < -K dB. SAR flood: changed on >= M dates. K = 3 dB, M = 2.
* Pre-existing water (class 4, 1,501.8 km2): JRC occurrence >= 50 % | DEM open-sea mask | pre-event optical NDWI & MNDWI >= 0 |
  pre-event SAR open water.
* Slope <= 5 deg, MMU 10 px; fused with optical event water (NDWI & MNDWI >= 0) outside class 4.

Code: `src/preprocessing/s1_change_stack.py`, `src/flood_mapping/sar_change_detection.py`, `water_masks.preexisting_water`,
`flood_extent_fusion.py`. Pipeline runner: `logs/rerun/chain_change_rebuild.sh`.

## Results

| Item | Value |
|---|---|
| Flood (classes 1-3) | 53.5 km2 (SAR only 36.9, optical only 9.0, both 7.6) |
| Exposure | 71 buildings, 5,937 people, 14.2 km road, 655 ha cropland proxy, 26 facilities within 500 m |
| UNOSAT FloodExtent (23 Oct), F1 | 0.39 fused, 0.40 SAR mask; previous open-water extent 0.13; single-scene change rule 0.56 |
| K x M grid, F1 against UNOSAT | 0.21 - 0.46 (K=4 dB, 1 date best; not adopted, to avoid circularity) |
| Charter TanDEM-X (20 Oct) | 3.4 km2 flooded; recall of fused 0.10 as published, 0.16 shifted, 0.35 with 50 m tolerance |
| Charter TerraSAR-X (16 Oct, UN-SPIDER, 3 m; 238 km2 NW sector) | map flooded 7.1 km2; fused F1 0.62 (P 0.60, R 0.65; 0.81 within 50 m), SAR mask 0.62, single scene 0.42, optical 0.28, open-water rule 0.21; map JRC layer IoU 0.51 with ours, no shift |
| Charter WorldView-2 Mepe buildings | 179 of 4,215 flagged; none in the extent; AUC VV .55, VH .62, dVV .48, dVH .54, MNDWI .70, XGB .77 |

## Caveats found during the rebuild

* The UFSM sub-area map of 16 Oct (img_28547948.jpg) was digitized by registering it to the UN-SPIDER map (its own graticule labels are rounded and unusable): 10.0 m per pixel, Dice 0.83 flooded / 0.88 water against the UN-SPIDER map, 13 km2 overlap with our raster. It is the same analysis, not an independent reference.
* The TanDEM-X map layers sit ~0.4-0.7 km from the project grid; coregistration used the map's own permanent-water layer vs JRC.
* Raw XGBoost output is not a probability at scene prevalence (177 km2 at 0.50); prior-shift correction in
  `src/validation/prior_correct_probability.py`.
* Persistence (M >= 2) makes the extent an event-window union, not a footprint at one date; the UNOSAT comparison is therefore
  conservative relative to a same-date rule.
* Western scatter of small patches (Shai-Osudoku, Ningo-Prampram) that UNOSAT does not show was not diagnosed (irrigation?).
* Recovery-index outputs are legacy and were not recomputed.


## Update 2026-10-14: release-period persistence (supersedes the numbers above)

A per-date series of 34 Sentinel-1 acquisitions (`src/preprocessing/s1_change_by_date.py`, `src/validation/s1_change_timeseries.py`,
`outputs/figures/fig_change_timeseries.png`) showed that November dry-season change satisfied the two-date persistence rule
without any change during the release: the Ningo-Prampram scatter (889 isolated patches, 4.9 km2, 0.1 % UNOSAT overlap) was
flagged on 6-15 % of the July-August pixels and on 0.4-0.9 from November to January, and the whole-domain flagged area
reached 600 km2 in late January against 40-55 km2 at the flood peak.

Primary definition now: dVV and dVH < -3 dB on >= 2 of the 7 release-period acquisitions (15 Sep - 30 Oct;
`src/flood_mapping/change_counts.py`). Event-window (15 Sep - 15 Nov) state archived in `archive/v6_event_window_persistence/`
and `outputs/tables/*_eventwindow_definition.csv`.

| Item | Event-window two-date | Release-period two-date |
|---|---|---|
| Flood (classes 1-3) | 53.5 km2 (SAR 36.9 / optical 9.0 / both 7.6) | 44.2 km2 (27.6 / 9.5 / 7.1) |
| Exposure | 71 bldgs, 5,937 people, 14.2 km road | 59 bldgs, 5,394 people, 11.8 km road |
| UNOSAT F1 fused / SAR mask | 0.391 / 0.402 | 0.402 / 0.415 |
| TerraSAR-X sector F1 fused / SAR | 0.624 / 0.616 | 0.634 / 0.628 |
| TanDEM-X recall (shifted / 50 m) | 0.095 / 0.26 | 0.095 / 0.26 |
| XGBoost test AUC / F1 | 0.991 / 0.950 | 0.994 / 0.950 |
| XGBoost raw 0.50 map; prior-corrected 0.50 | 177 km2; 35 km2 | 109 km2; 32 km2 |
| Full U-Net pooled test F1 | 0.728 | 0.792 |

Remaining unexplained: 8.9 km2 (20 %) of small patches along the Songor lagoon (Ada East/West) and in Ningo-Prampram with < 3 %
UNOSAT overlap. Recovery index: see `recovery_index_2026-10-14.md`.


## Correction 2026-10-17

`sar_flood_scene_20231023.tif` is the OPEN-WATER rule applied to the 23 Oct scene (73 km2), not a change rule; the Charter
agreement tables had labelled it 'single scene (change rule)'. The real single-scene change mask is
`sar_flood_scene_change_20231023.tif` (`src/validation/scene_change_mask.py`, 49 km2). Corrected scores: TerraSAR-X sector F1 0.59
(change rule) and 0.42 (open-water rule on that scene); TanDEM-X recall 0.31 and 0.29 (0.41 and 0.42 within 50 m). The main paper,
Table 5 and the TerraSAR-X/TanDEM-X paragraphs were corrected. New figures: `fig_method_overview.png`, `fig_alluvial_exposure.png`,
`fig_definition_reference_matrix.png` (data: `definition_reference_matrix.csv`, `exposure_by_detection_class.csv`,
`definition_area_overlap.csv`).

## Figures added 2026-10-17 (second batch)

* `fig_ablation_matrix.png` (data `xgb_ablation_pairwise.csv`, `src/models/ablation_pairwise.py`): six predictor groups removed one and two at a
  time. Test-block F1 stays 0.90-0.95 for every single and pair removal; leave-one-block-out F1 falls by 0.28 without SAR and
  0.49 without SAR and the event optical indices, but by only 0.05 without terrain. Gain importance (elevation 53 %) is not
  necessity: the uncertainty paper text that said the classifier 'relies mostly on elevation' was corrected.
* `fig_case_studies.png`: Mepe (WorldView-2 flagged buildings), Ningo-Prampram (seasonal scatter: 1.1 km2 flagged on 12 Aug, 0.2 km2 on
  23 Oct, 8.8 km2 on 3 Jan in the window), Songor lagoon margin (rectangular basins resembling salt ponds; cause unconfirmed),
  river bars in the TerraSAR-X sector. Added as Figure 8 of the main paper (matrix is now Figure 9); ablation matrix is Figure 5
  of the uncertainty paper.

## Land-use check of the lagoon-margin patches (2026-10-17)

`src/validation/patch_landuse_check.py` (data: OSM landuse in `data/raw/landuse/osm_landuse_songor.gpkg`, 20 salt-pond polygons, 36.5 km2;
ESA WorldCover 2021 via `src/preprocessing/download_worldcover.py`). The 8.96 km2 of unresolved patches (1,784 components, Ada East,
Ada West, Ningo-Prampram, < 3 % UNOSAT overlap): 10.1 % inside OSM salt ponds (background land in the same districts 3.0 %, rest of the
extent 0.2 %), 68.9 % inside OSM wetland polygons (11.9 % and 1.4 %); WorldCover: herbaceous wetland 32 %, bare/sparse 29 %, grassland 27 %,
cropland 7 % (background: grassland 56 %, cropland 30 %, wetland 2 %). The salt-pond reading explains at most about a tenth of the area;
the patches sit in lagoon-fringe wetland and bare flats. Cause (lagoon level, soil moisture, salt crust, or unmapped flooding) not
established. Of the OSM pond area outside pre-existing water (24.8 km2), 3.9 % is in the flood extent.

## Shared palette applied to the older figures (2026-10-17)
`src/visualization/palette.py` is now the single source of colour for every figure script (CLASS, DEFINITION, REFERENCE, ASSET, DISTRICT, AGREE, MODEL, SPLIT, CALIBRATION, PRF, PATCH_GROUP). Ported: study-area figure and the two regenerate_24 study-area maps (district + flood-class colours), exposure by district (asset colours, faded for districts without designated communities), water time series, UNOSAT and TerraSAR-X agreement maps (agreement scheme: both green, ours blue, UNOSAT orange, TerraSAR-X purple), uncertainty figures 2, 3 and the decomposition (calibration raw grey / corrected black, definition colours), DL figures (model colours, split colours, heatmap and probability maps changed from red-green to YlGnBu). No numbers changed. District hues were then moved off the flood-class hues (Ada East violet, North Tongu brown, Central Tongu black, Shai-Osudoku grey) and the DL probability-map 0.5 contour made black.

## Main-paper Figures 2, 6 and 8 reworked (2026-10-18)
Figure 2: research gap and questions removed from the figure (the Introduction already carries them); the three-step description moved into Section 4; step banners shortened; sensitivity-grid axis labels no longer collide. Figure 8: the paragraph beside each row removed (content is in Sections 5.1 and 5.4); only a short row title remains; the XGBoost (0.74) and optical-only U-Net (0.83) AUC for the Mepe buildings moved into the text. Figure 6: legend moved below the map, scale bar added, rarest class kept when down-sampling so small patches stay visible, and two 10 m close-ups added (A river reach Mepe-Adidome: both 8.6, ours only 6.3, UNOSAT only 4.3 km2; B eastern flats and delta: both 3.5, ours only 1.3, UNOSAT only 17.6 km2, i.e. 54% of UNOSAT-only area). Main body 8,989 words.
