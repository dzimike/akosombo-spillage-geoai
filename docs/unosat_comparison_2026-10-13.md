# Comparison with the UNOSAT flood extent FL20231018GHA (13 October 2026)

**Source.** UNOSAT / United Nations Satellite Centre, "Satellite detected water extent over Greater Accra & Volta Region, Ghana as of 23 October 2023" (code FL20231018GHA), Humanitarian Data Exchange, licence CC BY-SA. Sentinel-1 scene of 23 Oct 2023 18:10 UTC; analyst-reviewed, preliminary, not validated in the field. Files used: `FL20231018GHA_SHP.zip` (2.5 MB) and the population-exposure spreadsheet (13.6 KB), downloaded from unosat.org to `data/raw/validation/unosat/` (git-ignored; metadata saved in `metadata.txt`). Layers: FloodExtent (53.4 km2), WaterExtent (667 km2), AnalysisExtent (4,311 km2).

**Caveat.** UNOSAT is Sentinel-1 based and covers one date, so this is a check against an independent processing chain, not a sensor-independent or field validation.

## Results (common domain = UNOSAT analysis extent intersected with our raster, sea excluded)

| Our extent vs UNOSAT FloodExtent (52.0 km2) | Area km2 | Overlap km2 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Fused extent (event composite) | 70.0 | 7.9 | 0.11 | 0.15 | 0.13 |
| SAR flood mask (event composite) | 59.8 | 4.3 | 0.07 | 0.08 | 0.08 |
| New water only | 20.3 | 6.2 | 0.31 | 0.12 | 0.17 |
| Our open-water rule on the 23 Oct scene | 72.5 | 15.5 | 0.21 | 0.30 | 0.25 |
| Change rule dVV < -3 dB (untuned) | 67.8 | 30.5 | 0.45 | 0.60 | 0.51 |
| Change rule dVV and dVH < -3 dB (untuned) | 48.4 | 27.7 | 0.57 | 0.54 | 0.56 |

- 82 % of UNOSAT flood lies in none of our fused extent, SAR mask or permanent water; 2 % lies in our permanent water.
- Total water: 98 % of our land water lies inside UNOSAT WaterExtent; we capture 67 % of its 214 km2 (F1 0.79).
- UNOSAT flood pixels: scene VV median -16.3 dB, VH -24.5 dB (dry land -9.7 / -16.2); pre-event VV -11.9, VH -18.5; only 4 % of those missed by our map have NDWI >= 0; median change -4.3 dB (VV), -5.2 dB (VH).
- No single backscatter threshold reproduces it (VV < -13.0 dB recovers 96 % but flags 270 km2).
- Exposure within the UNOSAT analysis extent: ours 123 buildings / 6.9 km road / 7,903 people; UNOSAT 61 / 25.3 / 8,290; change rule 30 / 16.6 / 7,090.
- District flood area (ours vs UNOSAT, km2): South Tongu 3.3 vs 15.5; Anloga 3.7 vs 10.7; North Tongu 6.6 vs 8.8; Ada East 29.4 vs 7.8; Ada West 17.6 vs 1.4; Central Tongu 4.6 vs 4.0.

**Interpretation.** Our open-water extent is dominated by lagoon-wetland water that UNOSAT does not call flood, and it misses newly flooded floodplain at moderate backscatter that UNOSAT (apparently a change-based product) maps. A conventional untuned change rule agrees four times better than our open-water rule.

Scripts: `src/preprocessing/s1_scene_20231023.py`, `src/validation/compare_unosat.py`, `unosat_diagnostics.py`, `unosat_change_rule.py`, `unosat_exposure.py`. Tables: `outputs/tables/unosat_*.csv`. Figure: `outputs/figures/fig6_unosat_agreement.png`.
