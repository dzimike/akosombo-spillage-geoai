# Released data product

`fused_flood_confidence.tif` is the fused change-based flood map of the 2023 Akosombo/Kpong spillage described in the main paper.
It is built only from open data (Sentinel-1, Sentinel-2, JRC Global Surface Water, Copernicus GLO-30, OpenStreetMap); it does not use any
UNOSAT or International Charter product. Licence: CC BY 4.0 (the code is MIT).

| | |
|---|---|
| CRS | EPSG:32630 (WGS 84 / UTM 30N) |
| Grid | 10 m, 6,978 columns x 8,793 rows; extent E 852,740-922,520 m, N 620,530-708,460 m |
| Type, nodata | uint8, 255 |

| Value | Class | Pixels (area) |
|---|---|---|
| 0 | not flooded | 45,897,452 (4,590 km²) |
| 1 | SAR change only | 276,327 (27.6 km²) |
| 2 | optical water only | 95,236 (9.5 km²) |
| 3 | SAR change and optical water | 70,599 (7.1 km²) |
| 4 | pre-existing water (JRC occurrence >= 50 %, sea mask, pre-event optical and SAR water) | 15,017,940 (1,502 km²) |

The flood extent of the paper is classes 1-3 (44.2 km²). SAR change: VV and VH both more than 3 dB below an orbit-matched pre-event median on at
least two of seven release-period dates (15 September-30 October 2023). The map is a floodplain lower bound, not a building-scale product, and
it is not validated against field data.
