# Recovery index: re-run on the release-period labels and rebuilt as v2 (2026-10-14)

Scope note: the recovery index is not part of the three narrowed manuscripts; this note records what was done and how far
the ranking can be trusted.

## v1 (src/recovery/recovery_index_24.py) on the new flood mask

Re-run after `ndvi_recovery`, `ndwi_recovery`, `accessibility_recovery` on the release-period flood extent (44.2 km2).
The index remained effectively a two-component score:

* road accessibility is 1.0 for every community (no flooded road segment inside any buffer is wet post-event; the
  component is vacuous), settlement proxy 0.998-1.000; both fall under the MIN_INFORMATIVE_RANGE guard and are set to 0.5;
* the 10 km buffers overlap, so neighbouring communities average almost the same flood pixels (all seven South Tongu
  communities and Agordome sit at ranks 17-24);
* NDVI recovery and MNDWI recession are limited by cloud and by the 47 % valid pre-event optical composite.

Top of v1: Agbove (1), Agbeve (2), Alikekope (3), Ada (4), Adidome (5); bottom: Tadze (24), Fievie (23), Tadame (22).

## v2 (src/recovery/recovery_index_v2.py)

* Service areas: Voronoi cells of the 24 communities intersected with their 10 km buffers (non-overlapping).
  Areas with < 500 flood pixels (5 ha) are not assessed; Akuse and Torgorme lie outside the raster. 22 communities scored.
* Components (weights 0.4 / 0.3 / 0.3; Dirichlet(2) Monte Carlo, 2,000 draws, seed 42):
  1. SAR recovery = 1 - net persistence. Net persistence = [flagged share of flood pixels - flagged share of non-flooded
     land in the same area] on post-event dates (16 Nov 2023 - 31 Jan 2024) divided by the same quantity on release-period
     dates. The control term removes dry-season change, which flags hundreds of km2 after November and made a first
     version (no control) give persistence ratios above 1.
  2. NDVI recovery ratio (post/event, clipped to [0, 1]) on flood pixels.
  3. MNDWI recession on flood pixels.
* Road accessibility and settlement proxy are dropped (no information).

## Results

* Top five: Ada 0.92, Alikekope 0.91, Fievie 0.75, Kewum Atrobinya 0.73, Mepe 0.72; bottom: Vume 0.00, Mafi-Dugame 0.12,
  Sokpoe 0.16, Tadame 0.18, Asutsure 0.19 (`outputs/tables/community_recovery_index_v2.csv`,
  `outputs/figures/fig_recovery_index_v2.png`).
* The three components agree (Spearman 0.63-0.78 across communities), so the index carries a consistent signal.
* v1 and v2 ranks agree only weakly (Spearman 0.30 over the 22 communities): Fievie moves from 23 to 3, Tadze from 24 to 9,
  Mafi-Dugame from 8 to 21 and Asutsure from 10 to 18 (v1 ranks counted over all 24). The v1 ranking was driven by buffer overlap and should not be
  quoted.
* Mean Monte Carlo rank SD under weight uncertainty is 1.2 positions, but this is not the dominant uncertainty: the method
  change moved many communities by more than 10 places.

## Caveats

* No validation against recovery data exists; the index is a relative remote-sensing score, not a measure of livelihoods.
* Flood-pixel counts per area range from 601 (Asutsure) to 58,285 (Battor); six areas rest on fewer than 3,300 pixels.
* Persistence ratio exceeded 1 for several communities even with the control (clipped to recovery 0), which indicates
  remaining seasonal or land-use change in the flooded pixels.
* The pre-event optical composite is valid on 47 % of the domain; NDVI recovery and MNDWI recession use post/event
  composites with their own cloud gaps.
