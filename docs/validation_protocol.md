# Validation Protocol

## Flood Extent Validation

**Sampling design**: Stratified random sampling across four classes:
- flooded
- not_flooded
- permanent_water
- uncertain

**Minimum sample size**: 300 points (target: 500–1,000)

**Reference sources** (in order of preference):
1. Field GPS observations with date and class label
2. Drone imagery interpretation
3. High-resolution satellite imagery (Planet, Maxar) visual interpretation
4. Humanitarian assessment maps (UNOCHA, NADMO)
5. Community-reported flood extents (if geocoded)
6. Manually interpreted Sentinel-2 cloud-free composites

**Required CSV columns**: point_id, longitude, latitude, date_observed, class_label, source, confidence, notes

## Exposure Validation

- Building footprint completeness: compare OSM/Microsoft/Google counts against census totals
- Road completeness: compare OSM network length against official road statistics
- Population grid plausibility: compare zonal sums against district census totals
- Facility coordinates: verify against Ministry of Health facility list

## Model Validation

**Random Forest / XGBoost**:
- Spatial block cross-validation (10 km blocks)
- Holdout test set (15% of spatial blocks)
- Metrics: F1, IoU, AUC-ROC, AUC-PR, balanced accuracy

**U-Net** (if used):
- Patch-based validation on held-out spatial blocks
- Metrics: IoU, Dice coefficient, precision, recall

## Uncertainty Documentation

All outputs must state:
- Detection method and threshold used
- Confidence class and limitations
- Sources of potential error
- Data completeness caveats
