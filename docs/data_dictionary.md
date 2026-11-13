# Data Dictionary

## Table: affected_communities

| Column | Type | Description |
|--------|------|-------------|
| community_id | integer | Unique community identifier |
| community_name | string | Community name |
| district | string | Administrative district |
| region | string | Administrative region |
| longitude | float | Centroid longitude (EPSG:4326) |
| latitude | float | Centroid latitude (EPSG:4326) |
| population_estimate | integer | Estimated population (WorldPop or census) |
| primary_livelihood | string | Main livelihood type (farming, fishing, etc.) |
| reported_flooding | boolean | Whether flooding was reported in 2023 |
| data_source | string | Source of community record |
| notes | string | Any caveats or uncertainties |

## Table: flood_extent_polygons

| Column | Type | Description |
|--------|------|-------------|
| flood_id | integer | Unique polygon identifier |
| date_start | date | Start of detection window |
| date_end | date | End of detection window |
| sensor | string | Satellite sensor (Sentinel-1, Sentinel-2) |
| method | string | Detection method (SAR change, MNDWI threshold, fusion) |
| threshold | float | Threshold value used |
| area_km2 | float | Detected flood area in km² |
| confidence_class | string | high / medium / low |
| processing_version | string | Pipeline version |
| notes | string | Any limitations or caveats |

## Table: exposed_buildings

| Column | Type | Description |
|--------|------|-------------|
| building_id | integer | Unique building identifier |
| community_id | integer | Associated community |
| district | string | Administrative district |
| geometry_type | string | Point or Polygon |
| footprint_area_m2 | float | Building footprint area (m²) |
| flood_status | string | exposed / not_exposed |
| flood_date_window | string | Event window label |
| distance_to_river_m | float | Distance to nearest river in metres |
| source | string | Data source (OSM, Google, Microsoft) |
| confidence | string | Source data confidence |

## Table: exposed_roads

| Column | Type | Description |
|--------|------|-------------|
| road_id | integer | Unique road segment ID |
| road_class | string | OSM highway classification |
| district | string | Administrative district |
| length_km | float | Total segment length (km) |
| flooded_length_km | float | Flooded portion length (km) |
| percent_flooded | float | Percentage of segment flooded |
| accessibility_role | string | Role in community access network |
| source | string | Data source |

## Table: community_exposure_summary

| Column | Type | Description |
|--------|------|-------------|
| community_id | integer | Unique community identifier |
| community_name | string | Community name |
| district | string | Administrative district |
| flooded_area_km2 | float | Flooded area within community buffer (km²) |
| exposed_building_count | integer | Count of exposed buildings |
| exposed_road_length_km | float | Flooded road length within community buffer (km) |
| exposed_population_estimate | float | Estimated population in flood zone |
| exposed_cropland_ha | float | Exposed cropland area (hectares) |
| nearest_health_facility_disrupted | boolean | Whether nearest health facility is affected |
| recovery_index | float | Composite recovery index (0–1) |
| vulnerability_score | float | Combined exposure + accessibility vulnerability |

## Table: model_training_pixels

| Column | Type | Description |
|--------|------|-------------|
| sample_id | integer | Unique sample identifier |
| x | float | Easting (EPSG:32630, metres) |
| y | float | Northing (EPSG:32630, metres) |
| flood_label | integer | 0 = not flooded, 1 = flooded |
| elevation | float | Elevation (metres) |
| slope | float | Slope (degrees) |
| twi | float | Topographic Wetness Index |
| distance_to_river | float | Distance to main river (metres) |
| distance_to_dam | float | Distance to Akosombo dam (metres) |
| flow_accumulation | float | Flow accumulation (cells) |
| landcover_class | integer | Land cover class code |
| ndvi_pre | float | Pre-event NDVI |
| ndwi_pre | float | Pre-event NDWI |
| sar_vv_pre | float | Pre-event SAR VV backscatter (dB) |
| sar_vv_event | float | Event SAR VV backscatter (dB) |
| sar_vv_change | float | SAR VV change (event – pre, dB) |
| settlement_density | float | Building density per km² |
| road_density | float | Road length density per km² |
| population_density | float | Population per km² |
| block_id | integer | Spatial block ID for cross-validation |
| split | string | train / validation / test |
