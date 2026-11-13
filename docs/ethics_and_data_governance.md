# Ethics and Data Governance

## Principles

1. Do not expose exact household-level vulnerability in public outputs.
   Aggregate sensitive indicators to community or district level.

2. Do not publish identifiable displacement locations unless already public
   and ethically reviewed.

3. Avoid language that blames affected communities for their exposure.
   Exposure results from structural factors: dam siting, floodplain settlement
   history, infrastructure inequality, and dam-release decisions.

4. Distinguish controlled spillage governance decisions from climate-induced
   inflow pressures. Attribute responsibility accurately.

5. Use careful language around loss, displacement, and damage:
   - "exposed buildings", not "destroyed buildings" (unless confirmed)
   - "estimated affected population", not "displaced population" (unless verified)
   - "cropland exposed to floodwater", not "crop loss" (unless validated)

6. If field data are collected from community members, obtain prior informed consent.
   Protect the identities and coordinates of vulnerable households.

7. Clearly state uncertainty in all remote-sensing classifications.
   Remote sensing is an approximation, not ground truth.

8. Share outputs in forms accessible and useful to local planners, NADMO,
   district assemblies, and affected communities, not only to academic audiences.

## Data Sources and Licensing

- OpenStreetMap: Open Database Licence (ODbL) – attribution required
- Sentinel-1/2 (ESA Copernicus): Open access – attribution required
- SRTM DEM (NASA/USGS): Public domain
- WorldPop: Creative Commons Attribution 4.0
- GADM: For non-commercial use only
- JRC Global Surface Water: Free for non-commercial research

## Data Storage

- No raw satellite data should be committed to the repository (see .gitignore)
- No personal data or household coordinates should be committed
- Credential files and service account keys must never be committed

## Publication Ethics

- All claims in the manuscript must be traceable to pipeline outputs or cited literature
- All limitations in Section 28 of CLAUDE.md must be acknowledged in the paper
- Co-authorship should reflect genuine intellectual contributions
- Acknowledge Volta River Authority and affected communities in acknowledgements
