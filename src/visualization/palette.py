"""Shared colour scheme for the three manuscripts, so a quantity has the same colour in every figure.

Detection classes: SAR change only, optical only, both. Definitions and reference products have their own hues.
Colour-blind-safe (Okabe-Ito based).
"""
CLASS = {"sar": "#0072B2", "optical": "#E69F00", "both": "#009E73", "pre": "#B3D9F2", "none": "#EEEEEE"}
CLASS_LABEL = {"sar": "SAR change only", "optical": "Optical water only", "both": "SAR and optical", "pre": "Pre-existing water"}
DEFINITION = {"change": "#0072B2", "openwater": "#D55E00", "event_window": "#56B4E9", "scene": "#CC79A7"}
REFERENCE = {"unosat": "#444444", "tsx": "#8E24AA", "tdx": "#7E57C2", "mepe": "#F0A030"}
ASSET = {"area": "#555555", "buildings": "#CC79A7", "people": "#D55E00", "roads": "#56B4E9"}
# District hues deliberately avoid the flood-class hues (blue / amber / green / pale blue) so that a map showing both stays readable.
DISTRICT = {"Ada East": "#6A3D9A", "North Tongu": "#8C6D46", "Central Tongu": "#222222", "South Tongu": "#D55E00",
            "Shai-Osudoku": "#7F7F7F", "Lower Manya Krobo": "#CC79A7"}

# ---- added when the palette was applied to the older figures -------------------------------------------------
FLOOD_EXTENT = DEFINITION["change"]            # a single-colour flood extent is the change-based extent
AGREE = {"both": "#2E7D32", "ours": DEFINITION["change"], "unosat": "#E07020", "tsx": "#8E24AA",
         "pre": CLASS["pre"], "none": "#F4F4F4"}                 # ours vs a reference: agreement / ours only / reference only
MODEL = {"full": CLASS["both"], "sar_only": CLASS["sar"], "optical_only": CLASS["optical"], "terrain_only": "#8C6D46",
         "xgboost": "#333333"}                                   # a U-Net given SAR only / optical only / everything
SPLIT = {"train": "#BDBDBD", "val": "#CC79A7", "test": "#D55E00"}
CALIBRATION = {"raw": "#9E9E9E", "corrected": "#222222"}      # raw classifier score vs prior-corrected probability
PRF = {"precision": "#7A7A7A", "recall": "#B0B0B0", "f1": "#000000"}
PATCH_GROUP = {"adjacent": DEFINITION["change"], "isolated": "#009E73", "ningo": DEFINITION["openwater"], "adawest": "#CC79A7"}
DESIGNATED = {"yes": 1.0, "no": 0.45}                           # bar opacity for districts with designated communities
