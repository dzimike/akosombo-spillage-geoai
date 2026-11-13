"""
SUPERSEDED — do not run. This QGIS-console workflow for fig1_study_area.png
was replaced by src/figures/build_fig1_matplotlib.py (Ghana inset, no
embedded title, scale bar, north arrow, full headless matplotlib rebuild)
precisely because this script requires an interactive QGIS Python console
and cannot run headlessly. It is kept only for historical reference; running
it would export a stale, pre-fix version of fig1_study_area.png, the figure
currently embedded in docs/manuscript.md. Use build_fig1_matplotlib.py.

Original docstring: Rebuild Figure 1 — run from QGIS Python console after
loading akosombo_study_area.qgz
"""
from qgis.core import *
from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtCore import QRectF
from qgis.PyQt.QtWidgets import QApplication
import os
import time

project = QgsProject.instance()
layout  = project.layoutManager().layoutByName("Figure1_StudyArea")

# ── 1. Clean existing non-permanent items ─────────────────────────────────
keep = (QgsLayoutItemMap, QgsLayoutItemScaleBar, QgsLayoutItemPicture)
removed = 0
for it in list(layout.items()):
    if isinstance(it, QgsLayoutItem) and not isinstance(it, keep):
        layout.removeLayoutItem(it)
        removed += 1
QApplication.processEvents()
print(f"Step 1: removed {removed} old items")

# ── Constants ─────────────────────────────────────────────────────────────
MAP_X, MAP_Y, MAP_W, MAP_H = 8.0, 14.0, 152.0, 168.0
# Raise south extent to 5.83°N — trims ocean/estuary strip, keeps all communities
EXT_W, EXT_E, EXT_S, EXT_N = 0.01, 0.84, 5.83, 6.45

# ── 2. Update map item extent ─────────────────────────────────────────────
for item in layout.items():
    if isinstance(item, QgsLayoutItemMap):
        item.setExtent(QgsRectangle(EXT_W, EXT_S, EXT_E, EXT_N))
        item.attemptMove(QgsLayoutPoint(MAP_X, MAP_Y, QgsUnitTypes.LayoutMillimeters))
        item.attemptResize(QgsLayoutSize(MAP_W, MAP_H, QgsUnitTypes.LayoutMillimeters))
        item.invalidateCache()
        item.refresh()
        break
# Reposition scale bar + north arrow to bottom of frame
for item in layout.items():
    if isinstance(item, QgsLayoutItemScaleBar):
        item.attemptMove(QgsLayoutPoint(11, 172, QgsUnitTypes.LayoutMillimeters))
    elif isinstance(item, QgsLayoutItemPicture):
        item.attemptMove(QgsLayoutPoint(148, 167, QgsUnitTypes.LayoutMillimeters))
QApplication.processEvents()
print("Step 2: extent updated")

# ── Helpers ───────────────────────────────────────────────────────────────
def geo(lon, lat):
    x = MAP_X + (lon - EXT_W) / (EXT_E - EXT_W) * MAP_W
    y = MAP_Y + (EXT_N - lat) / (EXT_N - EXT_S) * MAP_H
    return x, y

def add_lbl(text, x, y, w, h, pt=7, bold=False, color="#111111"):
    item = QgsLayoutItemLabel(layout)
    item.setText(text)
    f = QFont("Arial", int(pt)); f.setBold(bold)
    item.setFont(f); item.setFontColor(QColor(color))
    item.attemptSetSceneRect(QRectF(x, y, w, h))
    layout.addLayoutItem(item)

def add_dot(cx, cy, d, fill):
    item = QgsLayoutItemShape(layout)
    item.setShapeType(QgsLayoutItemShape.Ellipse)
    item.setSymbol(QgsFillSymbol.createSimple(
        {"color": fill, "outline_color": "#ffffff", "outline_width": "0.6"}))
    item.attemptSetSceneRect(QRectF(cx - d/2, cy - d/2, d, d))
    layout.addLayoutItem(item)

def add_rect(x, y, w, h, fill, oc="#888888", ow="0.3", os_="solid"):
    item = QgsLayoutItemShape(layout)
    item.setShapeType(QgsLayoutItemShape.Rectangle)
    item.setSymbol(QgsFillSymbol.createSimple(
        {"color": fill, "outline_color": oc, "outline_width": ow, "outline_style": os_}))
    item.attemptSetSceneRect(QRectF(x, y, w, h))
    layout.addLayoutItem(item)

def add_hline(x, y, w, color, style="solid"):
    add_rect(x, y, w, 0.5, "0,0,0,0", oc=color, ow="0.8", os_=style)

# ── 3. Title ──────────────────────────────────────────────────────────────
add_lbl(
    "Figure 1. Lower Volta Basin: flood extent and dam-spillage affected communities, "
    "2023 Akosombo Dam spillage, Ghana",
    8, 1.5, 232, 7, pt=9, bold=True)
QApplication.processEvents()
print("Step 3: title added")

# ── 4. Community markers and labels ──────────────────────────────────────
xform = QgsCoordinateTransform(
    QgsCoordinateReferenceSystem("EPSG:32630"),
    QgsCoordinateReferenceSystem("EPSG:4326"), project)

comms = []
for layer in project.mapLayers().values():
    if 'communit' in layer.name().lower():
        for feat in layer.getFeatures():
            pt = feat.geometry().asPoint()
            ll = xform.transform(QgsPointXY(pt.x(), pt.y()))
            comms.append((feat["community_name"], ll.x(), ll.y(),
                          feat["flooded_point"], feat["flood_class_pt"]))
        break

# dx<0 → label right-aligned left of dot; dx>0 → label left-aligned right of dot
OFFSETS = {
    "Torgorme":        ( 1, -5.0), "Akuse":           (-1,  3.5),
    "Asutsure":        ( 1,  3.5), "Alabonu":         ( 1, -5.0),
    "Kewum Atrobinya": (-1, -5.0), "Volivo Landor":   ( 1, -5.0),
    "Battor":          (-1, -5.0), "Mepe":            ( 1, -5.0),
    "Devime":          (-1,  3.5), "Adidome":         ( 1, -5.0),
    "Mafi-Dugame":     ( 1,  3.5), "Fievie":          ( 1, -5.0),
    "Tadame":          (-1, -5.0), "Vume":            (-1,  3.5),
    "Tefle":           ( 1,  3.5), "Sogakope":        ( 1, -5.0),
    "Sokpoe":          ( 1,  3.5), "Agordome":        ( 1, -5.0),
    "Tadze":           (-1, -5.0), "Asidovui":        ( 1, -5.0),
    "Agbeve":          ( 1, -5.0), "Ada":             (-1, -5.0),
    "Agbove":          (-1, -5.0), "Alikekope":       ( 1, -5.0),
}
CW = 1.9  # mm per char, Arial 7pt

placed = 0
for name, lon, lat, fp, fc in comms:
    mx, my = geo(lon, lat)
    # Skip if outside map frame
    if not (MAP_X - 1 <= mx <= MAP_X + MAP_W + 1 and
            MAP_Y - 1 <= my <= MAP_Y + MAP_H + 1):
        print(f"  skip {name}")
        continue
    mx = max(MAP_X + 3, min(mx, MAP_X + MAP_W - 3))
    my = max(MAP_Y + 3, min(my, MAP_Y + MAP_H - 3))

    color = "#cc0000" if fp == 1 else ("#888888" if fc == 255 else "#f59200")
    diam  = 5.0 if fp == 1 else (2.8 if fc == 255 else 3.8)
    add_dot(mx, my, diam, color)

    dx, dy = OFFSETS.get(name, (1, -5.0))
    tw = len(name) * CW + 2.0
    lx = (mx + diam/2 + 0.8) if dx > 0 else (mx - diam/2 - 0.5 - tw)
    ly = my + dy
    lx = max(MAP_X + 0.3, min(lx, MAP_X + MAP_W - tw - 0.3))
    ly = max(MAP_Y + 0.3, min(ly, MAP_Y + MAP_H - 5.0))
    add_lbl(name, lx, ly, tw, 4.5, pt=7)
    placed += 1

QApplication.processEvents()
print(f"Step 4: placed {placed} communities")

# ── 5. Legend ─────────────────────────────────────────────────────────────
lx0, ly0, lw, x0 = 165.0, 14.0, 68.0, 168.0
add_rect(lx0, ly0, lw, 145.0, "255,255,255,230", "#888888", "0.4")
add_lbl("Legend", x0, ly0+2, lw-5, 5.5, pt=9, bold=True)

add_lbl("Dam Spill Communities (24)", x0, ly0+9.5, lw-5, 4.5, pt=7.5, bold=True)
add_dot(x0+2.5, ly0+16.5, 4.5, "#cc0000")
add_lbl("Core inundated (n=6)",        x0+7, ly0+14.5, lw-10, 4.5, pt=7)
add_dot(x0+2.5, ly0+23.0, 3.8, "#f59200")
add_lbl("Flood-zone community (n=18)", x0+7, ly0+21.0, lw-10, 4.5, pt=7)

add_lbl("Dam Locations", x0, ly0+29.5, lw-5, 4.5, pt=7.5, bold=True)
star = QgsLayoutItemLabel(layout)
star.setText("★"); star.setFont(QFont("Arial", 13))
star.setFontColor(QColor("#e8a000"))
star.attemptSetSceneRect(QRectF(x0, ly0+35.5, 8.0, 6.5))
layout.addLayoutItem(star)
add_lbl("Akosombo & Kpong Dams", x0+9, ly0+36.5, lw-12, 4.0, pt=7)

add_lbl("Infrastructure", x0, ly0+44.5, lw-5, 4.5, pt=7.5, bold=True)
add_hline(x0+1, ly0+51.5, 10, "#aaaaaa");         add_lbl("Roads (OSM)",     x0+13, ly0+49.5, lw-16, 4.0, pt=7)
add_hline(x0+1, ly0+57.5, 10, "#4a90d9");         add_lbl("Waterways (OSM)", x0+13, ly0+55.5, lw-16, 4.0, pt=7)
add_hline(x0+1, ly0+63.5, 10, "#333333", "dash"); add_lbl("Study Area AOI",  x0+13, ly0+61.5, lw-16, 4.0, pt=7)

add_lbl("Flood Confidence (5-class)", x0, ly0+70, lw-5, 4.5, pt=7.5, bold=True)
for i, (col, txt) in enumerate([
        ("#e8e8e8","Not flooded"),   ("#2196F3","SAR only"),
        ("#90CAF9","Optical only"),  ("#B71C1C","SAR + optical"),
        ("#1A237E","Permanent water"),("#222222","No data")]):
    oy = ly0+76.5+i*8
    add_rect(x0, oy, 9.5, 6.0, col)
    add_lbl(txt, x0+12, oy+0.5, lw-14, 5.0, pt=7)

QApplication.processEvents()
print("Step 5: legend done")

# ── 6. Attribution + export ───────────────────────────────────────────────
add_lbl(
    "CRS: WGS 84 (EPSG:4326)  |  Flood: Sentinel-1/2 Sep–Oct 2023  |  "
    "20-km AOI  |  All 24 communities designated by NADMO/VRA",
    8, 205, 232, 5, pt=6, color="#555555")

QApplication.processEvents()

for item in layout.items():
    if isinstance(item, QgsLayoutItemMap):
        item.invalidateCache(); item.refresh()
QApplication.processEvents()

out = os.path.join(os.environ.get("AKOSOMBO_ROOT", os.getcwd()), "outputs/figures/fig1_study_area.png")
s = QgsLayoutExporter.ImageExportSettings(); s.dpi = 300
r = QgsLayoutExporter(layout).exportToImage(out, s)
project.write()
print(f"Step 6: export result {r}  (0=success)")
