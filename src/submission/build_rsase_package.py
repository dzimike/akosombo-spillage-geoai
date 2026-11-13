from __future__ import annotations
"""
Build the submission package for Remote Sensing Applications: Society and Environment (Elsevier) from docs/manuscript.md.

Output (outputs/submission_RSASE/): title page, anonymised manuscript (line numbers, page numbers, 1.5 spacing, figures and tables in
the text), highlights, cover letter, numbered figure files (PNG and TIFF) and a checklist of what the authors still have to supply.
Formatting rules follow Elsevier's standard guide for authors; the journal's own page could not be read when this was written, so the
checklist marks every rule that should be confirmed against the live guide.
"""
import logging
import re
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from PIL import Image

from src.submission.docx_utils import pandoc, plain_doc, style_doc
from src.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TITLE = ("Change-based flood mapping and exposure after the 2023 Akosombo Dam spillage, Ghana: comparison with independent products "
         "and sensitivity to the flood definition")
RUNNING = "Akosombo spillage flood mapping and exposure"
KEYWORDS = "flood mapping; dam-release flooding; Sentinel-1 SAR; exposure assessment; flood definition; Akosombo Dam; Ghana"
HIGHLIGHTS = [
    "Change against a pre-event baseline separates dam-release flood from existing water",
    "Change-based extent agrees with UNOSAT at F1 0.40; the open-water rule scores 0.13",
    "In the TerraSAR-X sector F1 is 0.63, but 179 flagged Mepe buildings are missed",
    "Exposure is 59 buildings; defensible definitions give 49 to 107",
    "Dry-season change must be kept out of the persistence count",
]
AI_STATEMENT = ("During the preparation of this work the authors used Claude (Anthropic) to help write analysis and figure code and to draft "
                "and edit text. After using this tool, the authors reviewed and edited the content and take full responsibility for the "
                "content of the publication.")


def main() -> None:
    root = get_project_root(); out = root / "outputs/submission_RSASE"; figs = out / "figures"
    if out.exists():
        shutil.rmtree(out)
    figs.mkdir(parents=True)
    t = (root / "docs/manuscript.md").read_text()
    abstract = re.search(r"## Abstract\n\n(.+?)\n\n\*\*Keywords", t, re.S).group(1).strip()
    n_abs = len(abstract.split())
    body = t[t.index("## 1. Introduction"):t.index("## Acknowledgements")].rstrip().rstrip("-").rstrip()
    ack = re.search(r"## Acknowledgements\n\n(.+?)\n\n---", t, re.S).group(1).strip()
    refs = t[t.index("## References"):].strip()
    data_av = re.search(r"## Data Availability Statement\n\n(.+?)\n\n---", t, re.S).group(1).strip()
    n_body = len(body.split())
    assert all(len(h) <= 85 for h in HIGHLIGHTS), [len(h) for h in HIGHLIGHTS]

    # ---- figures, in order of appearance
    names = re.findall(r"!\[\]\(\.\./outputs/figures/([^)]+)\)", body)
    notes = []
    for i, n in enumerate(names, 1):
        src = root / "outputs/figures" / n
        dst = figs / f"Fig{i}.png"; shutil.copy(src, dst)
        im = Image.open(src).convert("RGB"); w, h = im.size
        im.save(figs / f"Fig{i}.tif", compression="tiff_lzw", dpi=(300, 300))
        notes.append((i, n, w, h, w / 7.48))
        body = body.replace(f"![](../outputs/figures/{n})", f"![](../outputs/figures/{n}){{width=16.5cm}}", 1)
    body = body.replace("../outputs/figures/", str(root / "outputs/figures") + "/")

    # ---- anonymised manuscript
    decl = ("## Declaration of competing interest\n\nThe authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.\n\n"
            "## Funding\n\nThis research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.\n\n"
            "## CRediT authorship contribution statement\n\n[The authors to complete with CRediT roles for each author, for example: Name 1: Conceptualization, Methodology, Software, Formal analysis, Visualization, Writing – original draft. Name 2: Supervision, Writing – review and editing.]\n\n"
            f"## Declaration of generative AI and AI-assisted technologies in the writing process\n\n{AI_STATEMENT}\n\n"
            "## Ethics\n\nNot applicable: the study uses satellite and open geospatial data only.\n\n"
            f"## Data availability\n\n{data_av}")
    md = f"# {TITLE}\n\n## Abstract\n\n{abstract}\n\n**Keywords:** {KEYWORDS}\n\n{body}\n\n## Acknowledgements\n\n{ack}\n\n{decl}\n\n{refs}\n"
    md = md.replace("## References", "## References", 1)
    tmp = out / "_manuscript.md"; tmp.write_text(md)
    pandoc(tmp, out / "02_Manuscript_anonymised.docx", root); tmp.unlink()
    style_doc(out / "02_Manuscript_anonymised.docx")

    # ---- title page, highlights, cover letter (plain documents)
    plain_doc(out / "01_Title_page.docx", TITLE, [
        ("Authors: ", "[Author names, with the corresponding author marked]"),
        ("Affiliations: ", "[Department, institution, city, country for each author]"),
        ("Corresponding author: ", "dzimike.md@gmail.com [add postal address and telephone if required]"),
        ("ORCID iDs: ", "[optional]"),
        ("Running title: ", RUNNING),
        ("Keywords: ", KEYWORDS),
        ("Word counts: ", f"abstract {n_abs}; main text from Introduction to Conclusions, including tables and figure captions, {n_body:,}."),
        ("Items: ", f"{len(names)} figures and 5 tables, in the text of the manuscript file; figure files are supplied separately."),
        ("Previous versions and related manuscripts: ", "[The authors to state. A companion manuscript on a machine-learning classifier trained on these labels and a manuscript on uncertainty propagation are in preparation; state their status and cite them if they are submitted or accepted first.]"),
    ])
    plain_doc(out / "03_Highlights.docx", "Highlights", ["• " + h for h in HIGHLIGHTS])
    plain_doc(out / "04_Cover_letter.docx", None, [
        "[Date]", "The Editor-in-Chief\nRemote Sensing Applications: Society and Environment", "Dear Editor,",
        f"We submit the manuscript \"{TITLE}\" for consideration as an original research article in Remote Sensing Applications: Society and Environment.",
        "In September 2023 the Volta River Authority released water from the Akosombo and Kpong dams for six weeks into a floodplain that was already seasonally wet. "
        "Satellite flood maps of such controlled releases are rarely checked against independent products, and an open-water definition counts water that was there before the release. "
        "We map the flooding at 10 m from Sentinel-1 backscatter change against an orbit-matched pre-event baseline, fused with Sentinel-2 water indices, and compare it with four analyst-derived products: "
        "the UNOSAT flood extent, International Charter TerraSAR-X and TanDEM-X maps, and a WorldView-2 map of flooded buildings in Mepe.",
        "The main findings are: (i) the change-based extent (44.2 km²) agrees with UNOSAT at F1 0.40 against 0.13 for an open-water rule, and scores F1 0.63 against TerraSAR-X in its sector; "
        "(ii) a per-date series of 34 Sentinel-1 acquisitions shows that dry-season change must be excluded from the persistence criterion; (iii) none of 179 buildings that analysts flagged as flooded in Mepe falls in the extent, "
        "so 10 m change detection gives a floodplain lower bound, not building-scale exposure; and (iv) exposure (59 buildings, about 5,400 people, 12 km of road) varies substantially across defensible definitions, from 49 to 107 buildings. "
        "The independent products are analyst-derived, disagree with one another, and are not field data; the paper treats them as comparisons, not ground truth, and says so throughout.",
        "The work suits the journal's emphasis on remote sensing applied to a societal problem in a regional case with wider relevance to the management of reservoir releases.",
        "This manuscript is not under consideration elsewhere and has not been published. [The authors to confirm.] "
        "Related work (a machine-learning classifier trained on these labels, and an uncertainty-propagation analysis) is reported in separate manuscripts [status to be stated]. "
        "All authors have approved the submission. [The authors to confirm.] The code and derived datasets are available at https://github.com/dzimike/akosombo-spillage-geoai.",
        "Sincerely,", "[Corresponding author name]\ndzimike.md@gmail.com",
    ], spacing=1.2)
    (out / "figures" / "README.txt").write_text("Fig1-Fig9: numbered as in the text. PNG for review upload; TIFF (LZW, 300 dpi tag) for production.\n")

    # ---- checklist
    low = [f"- Fig{i} ({n}): {w} x {h} px = {d:.0f} dpi at 190 mm width" + ("  <-- below 300 dpi at full width" if d < 300 else "") for i, n, w, h, d in notes]
    (out / "00_Checklist.md").write_text(f"""# RSASE submission checklist

Generated {__import__('datetime').date.today()} from docs/manuscript.md by src/submission/build_rsase_package.py.

## Files
01_Title_page.docx, 02_Manuscript_anonymised.docx (line numbers, page numbers, 1.5 spacing, figures and tables in the text),
03_Highlights.docx, 04_Cover_letter.docx, figures/Fig1-Fig9 (.png and .tif).

## What the authors must supply
- Author names, affiliations, ORCID, postal address (title page).
- CRediT authorship statement (placeholder in the manuscript file). Funding and competing-interest statements say there is nothing to declare.
- Push the public-release branch to https://github.com/dzimike/akosombo-spillage-geoai and confirm the repository is public before submission (the URL is already in the data availability statement and cover letter).
- Confirm the AI-use statement names the right tool and describes the use correctly.
- Status of the two companion manuscripts (cover letter, title page).
- Confirm that the shortened title is acceptable (the original ran to 28 words).
- The NADMO/VRA designation document for the 24 communities shown in Figure 1, in case reviewers ask for it.

## Counts
- Abstract: {n_abs} words (assumed limit 250; confirm).
- Highlights: {len(HIGHLIGHTS)} bullets, {min(len(h) for h in HIGHLIGHTS)}-{max(len(h) for h in HIGHLIGHTS)} characters (Elsevier norm: 3-5 bullets, max 85 characters each).
- Keywords: 7 (Elsevier norm: up to 7).
- Main text: {n_body:,} words including tables and captions.

## Figure files
""" + "\n".join(low) + """

## Rules taken from Elsevier's standard guide, to confirm on the live RSASE page
The journal's own guide (sciencedirect.com/journal/remote-sensing-applications-society-and-environment/publish/guide-for-authors) returned HTTP 403 when this was built,
so none of these could be checked against it: abstract limit, highlights format, whether a graphical abstract is required or optional, single- or double-anonymised review
(this package assumes single-anonymised, authors on the title page only), reference style (the manuscript keeps one consistent author-year style;
Elsevier journals normally accept any consistent style at submission and reformat on acceptance), figure format and resolution, and the APC if open access is chosen.
""")
    log.info("package written to %s (abstract %d words, body %d words)", out, n_abs, n_body)


if __name__ == "__main__":
    main()
