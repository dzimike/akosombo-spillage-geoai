from __future__ import annotations
"""
Build the submission package for Natural Hazards (Springer Nature) from docs/manuscript.md.

Rules applied are those on the journal's submission-guidelines page (read 2026-10-18): Word file with line numbers and page numbers, abstract
150-250 words, 4-6 keywords, decimal headings to three levels, author-date citations and reference list, figure captions in the text ("Fig. 1" in
bold, no closing punctuation), figure parts in lowercase letters, 'Statements and Declarations' after the references, AI use documented in
the Methods, acknowledgements on the title page.

Output: outputs/submission_NaturalHazards/ (title page, manuscript, cover letter, figure files, checklist).
"""
import datetime
import logging
import re
import shutil

from PIL import Image

from src.submission.docx_utils import pandoc, plain_doc, style_doc
from src.submission.springer_style import convert_citations, convert_reference, strip_caption_periods
from src.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TITLE = ("Change-based flood mapping and exposure after the 2023 Akosombo Dam spillage, Ghana: comparison with independent products "
         "and sensitivity to the flood definition")
KEYWORDS = "Flood mapping; Dam-release flooding; Sentinel-1 SAR; Exposure assessment; Flood definition; Ghana"
# Springer: no undefined abbreviations in the abstract
ABSTRACT_FIXES = [
    ("from Sentinel-1 backscatter change against an orbit-matched pre-event median (VV and VH both down more than 3 dB on at least two release-period acquisitions)",
     "from Sentinel-1 synthetic aperture radar (SAR) backscatter change against an orbit-matched pre-event median (VV and VH polarisations both down more than 3 dB on at least two release-period acquisitions)"),
    ("agreed with the UNOSAT extent at F1 0.13", "agreed with the UNOSAT (United Nations Satellite Centre) flood extent at an F1 score of 0.13"),
]
AI_METHODS = ("### 4.5 Use of generative AI\n\n"
              "Generative AI (Claude, Anthropic) assisted in writing analysis and figure-generation code and in drafting and editing text. The authors specified the "
              "analyses, ran the code, checked the reported numbers against the output tables, and take full responsibility for the content. No generative AI was used to create images.")
ACCESSED = "Accessed 12–13 October 2026"


def main() -> None:
    root = get_project_root(); out = root / "outputs/submission_NaturalHazards"; figs = out / "figures"
    if out.exists():
        shutil.rmtree(out)
    figs.mkdir(parents=True)
    t = (root / "docs/manuscript.md").read_text()
    abstract = re.search(r"## Abstract\n\n(.+?)\n\n\*\*Keywords", t, re.S).group(1).strip()
    for a, b in ABSTRACT_FIXES:
        assert a in abstract, a[:40]
        abstract = abstract.replace(a, b)
    n_abs = len(abstract.split())
    assert 150 <= n_abs <= 250, n_abs
    kw = [k.strip() for k in KEYWORDS.split(";")]; assert 4 <= len(kw) <= 6
    body = t[t.index("## 1. Introduction"):t.index("## Acknowledgements")].rstrip().rstrip("-").rstrip()
    ack = re.search(r"## Acknowledgements\n\n(.+?)\n\n---", t, re.S).group(1).strip()
    data_av = re.search(r"## Data Availability Statement\n\n(.+?)\n\n---", t, re.S).group(1).strip()
    ref_block = t[t.index("## References"):]

    # ---- AI use in the Methods (end of section 4, before section 5)
    i5 = body.index("## 5. Results")
    body = body[:i5].rstrip().rstrip("-").rstrip() + "\n\n" + AI_METHODS + "\n\n---\n\n" + body[i5:]
    n_body = len(body.split())

    # ---- figures in order of appearance; width 174 mm (single-column text area)
    names = re.findall(r"!\[\]\(\.\./outputs/figures/([^)]+)\)", body)
    notes = []
    for i, n in enumerate(names, 1):
        src = root / "outputs/figures" / n
        shutil.copy(src, figs / f"Fig{i}.png")
        im = Image.open(src).convert("RGB"); w, h = im.size
        im.save(figs / f"Fig{i}.tif", compression="tiff_lzw", dpi=(300, 300))
        notes.append((i, n, w, h, w / (174 / 25.4)))
        body = body.replace(f"![](../outputs/figures/{n})", f"![](../outputs/figures/{n}){{width=17.4cm}}", 1)
    body = body.replace("../outputs/figures/", str(root / "outputs/figures") + "/")
    body = strip_caption_periods(convert_citations(body))

    # ---- references: one alphabetical list in Springer style; web sources carry an access date
    web_prefix = ("Ankah", "Asaase", "International Charter", "Citi Newsroom", "UNOSAT")
    refs = []
    for line in ref_block.split("\n"):
        line = line.strip()
        if not line or line.startswith(("#", "---", "**News")):
            continue
        r = convert_reference(line)
        if r.startswith(web_prefix):
            r += f". {ACCESSED}"
        refs.append(r)
    refs.sort(key=lambda r: re.sub(r"[^a-z ]", "", r.lower()))
    ref_md = "## References\n\n" + "\n\n".join(refs)

    decl = ("## Statements and Declarations\n\n"
            "**Funding** The authors declare that no funds, grants, or other support were received during the preparation of this manuscript.\n\n"
            "**Competing interests** The authors have no relevant financial or non-financial interests to disclose.\n\n"
            "**Author contributions** [The authors to complete, for example: All authors contributed to the study conception and design. Material preparation, data collection and analysis were performed by [full name]. "
            "The first draft of the manuscript was written by [full name] and all authors commented on previous versions of the manuscript. All authors read and approved the final manuscript.]\n\n"
            f"**Data availability** {convert_citations(data_av)}\n\n"
            "**Code availability** The analysis code is in the repository named under Data availability (version 1.0.0).\n\n"
            "**Ethics approval and consent to participate** Not applicable: the study uses satellite and open geospatial data only, and reports results aggregated to districts.\n\n"
            "**Use of generative AI** Documented in Section 4.5.")
    md = f"# {TITLE}\n\n## Abstract\n\n{abstract}\n\n**Keywords** {'; '.join(kw)}\n\n{body}\n\n{ref_md}\n\n{decl}\n"
    tmp = out / "_manuscript.md"; tmp.write_text(md)
    pandoc(tmp, out / "02_Manuscript.docx", root); tmp.unlink()
    style_doc(out / "02_Manuscript.docx", line_numbers=True, spacing=1.5, body_pt=11)

    plain_doc(out / "01_Title_page.docx", TITLE, [
        ("Authors: ", "[Author names, in submission order]"),
        ("Affiliations: ", "[Department, institution, city, (state), country for each author]"),
        ("Corresponding author: ", "[Name], dzimike.md@gmail.com"),
        ("ORCID iDs: ", "[16-digit ORCID for each author, if available]"),
        ("Acknowledgements: ", ack),
        ("Counts: ", f"abstract {n_abs} words; main text from Introduction to Conclusions, including tables and figure captions, {n_body:,} words; "
                      f"{len(names)} figures and 5 tables."),
    ])
    plain_doc(out / "03_Cover_letter.docx", None, [
        "[Date]", "The Editor-in-Chief\nNatural Hazards", "Dear Editor,",
        f"We submit \"{TITLE}\" for consideration as an original research article in Natural Hazards.",
        "In September 2023 the Volta River Authority released water from the Akosombo and Kpong dams for six weeks into a floodplain that was already seasonally wet. "
        "Satellite flood maps of such controlled releases are rarely checked against independent products, and an open-water definition counts water that was there before the release. "
        "We map the flooding at 10 m from Sentinel-1 backscatter change against an orbit-matched pre-event baseline, fused with Sentinel-2 water indices, and compare the result with four analyst-derived products: "
        "the UNOSAT flood extent, International Charter TerraSAR-X and TanDEM-X maps, and a WorldView-2 map of flooded buildings in Mepe.",
        "The main findings are: (i) the change-based extent (44.2 km²) agrees with the UNOSAT extent at F1 0.40 against 0.13 for an open-water rule, and scores F1 0.63 against TerraSAR-X in its sector; "
        "(ii) a per-date series of 34 Sentinel-1 acquisitions shows that dry-season change must be excluded from the persistence criterion; (iii) none of 179 buildings that analysts flagged as flooded in Mepe falls in the extent, "
        "so 10 m change detection gives a floodplain lower bound, not building-scale exposure; and (iv) exposure (59 buildings, about 5,400 people, 12 km of road) varies substantially with the flood definition, from 49 to 107 buildings. "
        "The independent products are analyst-derived, disagree with one another, and are not field data; the paper treats them as comparisons, not ground truth.",
        "The topic fits Natural Hazards: remote-sensing hazard mapping and exposure assessment for a human-induced flood hazard in West Africa.",
        "Transparency on related work, as the guidelines ask: two further manuscripts use the same flood map. One reports a machine-learning classifier trained on labels derived from it, and one reports how the classifier threshold "
        "and the flood definition propagate into exposure [status of each to be stated by the authors]. They report different analyses and results, and neither is published. "
        "Fig. 5a (Mepe) reproduces a crop of an International Charter WorldView-2 map; [the authors to state that permission was obtained, or that the panel was replaced].",
        "This manuscript is not under consideration elsewhere and has not been published, and all authors have approved the submission [the authors to confirm]. "
        "Code and the derived flood map are at https://github.com/dzimike/akosombo-spillage-geoai.",
        "Suggested reviewers (institutional e-mail addresses required): [to be added by the authors]. Reviewers to exclude: [optional].",
        "Sincerely,", "[Corresponding author name]\ndzimike.md@gmail.com",
    ], spacing=1.2)
    (figs / "README.txt").write_text("Fig1-Fig9, numbered as cited in the text. TIFF (RGB, 8 bit, LZW) and PNG. Created with Python 3.11 and matplotlib; no EPS (preferred for vector) was produced.\n")

    low = [f"- Fig{i} ({n}): {w} x {h} px = {d:.0f} dpi at 174 mm" for i, n, w, h, d in notes]
    (out / "00_Checklist.md").write_text(f"""# Natural Hazards submission checklist

Generated {datetime.date.today()} by src/submission/build_springer_package.py from docs/manuscript.md. Rules are from the journal's submission-guidelines page
(https://link.springer.com/journal/11069/submission-guidelines), read on 2026-10-18.

## Files
01_Title_page.docx, 02_Manuscript.docx (line numbers, page numbers, 1.5 spacing, figures and tables in the text), 03_Cover_letter.docx, figures/.

## Needed from the authors before submission
- Author names, affiliations, ORCID iDs, corresponding-author name (title page, cover letter).
- Author-contribution statement (placeholder in 'Statements and Declarations'). Funding and competing-interests statements say there is nothing to declare.
- **Primary citation for the 24 designated communities** (NADMO/VRA document: issuer, title, date, URL). Section 2 still carries a bracketed placeholder.
- **Permission for Fig. 5a**: it contains a crop of a Charter WorldView-2 map (NASRDA). The guidelines require evidence of permission for material
  published elsewhere. If permission is not obtained, replace panel a with a schematic or drop it (and the Section 5.4 text that cites Fig. 5a).
- Companion-manuscript status (cover letter), suggested reviewers with institutional e-mail addresses.
- The VRA 2023 Annual Report was not found online; if you have it, cite it in Section 3.1 and check the peak-release date.
- Confirm the repository DOI/version: create the v1.0.0 tag, or deposit on Zenodo and cite the DOI.

## Checks done
- Abstract {n_abs} words (limit 150-250); 6 keywords (limit 4-6); abstract abbreviations defined (SAR, UNOSAT, F1 score).
- Headings are decimal, three levels at most. Captions: 'Fig. n' bold, no closing period; panel letters lowercase; 'Fig.' in the text.
- Citations are name-year in parentheses; reference list alphabetical, DOIs as links, journal names in full.
- Statements and Declarations follow the references; AI use is documented in Section 4.5 (Methods).
- Acknowledgements are on the title page, not in the manuscript.
- Main text {n_body:,} words (the journal states no word limit on the guidelines page).

## Known departures from the guidelines
- Figure files: raster only (PNG and TIFF), {min(d for *_, d in notes):.0f}-{max(d for *_, d in notes):.0f} dpi at 174 mm, so the 600 dpi minimum for combination art is met; no EPS (preferred for vector) was produced.
{chr(10).join(low)}
- Lettering size: Figs 2 and 8 were redrawn at final width (174 mm, 600 dpi). Their panel titles are 6.2-6.6 pt, legends 4.8-5.4 pt and in-panel notes about 5 pt:
  readable, but below the 8-12 pt the guidelines ask for. Sixteen panels (Fig. 5) or twelve plus a banner (Fig. 2) cannot carry larger text at one column width;
  reaching 8 pt would mean fewer panels per figure (for example splitting Fig. 5 into two figures).
- Captions mention colours; if the journal prints figures in black and white, they should be reworded (the guidelines say not to refer to colour then).
- Some figures also carry panel titles; the guidelines say not to put titles or captions inside illustrations.
- The page does not say whether review is single- or double-blind; author names are only on the title page.
""")
    log.info("package written to %s (abstract %d, body %d words)", out, n_abs, n_body)


if __name__ == "__main__":
    main()
