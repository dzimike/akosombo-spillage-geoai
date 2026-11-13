from __future__ import annotations
"""
Submission packages for the uncertainty paper (Journal of Flood Risk Management, Wiley/CIWEM) and the deep-learning paper (Geocarto International,
Taylor & Francis), built from docs/uncertainty_manuscript.md and docs/dl_manuscript.md.

Rules applied come from each journal's author guidelines, read on 2026-10-20:
  JFRM      single-blind; title page separate; Word main text with title, abstract, up to eight keywords, body, references (APA), tables with titles and
            footnotes, figure legends as a list; figures as separate files; 6,000-word limit for original papers excluding references, tables and figures;
            data availability statement required; AI use disclosed; running title under 40 characters; ORCID of the submitting author required.
  Geocarto  double-anonymous review; research article of at most 25 pages including tables, references and captions; unstructured abstract of 200 words;
            3-6 keywords; order: title page, abstract, keywords, text, acknowledgments, declaration of interest, references, tables (one per page), figures,
            figure captions; any consistent reference style; figures at 300 dpi (colour) at final size; funding, disclosure, generative-AI and data
            availability statements.

Output: outputs/submission_JFRM_uncertainty/ and outputs/submission_EJRS_DL/ (build_geocarto is kept for reference; the DL paper now targets
the European Journal of Remote Sensing, single-anonymous, abstract 150-200 words, 2-6 keywords, T&F APA, no stated length limit).
"""
import datetime
import logging
import re
import shutil

from PIL import Image

from src.submission.docx_utils import pandoc, plain_doc, style_doc
from src.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
REPO = "https://github.com/dzimike/akosombo-spillage-geoai"
PAGEBREAK = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```'
NO_FUNDING = "The authors declare that no funds, grants, or other support were received during the preparation of this manuscript."
NO_COI = "The authors have no relevant financial or non-financial interests to disclose."

GEOCARTO_ABSTRACT = (
    "Satellite flood mapping is critical for disaster response, yet the contribution of different data sources to deep learning "
    "segmentation of dam-release inundation in West Africa is poorly characterised. We apply a U-Net to the 15 September–30 October 2023 Akosombo Dam "
    "spillage along Ghana's Lower Volta River, using a 28-band stack of Sentinel-1 SAR, Sentinel-2, terrain, hydrological and settlement features on a 10 m grid, "
    "with labels from Sentinel-1 backscatter change and optical water outside pre-existing water (44.2 km²). A four-experiment ablation (SAR 5 bands, "
    "optical 7, terrain 21, all 28) is trained on one spatial block partition and repeated on three more. On the main partition the full U-Net reached "
    "pooled test F1 0.792 against 0.680 (SAR-only), 0.636 (optical-only) and 0.528 (terrain-only), but none recovered more than 0.69 of the SAR-change-only "
    "class. A prior-corrected pixel-level XGBoost matched the full U-Net on the main partition (0.754 against 0.750); over four partitions the full U-Net "
    "averaged 0.701 and the corrected XGBoost 0.724. No model reliably separated 179 buildings flagged as flooded in Mepe from others (ROC AUC 0.64 to 0.83). "
    "One event is used; the labels share inputs with the predictors.")
KEYWORDS = {
    "JFRM": ["flood mapping", "uncertainty propagation", "threshold sensitivity", "flood definition", "exposure", "Akosombo Dam", "spatial cross-validation", "Lower Volta Basin"],
    "GEO": ["flood segmentation", "U-Net", "Sentinel-1 SAR", "ablation study", "spatial cross-validation", "Akosombo Dam"],
}
TITLES = {
    "JFRM": "Sensitivity of Satellite-Derived Flood Exposure to the Classification Threshold and the Flood Definition: A Case Study of the 2023 Akosombo Dam Spillage, Ghana",
    "GEO": "Deep Learning Semantic Segmentation of Dam-Release Flood Inundation Using Multi-Source Satellite Features: A U-Net Ablation Study of the 2023 Akosombo Spillage, Ghana",
}


# ------------------------------------------------------------------------------------------------ references (APA 7)
def _initials(s: str) -> str:
    return re.sub(r"(?<=\.)(?=[A-Z])", " ", s)


def apa_reference(e: str) -> str:
    """Convert 'Surname, I.J., Surname, K., 2017. Title. Journal 65, 126-139.' (and web/dataset entries) to APA 7."""
    manual = {
        "Chen, T.": "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM.",
        "Long, J.": "Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 3431–3440). IEEE.",
        "Bonafilia, D.": "Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops* (pp. 210–211). IEEE.",
        "Ronneberger, O.": "Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention* (Vol. 9351, pp. 234–241). Springer.",
    }
    for k, v in manual.items():
        if e.startswith(k):
            return v
    m = re.match(r"^(?P<au>.+?),\s*(?P<yr>\d{4}[a-z]?\.|n\.d\.)\s+(?P<rest>.+)$", e, re.S)
    if not m:
        return e
    au, yr, rest = m.group("au"), m.group("yr").rstrip("."), m.group("rest")
    if m.group("yr") == "n.d.":
        yr = "n.d."
    people = re.findall(r"([^,]+?),\s*((?:[A-Z]\.-?)+)(?:,\s*|$)", au + ",") if re.search(r",\s*[A-Z]\.", au) else None
    if people:
        names = [f"{s.strip()}, {_initials(i)}" for s, i in people]
        if len(names) > 20:
            names = names[:19] + ["..."] + [names[-1]]
            who = ", ".join(names[:19]) + ", . . . " + names[-1]
        elif len(names) == 1:
            who = names[0]
        elif len(names) == 2:
            who = f"{names[0]}, & {names[1]}"
        else:
            who = ", ".join(names[:-1]) + f", & {names[-1]}"
    else:
        who = au
    date = f"({yr})"
    org = people is None
    if org:
        who = who.rstrip(".") + "."
    rest = re.sub(r"\s*\(accessed [^)]*\)\.?$", "", rest).strip()
    j = re.match(r"^(?P<title>.+?[.?])\s+(?P<jour>[A-Z][A-Za-z&' ,\-]*?)\s+(?P<vol>\d+)(?:\((?P<iss>\d+)\))?,\s*(?P<pg>[\dA-Za-z\-–]+)\.?(?:\s+(?P<doi>https?://\S+))?$", rest)
    if j and not rest.startswith(("Manuscript",)):
        pg = j.group("pg").replace("-", "–")
        loc = f"*{j.group('jour')}*, *{j.group('vol')}*" + (f"({j.group('iss')})" if j.group("iss") else "")
        loc += f", {pg}" if "–" in pg else f", Article {pg}"
        return f"{who} {date}. {j.group('title').rstrip('.')}. {loc}." + (f" {j.group('doi')}" if j.group("doi") else "")
    t = re.match(r"^(?P<title>.+?[.?])(?:\s+|$)(?P<tail>.*)$", rest, re.S)
    if t:
        return f"{who} {date}. *{t.group('title').rstrip('.')}*. {t.group('tail')}".strip()
    return f"{who} {date}. {rest}"


# ------------------------------------------------------------------------------------------------ parsing
def parse(root, name):
    t = (root / "docs" / name).read_text()
    abstract = re.search(r"## Abstract\n\n(.+?)\n\n\*\*Keywords", t, re.S).group(1).strip()
    i, j = t.index("## 1. Introduction"), t.index("## Acknowledgements")
    body = t[i:j].rstrip().rstrip("-").rstrip()
    ack = re.search(r"## Acknowledgements\n\n(.+?)\n\n---", t, re.S).group(1).strip()
    data_av = re.search(r"## Data Availability\n\n(.+?)\n\n---", t, re.S).group(1).strip()
    ref_block = t[t.index("## References"):]
    refs = [l.strip() for l in ref_block.split("\n") if l.strip() and not l.startswith(("#", "---"))]
    blocks = body.split("\n\n")
    text, figs, tabs, k = [], [], [], 0
    while k < len(blocks):
        b = blocks[k]
        if b.startswith("!["):
            path = re.search(r"\]\(\.\./outputs/figures/([^)]+)\)", b).group(1)
            cap = blocks[k + 1]; assert cap.startswith("**Figure"), cap[:40]
            figs.append((path, re.sub(r"^\*\*Figure (\d+)\.\*\*\s*", r"Figure \1. ", cap)))
            k += 2; continue
        if b.startswith("**Table"):
            tab = blocks[k + 1]; assert tab.startswith("|"), tab[:40]
            note = ""
            if k + 2 < len(blocks) and blocks[k + 2].startswith("*") and not blocks[k + 2].startswith("**"):
                note = blocks[k + 2]; k += 1
            tabs.append((b, tab, note)); k += 2; continue
        if b.strip() == "---":
            k += 1; continue
        text.append(b); k += 1
    return dict(title=re.match(r"# (.+)", t).group(1).strip(), abstract=abstract, body="\n\n".join(text), ack=ack, data_av=data_av, refs=refs, figs=figs, tabs=tabs)


def figure_files(root, out, figs, dpi_target):
    figdir = out / "figures"; figdir.mkdir(parents=True)
    notes = []
    for i, (path, _) in enumerate(figs, 1):
        src = root / "outputs/figures" / path
        shutil.copy(src, figdir / f"Fig{i}.png")
        im = Image.open(src).convert("RGB"); w, h = im.size
        im.save(figdir / f"Fig{i}.tif", compression="tiff_lzw", dpi=(300, 300))
        notes.append((i, path, w, h, w / (170 / 25.4)))
    return notes


def wc(s):
    return len(re.findall(r"\S+", s))


def body_word_count(body):
    return wc("\n".join(l for l in body.split("\n") if not l.startswith("|")))


# ------------------------------------------------------------------------------------------------ JFRM
def build_jfrm(root):
    out = root / "outputs/submission_JFRM_uncertainty"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    p = parse(root, "uncertainty_manuscript.md")
    kw = KEYWORDS["JFRM"]; assert len(kw) <= 8
    title = TITLES["JFRM"]; running = "Uncertainty in flood exposure estimates"; assert len(running) < 40
    n_abs = wc(p["abstract"]); n_body = body_word_count(p["body"])
    figs = figure_files(root, out, p["figs"], 300)
    body = re.sub(r"\(([A-Z][A-Za-z\-]+) and ([A-Z][A-Za-z\-]+), (\d{4})\)", r"(\1 & \2, \3)", p["body"])
    refs = sorted((apa_reference(r) for r in p["refs"]), key=lambda r: re.sub(r"[^a-z ]", "", r.lower()))
    data_av = p["data_av"]
    tables = "\n\n".join(f"{c}\n\n{tb}" + (f"\n\n{n}" if n else "") for c, tb, n in p["tabs"])
    legends = "\n\n".join(c for _, c in p["figs"])
    md = (f"# {title}\n\n## Abstract\n\n{p['abstract']}\n\n**Keywords:** {'; '.join(kw)}\n\n{body}\n\n"
          f"## Data Availability Statement\n\n{data_av}\n\n## Conflict of Interest Statement\n\n{NO_COI}\n\n"
          f"## Use of Generative AI\n\nDocumented in Section 4.4 (Methods).\n\n"
          f"## References\n\n" + "\n\n".join(refs) + f"\n\n{PAGEBREAK}\n\n## Tables\n\n{tables}\n\n{PAGEBREAK}\n\n## Figure Legends\n\n{legends}\n")
    tmp = out / "_m.md"; tmp.write_text(md)
    pandoc(tmp, out / "02_Main_text.docx", root); tmp.unlink()
    style_doc(out / "02_Main_text.docx", line_numbers=True, spacing=1.5, body_pt=11)
    plain_doc(out / "01_Title_page.docx", title, [
        ("Running title: ", running + f" ({len(running)} characters)"),
        ("Authors: ", "[Full names, with the corresponding author marked]"),
        ("Affiliations: ", "[Institution, department, city, country where the work was done; CIWEM membership grade where applicable, e.g. MCIWEM]"),
        ("Corresponding author: ", "[Name], dzimike.md@gmail.com; ORCID iD required for the submitting author: [add]"),
        ("Acknowledgments: ", p["ack"] + " " + NO_FUNDING),
        ("Counts: ", f"abstract {n_abs} words; main text {n_body:,} words (Introduction to Conclusions, headings included, excluding tables and figure and table captions); {len(figs)} figures and {len(p['tabs'])} tables."),
    ])
    plain_doc(out / "03_Cover_letter.docx", None, [
        "[Date]", "The Editor-in-Chief\nJournal of Flood Risk Management", "Dear Editor,",
        f"We submit \"{title}\" as an Original Paper.",
        "Satellite flood maps are routinely intersected with infrastructure and population layers to estimate exposure, but the sensitivity of those estimates to analyst choices is rarely reported. "
        "For the 2023 Akosombo Dam spillage in Ghana we propagate two such choices, the classification threshold of a machine-learning flood classifier and the definition of the flood label, into exposure of buildings, roads and people. "
        "The raw classifier score overstates flooded area 2.5-fold because of the sampling design; after a prior-shift correction, thresholds from 0.50 to 0.75 still change flooded area by 34%, buildings by 26% and flooded road by 53%. "
        "Changing the label definition moves exposure by a similar amount (29 to 97 km², 49 to 107 buildings, 7 to 32 km of road). A ten-split repeat of the predictor ablation shows that a single test split understates the classifier's dependence on SAR. "
        "The results bear directly on how exposure from satellite flood maps should be reported for flood risk management.",
        "Related work, disclosed as the journal's policies require: this paper uses the flood map of a companion mapping paper (the 2023 Akosombo flood extent and its comparison with independent products), and a third manuscript reports a deep-learning comparison on the same labels [status of each to be stated by the authors]. "
        "The analyses and results differ; where the first paper's results are needed they are summarised here with a citation. All three were prepared with the assistance of generative AI as described in Section 4.4.",
        "The manuscript is not under consideration elsewhere and has not been published, and all authors have approved the submission [the authors to confirm]. The code is available at " + REPO + ". No material requiring third-party permission is reproduced.",
        "Suggested reviewers: [optional, to be added by the authors].", "Sincerely,", "[Corresponding author name]\ndzimike.md@gmail.com",
    ], spacing=1.2)
    (out / "figures" / "README.txt").write_text("Fig1-Fig5 as cited in the text, PNG and TIFF; legends are in the main text file (Figure Legends) and must also be entered at upload.\n")
    low = [f"- Fig{i} ({n}): {w} x {h} px, {d:.0f} dpi at 170 mm" for i, n, w, h, d in figs]
    (out / "00_Checklist.md").write_text(f"""# JFRM submission checklist (uncertainty paper)

Built {datetime.date.today()} by src/submission/build_journal_packages.py. Rules: https://onlinelibrary.wiley.com/page/journal/1753318x/homepage/forauthors.html (read 2026-10-20).

## Files
01_Title_page.docx, 02_Main_text.docx (title, abstract, keywords, text, statements, APA references, tables, figure legends; line-numbered), 03_Cover_letter.docx, figures/ (separate files).

## Needed from the authors
- Names, affiliations, CIWEM grades, **ORCID iD of the submitting author (required)**; running title is in the title page (editable).
- **Primary citation for the 24 designated communities** (Section 2 and the Fig. 1 legend still carry a bracketed placeholder).
- Status of the two other manuscripts (cover letter). Suggested reviewers (optional). Article publication charge: the journal is fully open access (CC BY); the amount and waivers are on Wiley's APC page (a third-party listing gave about $2,970; unverified).
- Funding is declared in the Acknowledgments (title page) as none; competing interests: none.
- Data availability statement is in the text; it names a GitHub repository. The journal's own statement options are on Wiley's data-sharing page.

## Checks against the guidelines
- Word limit for original papers: 6,000 excluding references, tables and figures. Main text here: {n_body:,} words counting headings, excluding tables and figure/table captions ({wc(p['body'])} with everything).
- Abstract {n_abs} words (the page states no limit); {len(kw)} keywords (limit 8); title has no abbreviations; running title {len(running)} characters (limit 40).
- References converted to APA 7; in-text citations use '&' for two authors. Web sources lose their access dates (APA does not require them).
- Single-blind review: names go on the title page only. Colour: no red-green pairs in these figures (Fig. 4 uses a single-hue red scale).
- AI use is disclosed in Methods 4.4 and in the statements.

## Figures
""" + "\n".join(low) + """

Known weak points: Figs 2, 3 and 5 were drawn about 10-14 inches wide, so 8-9 pt lettering prints at roughly 5-6 pt at one column; the guidelines give no minimum lettering size. Wiley's detailed post-acceptance figure rules were not on the page that was read.
""")
    return n_body


# ------------------------------------------------------------------------------------------------ Geocarto
def build_geocarto(root):
    out = root / "outputs/submission_Geocarto_DL"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    p = parse(root, "dl_manuscript.md")
    kw = KEYWORDS["GEO"]; assert 3 <= len(kw) <= 6
    n_abs = wc(GEOCARTO_ABSTRACT); assert n_abs <= 200, n_abs
    title = TITLES["GEO"]
    figs = figure_files(root, out, p["figs"], 300)
    anon = lambda s: s.replace(REPO, "[repository URL withheld for review]")
    body = anon(p["body"])
    data_av = anon(p["data_av"])
    refs = []
    for r in p["refs"]:
        r2 = anon(r)
        refs.append(r2)
    refs.sort(key=lambda r: re.sub(r"[^a-z ]", "", r.lower()))
    tables = f"\n\n{PAGEBREAK}\n\n".join(f"{c}\n\n{tb}" + (f"\n\n{n}" if n else "") for c, tb, n in p["tabs"])
    captions = "\n\n".join(c for _, c in p["figs"])
    md = (f"# {title}\n\n## Abstract\n\n{GEOCARTO_ABSTRACT}\n\n**Keywords:** {'; '.join(kw)}\n\n{body}\n\n"
          f"## Disclosure statement\n\nThe authors report there are no competing interests to declare.\n\n"
          f"## Funding\n\nNo funding was obtained for this work.\n\n"
          f"## Declaration of generative AI use\n\nGenerative AI (Claude, Anthropic) assisted in writing the model, training and figure-generation code and in drafting and editing text; details are in Section 4.8. The authors reviewed and edited the content and take full responsibility for it. No generative AI was used to create images.\n\n"
          f"## Data availability statement\n\n{data_av}\n\n"
          f"## References\n\n" + "\n\n".join(refs) + f"\n\n{PAGEBREAK}\n\n## Tables\n\n{tables}\n\n{PAGEBREAK}\n\n## Figure captions\n\n{captions}\n")
    tmp = out / "_m.md"; tmp.write_text(md)
    pandoc(tmp, out / "02_Manuscript_anonymised.docx", root); tmp.unlink()
    style_doc(out / "02_Manuscript_anonymised.docx", line_numbers=True, spacing=1.5, body_pt=11)
    n_all = wc(md)
    plain_doc(out / "01_Title_page.docx", title, [
        ("Authors: ", "[Full names]"), ("Affiliations: ", "[Where the research was conducted]"),
        ("Corresponding author: ", "[Name], dzimike.md@gmail.com"), ("ORCiD: ", "[for each author, where available]"),
        ("CRediT roles: ", "[to be entered in the submission portal for each author]"),
        ("Acknowledgments: ", p["ack"]), ("Funding details: ", "No funding was obtained for this work."),
        ("Disclosure statement: ", "The authors report there are no competing interests to declare."),
        ("Repository (withheld in the anonymised manuscript): ", REPO),
        ("Counts: ", f"abstract {n_abs} words; manuscript about {n_all:,} words in total including references, tables and captions; {len(figs)} figures and {len(p['tabs'])} tables."),
    ])
    plain_doc(out / "03_Cover_letter.docx", None, [
        "[Date]", "The Editor\nGeocarto International", "Dear Editor,",
        f"We submit \"{title}\" as a Research Article.",
        "We ask which Sentinel data streams a U-Net needs to segment dam-release flooding and how the answer compares with a calibrated pixel-level classifier. Using a 28-band stack at 10 m and labels from Sentinel-1 backscatter change and Sentinel-2 water indices, "
        "we run a four-experiment ablation on one spatial block partition and repeat it on three more, and we repeat the XGBoost comparison over ten. The full U-Net is the best U-Net on every partition, but a prior-corrected XGBoost is level with it "
        "(F1 0.724 against 0.701 on average), the U-Net ranking is less secure on other partitions than on the first, and no model separates analyst-flagged flooded buildings reliably. We state these limits in the paper.",
        "Related work, disclosed as the journal asks: the labels come from a companion mapping paper and an uncertainty-propagation paper uses the same classifier family [status of each to be stated by the authors]. They report different analyses and results. All three were prepared with generative AI assistance, declared in the manuscript.",
        "The manuscript is not under consideration elsewhere and has not been published; all authors have approved it [the authors to confirm]. The review is double anonymous, so the repository link is withheld in the manuscript and given on the title page: " + REPO + ".",
        "Sincerely,", "[Corresponding author name]\ndzimike.md@gmail.com",
    ], spacing=1.2)
    (out / "figures" / "README.txt").write_text("Fig1-Fig7 as cited in the text, PNG and TIFF (RGB, 300 dpi tag; colour figures need 300 dpi at final size).\n")
    low = [f"- Fig{i} ({n}): {w} x {h} px, {d:.0f} dpi at 170 mm" for i, n, w, h, d in figs]
    pages_est = n_all / 330
    (out / "00_Checklist.md").write_text(f"""# Geocarto International submission checklist (DL paper)

Built {datetime.date.today()} by src/submission/build_journal_packages.py. Rules: https://www.tandfonline.com/action/authorSubmission?show=instructions&journalCode=tgei20 (read 2026-10-20).

## Files
01_Title_page.docx (names, acknowledgments, funding, repository), 02_Manuscript_anonymised.docx (no names, acknowledgments or repository URL; tables one per page; figure captions listed at the end), 03_Cover_letter.docx, figures/.

## Needed from the authors
- Names, affiliations, ORCiDs, CRediT roles (entered in the portal), corresponding author.
- **Primary citation for the 24 designated communities** (Section 2 and the Fig. 1 caption still carry a bracketed placeholder).
- Status of the two other manuscripts (cover letter).
- Article publishing charge: the page lists $3,230 / £2,392 / EUR 2,875 plus tax; discounts and waivers are described on the journal page.
- Data deposition: T&F asks for a DOI or persistent identifier at submission if a data set is associated; the paper cites the GitHub repository and states that weights and probability rasters are not distributed.

## Checks against the instructions
- Double-anonymous review: names, acknowledgments and the repository URL are out of the manuscript; self-citations appear as 'Authors (2026b)' and must be changed only after acceptance.
- **Abstract rewritten to {n_abs} words** (limit 200, unstructured) from the 250-word version, dropping the partition counts and some secondary numbers; the 250-word abstract remains in docs/dl_manuscript.md.
- {len(kw)} keywords (3-6). Order of parts follows the instructions. Reference style: one consistent style kept (the journal applies its own after acceptance).
- Statements present: disclosure, funding, generative AI, data availability. SI units used.

## Length risk
The limit is 25 pages including tables, references, captions and footnotes. The manuscript is about {n_all:,} words in total; at a typical 300-350 words per double-spaced page that is roughly {pages_est:.0f} pages before tables are paged separately. The page count was not measured. If it is over, the cuts I would make first are Table 1 (training summary), Section 6.2 and part of Section 5.4.

## Figures
""" + "\n".join(low) + """

Known weak points: Fig. 5 (probability maps) and Figs 1-4 were drawn 10-13 inches wide and print with 4-6 pt lettering at 170 mm; the instructions give no lettering size, only 300 dpi for colour and 600 dpi for grayscale at final size, which these meet.
""")
    return n_all

# ------------------------------------------------------------------------------------------------ European Journal of Remote Sensing
EJRS_HIGHLIGHTS = [
    "On one spatial partition, a U-Net using Sentinel-1, Sentinel-2, terrain and settlement layers separated flooded from dry pixels better than any single data stream (F1 0.792 against 0.680 for SAR alone).",
    "A prior-corrected pixel-level classifier matched the full U-Net (0.754 against 0.750), and over four partitions it was slightly ahead (0.724 against 0.701), so a deep network is not automatically the better tool.",
    "No model reliably identified which of 179 buildings in Mepe were flagged as flooded (ROC AUC 0.64 to 0.83), so these maps should not drive building-level decisions.",
]


def build_ejrs(root):
    out = root / "outputs/submission_EJRS_DL"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    p = parse(root, "dl_manuscript.md")
    kw = KEYWORDS["GEO"]; assert 2 <= len(kw) <= 6
    n_abs = wc(GEOCARTO_ABSTRACT); assert 150 <= n_abs <= 200, n_abs
    n_hl = sum(wc(h) for h in EJRS_HIGHLIGHTS); assert 3 <= len(EJRS_HIGHLIGHTS) <= 5
    title = TITLES["GEO"]
    figs = figure_files(root, out, p["figs"], 300)
    refs = sorted(p["refs"], key=lambda r: re.sub(r"[^a-z ]", "", r.lower()))
    tables = f"\n\n{PAGEBREAK}\n\n".join(f"{c}\n\n{tb}" + (f"\n\n{n}" if n else "") for c, tb, n in p["tabs"])
    captions = "\n\n".join(c for _, c in p["figs"])
    hl = "\n".join(f"- {h}" for h in EJRS_HIGHLIGHTS)
    md = (f"# {title}\n\n## Abstract\n\n{GEOCARTO_ABSTRACT}\n\n**Key policy highlights**\n\n{hl}\n\n**Keywords:** {'; '.join(kw)}\n\n{p['body']}\n\n"
          f"## Declaration of interest statement\n\nThe authors report there are no competing interests to declare.\n\n"
          f"## Funding\n\nNo funding was obtained for this work.\n\n"
          f"## Declaration of generative AI use\n\nGenerative AI (Claude, Anthropic) assisted in writing the model, training and figure-generation code and in drafting and editing text; details are in Section 4.8. The authors reviewed and edited the content and take full responsibility for it. No generative AI was used to create images.\n\n"
          f"## Data availability statement\n\n{p['data_av']}\n\n"
          f"## References\n\n" + "\n\n".join(refs) + f"\n\n{PAGEBREAK}\n\n## Tables\n\n{tables}\n\n{PAGEBREAK}\n\n## Figure captions\n\n{captions}\n")
    tmp = out / "_m.md"; tmp.write_text(md)
    pandoc(tmp, out / "02_Manuscript.docx", root); tmp.unlink()
    style_doc(out / "02_Manuscript.docx", line_numbers=True, spacing=1.5, body_pt=11)
    n_all = wc(md)
    plain_doc(out / "01_Title_page.docx", title, [
        ("Authors: ", "[Full names]"), ("Affiliations: ", "[Where the research was conducted]"),
        ("Corresponding author: ", "[Name], dzimike.md@gmail.com"), ("ORCiD: ", "[for each author, where available]"),
        ("CRediT roles: ", "[to be entered in the submission portal for each author]"),
        ("Acknowledgments: ", p["ack"]), ("Funding details: ", "No funding was obtained for this work."),
        ("Disclosure statement: ", "The authors report there are no competing interests to declare."),
        ("Repository: ", REPO),
        ("Counts: ", f"abstract {n_abs} words; key policy highlights {n_hl} words; manuscript about {n_all:,} words in total including references, tables and captions; {len(figs)} figures and {len(p['tabs'])} tables."),
    ])
    plain_doc(out / "03_Cover_letter.docx", None, [
        "[Date]", "The Editor\nEuropean Journal of Remote Sensing", "Dear Editor,",
        f"We submit \"{title}\" as a Research Article.",
        "We ask which Sentinel data streams a U-Net needs to segment dam-release flooding and how the answer compares with a calibrated pixel-level classifier. Using a 28-band stack at 10 m and labels from Sentinel-1 backscatter change and Sentinel-2 water indices, "
        "we run a four-experiment ablation on one spatial block partition and repeat it on three more, and we repeat the XGBoost comparison over ten. The full U-Net is the best U-Net on every partition, but a prior-corrected XGBoost is level with it "
        "(F1 0.724 against 0.701 on average), the U-Net ranking is less secure on other partitions than on the first, and no model separates analyst-flagged flooded buildings reliably. We state these limits in the paper.",
        "Related work: the labels come from a mapping paper and an uncertainty-propagation paper uses the same classifier family [status of each to be stated by the authors]. They report different analyses and results. All three were prepared with generative AI assistance, declared in the manuscript.",
        "The manuscript is not under consideration elsewhere and has not been published; all authors have approved it [the authors to confirm]. Code and the derived label and feature tables are at " + REPO + "; model weights and probability rasters are not distributed, as the data availability statement explains.",
        "Sincerely,", "[Corresponding author name]\ndzimike.md@gmail.com",
    ], spacing=1.2)
    (out / "figures" / "README.txt").write_text("Fig1-Fig7 as cited in the text, PNG and TIFF (RGB, 300 dpi tag; colour figures need 300 dpi at final size).\n")
    low = [f"- Fig{i} ({n}): {w} x {h} px, {d:.0f} dpi at 170 mm" for i, n, w, h, d in figs]
    (out / "00_Checklist.md").write_text(f"""# European Journal of Remote Sensing submission checklist (DL paper)

Built {datetime.date.today()} by src/submission/build_journal_packages.py. Rules: https://www.tandfonline.com/action/authorSubmission?show=instructions&journalCode=tejr20 (read 2026-11-13).

## Files
01_Title_page.docx (names, acknowledgments, funding, repository), 02_Manuscript.docx (tables one per page, figure captions listed at the end), 03_Cover_letter.docx, figures/.

## Needed from the authors
- Names, affiliations, ORCiDs, CRediT roles (entered in the portal), corresponding author.
- **Primary citation for the 24 designated communities** (Section 2 and the Fig. 1 caption still carry a bracketed placeholder).
- Status of the two other manuscripts (cover letter).
- Article publishing charge: $1,815 / GBP 1,344 / EUR 1,615 plus tax (open access); waivers and agreements are described on the journal page.
- Whether the journal wants the 3-5 key policy highlights: the page says to check; they are drafted ({n_hl} words) and sit under the abstract, delete them if the editor says not.

## Checks against the instructions
- Single-anonymous review, so the repository link stays in the manuscript. No stated page or word limit for a Research Article.
- Abstract {n_abs} words (150-200, unstructured); {len(kw)} keywords (2-6).
- Order: title page, abstract, keywords, main text (Introduction, Materials and methods, Results, Discussion), acknowledgments, declaration of interest, references (T&F APA 7, converted from the numbered list), tables on individual pages, figures separate, captions as a list.
- Statements present: disclosure, funding, generative AI, data availability. The journal applies a share-upon-reasonable-request policy; the paper says what is public (code, labels, feature tables) and what is not (weights, probability rasters).
- Single quotation marks and consistent spelling are asked for: the manuscript uses British/American spelling as written, not checked line by line.
- Figures: 300 dpi colour at final size.

## Figures
""" + "\n".join(low) + """

Known weak points: Fig. 5 (probability maps) and Figs 1-4 were drawn 10-13 inches wide and print with 4-6 pt lettering at 170 mm; redraw at final width if time allows.
""")
    return n_all


def main() -> None:
    root = get_project_root()
    n1 = build_jfrm(root); n2 = build_ejrs(root)
    log.info("JFRM main text %d words; EJRS manuscript about %d words", n1, n2)


if __name__ == "__main__":
    main()
