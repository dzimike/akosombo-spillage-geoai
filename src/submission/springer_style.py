from __future__ import annotations
"""Convert the manuscript's APA-like references and citations to Springer Nature's basic author-date style (Natural Hazards)."""
import re

# Entries that are not plain journal articles are written out by hand (keyed by the start of the source entry).
MANUAL = {
    "Angnuureng, B. D.": "Angnuureng BD, Ofosu GT, Ayimbisa AE, Almar R (2024) Evaluating the consequences of the 2023 Akosombo Dam spillage into the Volta River on Fuveme Beach, Ghana (West Africa). In: Phillips MR, Al-Naemi S, Duarte CM (eds) Coastlines under global change: proceedings from the International Coastal Symposium (ICS) 2024, Doha, Qatar. J Coast Res Special Issue 113, pp 1021–1025. https://doi.org/10.2112/JCR-SI113-200.1",
    "Authors. (2026)": "Authors (2026) akosombo-spillage-geoai: code, result tables and fused flood map for the 2023 Akosombo spillage studies (version 1.0.0) [Software and data]. GitHub. https://github.com/dzimike/akosombo-spillage-geoai",
    "Zwarts, L.": "Zwarts L, van Beukering P, Kone B, Wymenga E, Taylor D (2005) The Niger, a lifeline: effective water management in the Upper Niger Basin. RIZA, Lelystad",
    "Zanaga, D.": "Zanaga D, Van De Kerchove R, Daems D et al (2022) ESA WorldCover 10 m 2021 v200. Zenodo. https://doi.org/10.5281/zenodo.7254221",
}
YEAR = r"\((\d{4})\)"


def _initials(given: str) -> str:
    return re.sub(r"[.\s]", "", given.replace("-", "‑")).replace("‑", "-")


def _authors(chunk: str) -> str:
    """'Surname, A. B., Surname, C., and Surname, D.' (possibly with an ellipsis) -> 'Surname AB, Surname C, Surname D' (+ et al)."""
    et_al = "…" in chunk or "et al" in chunk
    chunk = chunk.split("…")[0].replace("et al.", " ").replace(", and ", ", ").replace(" and ", ", ")
    people = re.findall(r"([^,]+?),\s*((?:[A-Z][A-Za-z\-]*\.?[\s\-]*)+?)(?=,|$)", chunk.strip().strip(","))
    out = [f"{s.strip()} {_initials(g)}" for s, g in people]
    return ", ".join(out) + (" et al" if et_al else "")


def convert_reference(entry: str) -> str:
    for key, text in MANUAL.items():
        if entry.startswith(key):
            return text
    m = re.match(rf"^(.*?)\.? {YEAR}\.? (.*)$", entry, re.S)
    if not m:
        return entry
    who, year, rest = m.group(1).strip(), m.group(2), m.group(3).strip()
    if re.search(r",\s*[A-Z]\b", who):                      # personal authors
        who = _authors(who)
    j = re.match(r"^(?P<title>.+?)\. \*(?P<jour>[^*]+)\*,? ?(?P<tail>.*)$", rest, re.S)
    if j:                                                      # journal article
        tail = j.group("tail").rstrip(".")
        doi = ""
        d = re.search(r"\s*(https?://\S+)$", tail)
        if d:
            doi = " " + d.group(1); tail = tail[:d.start()].rstrip(", .")
        vol = re.match(r"^(\d+)(\((\d+(?:[–-]\d+)?)\))?,?\s*(.*)$", tail)
        if vol:
            v, iss, pages = vol.group(1), vol.group(2) or "", vol.group(4)
            loc = f"{v}{iss}" + (f":{pages}" if pages else "")
        else:
            loc = tail
        ref = f"{who} ({year}) {j.group('title')}. *{j.group('jour')}* {loc}".rstrip(" .,")
        return ref + (f".{doi}" if doi else "")
    return f"{who} ({year}) {rest}"                           # web page, dataset or report: keep the remainder


def convert_citations(text: str) -> str:
    """'(Lehner et al., 2011; Citi Newsroom, 2025)' -> '(Lehner et al. 2011; Citi Newsroom 2025)'; Figure -> Fig."""
    text = re.sub(r"(?<=[A-Za-z\.]), (?=(?:19|20)\d\d[a-z]?\b)", " ", text)
    text = re.sub(r"\bFigures (?=\d)", "Figs. ", text)
    text = re.sub(r"\bFigure (?=\d)", "Fig. ", text)
    text = re.sub(r"^\*\*Fig\. (\d+)\.\*\* ", r"**Fig. \1** ", text, flags=re.M)
    text = re.sub(r"^\*\*Table (\d+)\.\*\* ", r"**Table \1** ", text, flags=re.M)
    return text


def strip_caption_periods(text: str) -> str:
    """Springer: no punctuation after the figure number (done above) and none at the end of a figure caption."""
    return re.sub(r"^(\*\*Fig\. \d+\*\* .*?)\.$", r"\1", text, flags=re.M)
