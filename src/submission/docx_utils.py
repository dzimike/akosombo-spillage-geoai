from __future__ import annotations
"""Shared Word-document helpers for the journal submission packages (styling, line numbers, tables, plain documents)."""
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


def set_cell_border(table):
    tbl = table._tbl
    pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz in (("top", 12), ("bottom", 12), ("insideH", 4)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(sz)); el.set(qn("w:color"), "000000")
        borders.append(el)
    # schema order inside w:tblPr: ..., tblW, jc, tblInd, tblBorders, shd, tblLayout, tblCellMar, tblLook
    anchor = next((pr.find(qn(tag)) for tag in ("w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook") if pr.find(qn(tag)) is not None), None)
    if anchor is not None:
        anchor.addprevious(borders)
    else:
        pr.append(borders)


def add_page_number(section):
    p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._p.get_or_add_pPr().append(OxmlElement("w:suppressLineNumbers"))  # some renderers number footer lines too
    run = p.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
        run._r.append(el)


def style_doc(path: Path, line_numbers: bool = True, spacing: float = 1.5, body_pt: float = 11) -> None:
    d = Document(path)
    for s in d.styles:
        if s.type == 1 and s.font is not None:
            s.font.name = "Times New Roman"; s.font.color.rgb = None
            rpr = s.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts"); rpr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rf.set(qn(a), "Times New Roman")
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                if rf.get(qn(a)) is not None:
                    del rf.attrib[qn(a)]
    for name, size in (("Normal", body_pt), ("Body Text", body_pt), ("First Paragraph", body_pt), ("Compact", body_pt - 1), ("Title", 16), ("Heading 1", 14),
                       ("Heading 2", 13), ("Heading 3", 11.5), ("Image Caption", 10), ("Table Caption", 10), ("Captioned Figure", 11)):
        if name in [s.name for s in d.styles]:
            d.styles[name].font.size = Pt(size)
            if name.startswith("Heading") or name == "Title":
                d.styles[name].font.bold = True
    for name in ("Body Text", "First Paragraph", "Normal"):
        if name in [s.name for s in d.styles]:
            pf = d.styles[name].paragraph_format; pf.line_spacing = spacing; pf.space_after = Pt(6)
    for t in d.tables:
        set_cell_border(t)
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(0)
                    for r in p.runs:
                        r.font.size = Pt(9)
    for sec in d.sections:
        sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.5)
        add_page_number(sec)
        if line_numbers:
            # distance 1 cm keeps three-digit numbers clear of the text; lnNumType must precede pgNumType/cols/docGrid in sectPr
            ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); ln.set(qn("w:distance"), "567")
            after = [c for c in sec._sectPr if c.tag in {qn(t) for t in ("w:pgNumType", "w:cols", "w:formProt", "w:vAlign", "w:noEndnote", "w:titlePg",
                                                                         "w:textDirection", "w:bidi", "w:rtlGutter", "w:docGrid", "w:printerSettings")}]
            if after:
                after[0].addprevious(ln)
            else:
                sec._sectPr.append(ln)
    paras = d.paragraphs
    for i, p in enumerate(paras):
        if p.style.name in ("Image Caption", "Captioned Figure"):
            p.paragraph_format.keep_with_next = False
        drawings = p._p.findall(".//" + qn("wp:docPr"))
        if drawings and i + 1 < len(paras):
            # keep each figure on the page of its caption, keep the caption in one piece, and give the image alt text from the caption
            cap = paras[i + 1]
            p.paragraph_format.keep_with_next = True
            cap.paragraph_format.keep_together = True
            alt = cap.text.strip()
            for dp in drawings:
                if alt and not dp.get("descr"):
                    dp.set("descr", alt[:1000])
    d.save(path)


def pandoc(md: Path, out: Path, root: Path) -> None:
    subprocess.run(["pandoc", str(md), "-o", str(out), "--resource-path", str(root / "docs")], check=True)


def plain_doc(path: Path, title: str | None, paras: list, spacing: float = 1.15) -> None:
    """Write a simple Word file: an optional heading, then paragraphs (a (bold, text) tuple gives a bold lead-in)."""
    d = Document(); sec = d.sections[0]
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.5)
    st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11.5); st.paragraph_format.line_spacing = spacing
    if title:
        h = d.add_heading(title, level=1)
        for r in h.runs:
            r.font.name = "Times New Roman"; r.font.size = Pt(15)
    for p in paras:
        if isinstance(p, tuple):
            para = d.add_paragraph(); r = para.add_run(p[0]); r.bold = True; para.add_run(p[1])
        else:
            d.add_paragraph(p)
    d.save(path)
