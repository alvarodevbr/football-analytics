"""Gera o artigo em .docx, no padrão ABNT do CLAUDE.md (§4), a partir de ``article/artigo.tex``.

O PDF sai do próprio LaTeX (``latexmk`` em ``article/``); este script lê o mesmo ``artigo.tex`` e o
``referencias.bib`` e reproduz o documento no Word:
* numeração de seções, ilustrações e equações;
* citações e referências no formato do biblatex-abnt;
* equações nativas do Word (OMML) e notas de rodapé.

O subconjunto de LaTeX aceito está descrito em ``latex_reader.py``.

Uso: ``.venv/bin/python article/build_docx.py [--fonte artigo.tex] [--saida artigo.docx]``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.document import Document as DocxDocument
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import XmlPart
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml import etree

from football_analytics.logging_utils import get_logger, setup_logging
from latex_reader import ILLUSTRATIONS, TEXT_WIDTH_CM, Article, Block, Run, Runs, plain, read_article

ARTICLE_DIR = Path(__file__).resolve().parent
ROOT = ARTICLE_DIR.parent
log = get_logger("article")

FONT = "Arial"
BODY_PT, TITLE_PT, SMALL_PT = 12, 14, 10


# --------------------------------------------------------------------------------------------
# Equações: subconjunto de LaTeX → OMML
# --------------------------------------------------------------------------------------------
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
         "kappa": "κ", "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
         "phi": "φ", "varphi": "φ", "omega": "ω", "theta": "θ", "Delta": "Δ", "Sigma": "Σ", "Phi": "Φ"}
SYMBOLS = {"cdot": "·", "times": "×", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠",
           "approx": "≈", "in": "∈", "infty": "∞", "to": "→", "ldots": "…", "pm": "±", "%": "%",
           "{": "{", "}": "}", ",": " ", ";": " ", "quad": " ", "qquad": "  ",
           " ": " ", "!": ""}
FUNCTIONS = {"ln", "exp", "log", "max", "min"}
NARY = {"sum": "∑", "prod": "∏"}
ACCENTS = {"hat": "̂", "widehat": "̂", "tilde": "̃", "widetilde": "̃"}
TOKEN = re.compile(r"\\[A-Za-z]+|\\[^A-Za-z]|[{}_^]|\s+|[^\\{}_^\s]")


def _m(tag: str, *children: etree._Element, **attrs: str) -> etree._Element:
    el = etree.Element(f"{{{M_NS}}}{tag}")
    for key, value in attrs.items():
        el.set(f"{{{M_NS}}}{key}", value)
    el.extend(children)
    return el


def _wrap(tag: str, content: list[etree._Element]) -> etree._Element:
    return _m(tag, *content)


def _math_run(text: str, plain: bool = False) -> etree._Element:
    run = _m("r")
    if plain:
        run.append(_m("rPr", _m("sty", val="p")))
    rpr = etree.SubElement(run, f"{{{W_NS}}}rPr")
    fonts = etree.SubElement(rpr, f"{{{W_NS}}}rFonts")
    fonts.set(f"{{{W_NS}}}ascii", "Cambria Math")
    fonts.set(f"{{{W_NS}}}hAnsi", "Cambria Math")
    t = etree.SubElement(run, f"{{{M_NS}}}t")
    t.set(XML_SPACE, "preserve")
    t.text = text.replace("-", "−")
    return run


@dataclass
class _Nary:
    char: str


class LatexToOmml:
    """Analisador recursivo de um subconjunto de LaTeX: frações, índices, delimitadores, acentos,
    somatórios, letras gregas, funções e texto."""

    def __init__(self, latex: str) -> None:
        self.toks = TOKEN.findall(latex)
        self.i = 0

    def _skip(self) -> None:
        while self.i < len(self.toks) and self.toks[self.i].isspace():
            self.i += 1

    def peek(self) -> str | None:
        self._skip()
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        self._skip()
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def expect(self, tok: str) -> None:
        if (got := self.take()) != tok:
            raise ValueError(f"LaTeX: esperado {tok!r}, encontrado {got!r}")

    def convert(self) -> etree._Element:
        content = self.seq(set())
        return _wrap("oMath", content)

    def seq(self, stop: set[str]) -> list[etree._Element]:
        out: list[etree._Element] = []
        while (tok := self.peek()) is not None and tok not in stop:
            out.extend(self.script())
        return out

    def arg(self) -> list[etree._Element]:
        if self.peek() == "{":
            self.take()
            content = self.seq({"}"})
            self.expect("}")
            return content
        atom = self.atom()
        if isinstance(atom, _Nary):
            raise ValueError("LaTeX: somatório como argumento")
        return atom

    def raw_braced(self) -> str:
        self.expect("{")
        depth, parts = 1, []
        while True:
            tok = self.toks[self.i]
            self.i += 1
            depth += tok == "{"
            depth -= tok == "}"
            if depth == 0:
                return "".join(parts)
            parts.append(tok)

    def delimiter(self) -> str:
        tok = self.take()
        if tok == ".":
            return ""
        return SYMBOLS.get(tok[1:], tok[1:]) if tok.startswith("\\") else tok

    def script(self) -> list[etree._Element]:
        base = self.atom()
        sub = sup = None
        while self.peek() in ("_", "^"):
            if self.take() == "_":
                sub = self.arg()
            else:
                sup = self.arg()
        if isinstance(base, _Nary):
            pr = _m("naryPr", _m("chr", val=base.char), _m("limLoc", val="undOvr"))
            if sub is None:
                pr.append(_m("subHide", val="1"))
            if sup is None:
                pr.append(_m("supHide", val="1"))
            operand = self.script()
            return [_m("nary", pr, _wrap("sub", sub or []), _wrap("sup", sup or []), _wrap("e", operand))]
        if sub is not None and sup is not None:
            return [_m("sSubSup", _wrap("e", base), _wrap("sub", sub), _wrap("sup", sup))]
        if sub is not None:
            return [_m("sSub", _wrap("e", base), _wrap("sub", sub))]
        if sup is not None:
            return [_m("sSup", _wrap("e", base), _wrap("sup", sup))]
        return base

    def atom(self) -> list[etree._Element] | _Nary:
        tok = self.take()
        if tok == "{":
            content = self.seq({"}"})
            self.expect("}")
            return content
        if not tok.startswith("\\"):
            return [_math_run(tok)]
        name = tok[1:]
        if name == "frac":
            num, den = self.arg(), self.arg()
            return [_m("f", _wrap("num", num), _wrap("den", den))]
        if name == "left":
            opening = self.delimiter()
            content = self.seq({"\\right"})
            self.expect("\\right")
            closing = self.delimiter()
            props = _m("dPr", _m("begChr", val=opening), _m("endChr", val=closing))
            return [_m("d", props, _wrap("e", content))]
        if name in ("text", "mathrm", "operatorname"):
            return [_math_run(self.raw_braced(), plain=True)]
        if name in FUNCTIONS:
            return [_math_run(name, plain=True)]
        if name in NARY:
            return _Nary(NARY[name])
        if name in ACCENTS:
            return [_m("acc", _m("accPr", _m("chr", val=ACCENTS[name])), _wrap("e", self.arg()))]
        if name in ("bar", "overline"):
            return [_m("bar", _m("barPr", _m("pos", val="top")), _wrap("e", self.arg()))]
        if name in GREEK:
            return [_math_run(GREEK[name])]
        if name in SYMBOLS:
            return [_math_run(SYMBOLS[name])]
        raise ValueError(f"comando LaTeX não suportado: {tok}")


# --------------------------------------------------------------------------------------------
# Formatação (ABNT, CLAUDE.md §4)
# --------------------------------------------------------------------------------------------
def _set_fonts(rpr: etree._Element, name: str = FONT) -> None:
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        fonts.attrib.pop(qn(f"w:{attr}"), None)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{attr}"), name)


def setup_document() -> DocxDocument:
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin, section.right_margin = Cm(3), Cm(2)
    section.top_margin, section.bottom_margin = Cm(3), Cm(2)
    section.header_distance = Cm(2)

    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = FONT, Pt(BODY_PT)
    _set_fonts(normal.element.get_or_add_rPr())
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before, pf.space_after = Pt(0), Pt(0)
    pf.first_line_indent = Cm(1.0)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for level in (1, 2):
        style = doc.styles[f"Heading {level}"]
        style.font.name, style.font.size = FONT, Pt(TITLE_PT)
        style.font.bold, style.font.italic = True, False
        style.font.color.rgb = RGBColor(0, 0, 0)
        _set_fonts(style.element.get_or_add_rPr())
        hf = style.paragraph_format
        hf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        hf.space_before, hf.space_after = Pt(0), Pt(0)
        hf.first_line_indent = Cm(0)
        hf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        hf.keep_with_next = True

    header_style = doc.styles["Header"]
    header_style.font.name, header_style.font.size = FONT, Pt(SMALL_PT)
    _set_fonts(header_style.element.get_or_add_rPr())
    _page_number(section.header.paragraphs[0])
    return doc


def _page_number(p: Paragraph) -> None:
    """Paginação no canto superior direito (campo PAGE)."""
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run()
    for kind in ("begin", "instr", "end"):
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = " PAGE "
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        run._r.append(el)


def fmt(p: Paragraph, *, align: WD_ALIGN_PARAGRAPH | None = None, indent: float | None = None,
        left: float | None = None, single: bool = False, before: float = 0, after: float = 0,
        keep: bool = False) -> Paragraph:
    pf = p.paragraph_format
    if align is not None:
        pf.alignment = align
    if indent is not None:
        pf.first_line_indent = Cm(indent)
    if left is not None:
        pf.left_indent = Cm(left)
    if single:
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.keep_with_next = keep or None
    return p


def _border(parent: etree._Element, edge: str, on: bool) -> None:
    el = OxmlElement(f"w:{edge}")
    if on:
        for key, value in (("val", "single"), ("sz", "8"), ("space", "0"), ("color", "000000")):
            el.set(qn(f"w:{key}"), value)
    else:
        el.set(qn("w:val"), "nil")
    parent.append(el)


def style_table(table: Table, kind: str, header_rows: int) -> None:
    """Tabela: padrão IBGE (linhas só no topo, sob o cabeçalho e no fim). Quadro: todas as bordas."""
    borders = OxmlElement("w:tblBorders")
    full = kind == "quadro"
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        _border(borders, edge, full or edge in ("top", "bottom"))
    table._tbl.tblPr.append(borders)
    if not full:
        for cell in table.rows[header_rows - 1].cells:
            tc_borders = OxmlElement("w:tcBorders")
            _border(tc_borders, "bottom", True)
            cell._tc.get_or_add_tcPr().append(tc_borders)
    for row in table.rows[:header_rows]:
        repeat = OxmlElement("w:tblHeader")
        repeat.set(qn("w:val"), "true")
        row._tr.get_or_add_trPr().append(repeat)


CHAR_CM, PAD_CM = 0.21, 0.35            # largura de um caractere em Arial 10 negrito e margens da célula


def column_widths(rows: list[list[str]], total: float = TEXT_WIDTH_CM) -> list[float]:
    """Larguras proporcionais ao texto de cada coluna, sem quebrar a palavra mais longa dela."""
    n = len(rows[0])
    weights, minimum = [], []
    for j in range(n):
        texts = [r[j] for r in rows if j < len(r)]
        lengths = [len(t) for t in texts]
        weights.append(max(4.0, sum(lengths) / len(lengths), 0.45 * max(lengths)))
        minimum.append(max((len(w) for t in texts for w in t.split()), default=1) * CHAR_CM + PAD_CM)
    fixed: dict[int, float] = {}
    while True:                                      # quem fica abaixo do mínimo recebe o mínimo
        free = total - sum(fixed.values())
        rest = [j for j in range(n) if j not in fixed]
        widths = {j: free * weights[j] / sum(weights[k] for k in rest) for j in rest}
        short = [j for j in rest if widths[j] < minimum[j]]
        if not short or len(short) == len(rest):
            break
        fixed.update({j: minimum[j] for j in short})
    return [fixed.get(j, widths.get(j, 0.0)) for j in range(n)]


# --------------------------------------------------------------------------------------------
# Notas de rodapé (o python-docx não as cria): parte footnotes.xml e estilos próprios
# --------------------------------------------------------------------------------------------
FOOTNOTES_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"


class Footnotes:
    def __init__(self, doc: DocxDocument) -> None:
        separators = "".join(
            f'<w:footnote w:type="{kind}" w:id="{fid}"><w:p><w:pPr><w:spacing w:after="0" w:line="240" '
            f'w:lineRule="auto"/></w:pPr><w:r><w:{kind}/></w:r></w:p></w:footnote>'
            for fid, kind in ((-1, "separator"), (0, "continuationSeparator")))
        self.root = parse_xml(f'<w:footnotes xmlns:w="{W_NS}" xmlns:m="{M_NS}">{separators}</w:footnotes>')
        part = XmlPart(PackURI("/word/footnotes.xml"), FOOTNOTES_CT, self.root, doc.part.package)
        doc.part.relate_to(part, RT.FOOTNOTES)
        settings = doc.settings.element
        props = parse_xml(f'<w:footnotePr xmlns:w="{W_NS}"><w:footnote w:id="-1"/><w:footnote w:id="0"/>'
                          '</w:footnotePr>')
        compat = settings.find(qn("w:compat"))
        if compat is not None:
            compat.addprevious(props)

        text = doc.styles.add_style("Footnote Text", WD_STYLE_TYPE.PARAGRAPH)
        text.base_style = doc.styles["Normal"]
        text.font.size = Pt(SMALL_PT)
        tf = text.paragraph_format
        tf.line_spacing_rule = WD_LINE_SPACING.SINGLE
        tf.first_line_indent, tf.alignment = Cm(0), WD_ALIGN_PARAGRAPH.LEFT
        reference = doc.styles.add_style("Footnote Reference", WD_STYLE_TYPE.CHARACTER)
        reference.font.superscript = True
        self.next_id = 1

    @staticmethod
    def _reference_run(tag: str, fid: int | None = None) -> etree._Element:
        run = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        style = OxmlElement("w:rStyle")
        style.set(qn("w:val"), "FootnoteReference")
        rpr.append(style)
        run.append(rpr)
        mark = OxmlElement(tag)
        if fid is not None:
            mark.set(qn("w:id"), str(fid))
        run.append(mark)
        return run

    def add(self, paragraph: Paragraph, content: Runs, renderer: Renderer) -> None:
        fid, self.next_id = self.next_id, self.next_id + 1
        paragraph._p.append(self._reference_run("w:footnoteReference", fid))
        note = OxmlElement("w:footnote")
        note.set(qn("w:id"), str(fid))
        p = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        pstyle = OxmlElement("w:pStyle")
        pstyle.set(qn("w:val"), "FootnoteText")
        ppr.append(pstyle)
        p.append(ppr)
        p.append(self._reference_run("w:footnoteRef"))
        note.append(p)
        self.root.append(note)
        renderer.add_runs(Paragraph(p, None), [Run(" ")] + list(content), size=SMALL_PT)


# --------------------------------------------------------------------------------------------
# Montagem
# --------------------------------------------------------------------------------------------
ALIGN = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT}


class Renderer:
    def __init__(self, doc: DocxDocument, article: Article) -> None:
        self.doc, self.article = doc, article
        self.footnotes = Footnotes(doc)
        self.last_blank = True
        self.words = 0

    def add_runs(self, p: Paragraph, runs: Runs, size: float | None = None, bold: bool = False,
                 italic: bool = False) -> None:
        """Acrescenta ao parágrafo os trechos, as fórmulas em linha e as notas de rodapé."""
        for r in runs:
            if r.note is not None:
                self.footnotes.add(p, list(r.note), self)
                continue
            if r.math:
                p._p.append(LatexToOmml(r.math).convert())
                continue
            for k, line in enumerate(r.text.split("\n")):
                run = p.add_run(line)
                if k:
                    run.add_break()
                run.bold, run.italic = (bold or r.bold) or None, (italic or r.italic) or None
                if size:
                    run.font.size = Pt(size)
                run.font.subscript, run.font.superscript = r.sub or None, r.sup or None

    def para(self, runs: Runs | None = None, **kw: object) -> Paragraph:
        size = kw.pop("size", None)
        bold, italic = bool(kw.pop("bold", False)), bool(kw.pop("italic", False))
        p = self.doc.add_paragraph()
        if runs:
            self.add_runs(p, runs, size, bold, italic)
        fmt(p, **kw)
        self.last_blank = not runs
        return p

    def blank(self, keep: bool = False) -> None:
        if not self.last_blank:
            self.para(None, indent=0, keep=keep)

    def latex(self, text: str, **style: bool) -> Runs:
        return self.article.runs(text, **style)

    def front(self) -> None:
        a = self.article
        self.para(self.latex(a.title), size=TITLE_PT, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0)
        self.blank()
        lines = [(name, note) for name, note in a.authors]
        if a.orientation:
            label, name, note = a.orientation
            lines.append((f"{label}: {name}", note))
        for text, note in lines:                       # a nota de rodapé de cada autor é opcional
            runs = self.latex(text) + ([Run(note=tuple(self.latex(note)))] if note else [])
            self.para(runs, align=WD_ALIGN_PARAGRAPH.RIGHT, indent=0)
        self.blank()
        self.para([Run("RESUMO")], size=TITLE_PT, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0)
        self.blank()
        self.para(self.latex(a.abstract), indent=0)
        self.blank()
        p = self.para([Run("Palavras-chave:", bold=True), Run(" ")] + self.latex(a.keywords), indent=0)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        self.words += len(plain(self.latex(a.abstract)).split())

    def heading(self, b: Block) -> None:
        self.blank()
        runs = self.latex(b.text)
        if b.level == 1:
            runs = [Run(r.text.upper(), r.bold, r.italic, r.sub, r.sup, r.math, r.note) for r in runs]
        if b.numbered:
            runs = [Run(f"{b.number} ")] + runs
        p = self.doc.add_paragraph(style=f"Heading {b.level}")
        self.add_runs(p, runs, TITLE_PT, bold=True)
        if not b.numbered:
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.last_blank = False
        self.blank(keep=True)                          # o título não fica sozinho no pé da página

    def caption(self, b: Block) -> None:
        self.blank()
        self.para([Run(f"{ILLUSTRATIONS[b.kind]} {b.number} – ")] + self.latex(b.text), size=SMALL_PT, indent=0,
                  single=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=3, keep=True)

    def notes(self, b: Block) -> None:
        if b.source:
            self.para([Run("Fonte: ")] + self.latex(b.source), size=SMALL_PT, indent=0, single=True, before=3,
                      align=WD_ALIGN_PARAGRAPH.LEFT, keep=bool(b.note))
        if b.note:
            self.para([Run("Nota: ")] + self.latex(b.note), size=SMALL_PT, indent=0, single=True,
                      align=WD_ALIGN_PARAGRAPH.LEFT)
        self.blank()

    def figure(self, b: Block) -> None:
        self.caption(b)
        p = self.para(None, indent=0, single=True, align=WD_ALIGN_PARAGRAPH.CENTER, keep=True)
        p.add_run().add_picture(str(b.path), width=Cm(b.width_cm))
        self.last_blank = False
        self.notes(b)

    def table(self, b: Block) -> None:
        self.caption(b)
        cells = [[self.latex(c) for c in row] for row in b.rows]
        ncols = len(b.aligns)
        table = self.doc.add_table(rows=len(cells), cols=ncols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        widths = column_widths([[plain(c) for c in row] for row in cells])
        for j, width in enumerate(widths):
            table.columns[j].width = Cm(width)
        for i, row in enumerate(cells):
            for j, content in enumerate(row):
                cell = table.cell(i, j)
                cell.width = Cm(widths[j])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                p = cell.paragraphs[0]
                self.add_runs(p, content, SMALL_PT)
                # todas as linhas presas à seguinte: a tabela não se separa da fonte
                fmt(p, indent=0, single=True, align=ALIGN[b.aligns[j]], before=1, after=1, keep=True)
            no_split = OxmlElement("w:cantSplit")
            table.rows[i]._tr.get_or_add_trPr().append(no_split)
        style_table(table, b.kind, b.header_rows)
        self.last_blank = False
        self.notes(b)

    def equation(self, b: Block) -> None:
        p = self.para(None, indent=0, before=6, after=6)
        stops = p.paragraph_format.tab_stops
        stops.add_tab_stop(Cm(TEXT_WIDTH_CM / 2), WD_TAB_ALIGNMENT.CENTER)
        stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT)
        p.add_run("\t")
        p._p.append(LatexToOmml(b.text).convert())
        if b.numbered:
            p.add_run(f"\t({b.number})")
        self.last_blank = False

    def body(self) -> None:
        for b in self.article.blocks:
            if b.kind == "heading":
                self.heading(b)
            elif b.kind == "para":
                runs = self.latex(b.text)
                self.para(runs)
                self.words += len(plain(runs).split())
            elif b.kind == "list":
                for label, item in b.items:
                    runs = self.latex(item)
                    self.para([Run(f"{label} ")] + runs, left=1.6, indent=-0.6)
                    self.words += len(plain(runs).split())
            elif b.kind == "quote":                     # 6 pt antes e depois, sem linha em branco
                self.para(self.latex(b.text), size=SMALL_PT, indent=0, left=4.0, single=True, before=6, after=6)
            elif b.kind == "figura":
                self.figure(b)
            elif b.kind in ("tabela", "quadro"):
                self.table(b)
            elif b.kind == "equation":
                self.equation(b)
            elif b.kind == "pagebreak":
                self.doc.add_page_break()
                self.last_blank = True
            elif b.kind == "references":
                self.references()

    def references(self) -> None:
        """Referências: alinhadas à esquerda, espaço simples, 6 pt entre elas, em ordem alfabética."""
        self.heading(Block("heading", "Referências", level=1, numbered=False))
        entries = self.article.references()
        for entry in entries:
            self.para(entry, indent=0, single=True, after=6, align=WD_ALIGN_PARAGRAPH.LEFT)
        log.info("referências", extra={"n": len(entries)})


def _shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def build(source: Path, output: Path) -> None:
    article = read_article(source)
    doc = setup_document()
    renderer = Renderer(doc, article)
    renderer.front()
    renderer.body()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.core_properties.title = plain(article.runs(article.title))
    doc.core_properties.author = "; ".join(plain(article.runs(name)) for name, _ in article.authors)
    doc.save(output)
    counts = {kind: sum(b.kind == kind for b in article.blocks) for kind in ("figura", "tabela", "quadro", "equation")}
    log.info("artigo gerado", extra={"arquivo": _shown(output), "palavras_texto": renderer.words,
                                     "resumo": len(plain(article.runs(article.abstract)).split()), **counts})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fonte", type=Path, default=ARTICLE_DIR / "artigo.tex")
    parser.add_argument("--saida", type=Path, default=ARTICLE_DIR / "artigo.docx")
    args = parser.parse_args()
    setup_logging(ROOT / "logs" / "article.log")
    build(args.fonte, args.saida)


if __name__ == "__main__":
    main()
