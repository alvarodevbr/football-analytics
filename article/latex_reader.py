r"""Leitura de ``artigo.tex`` e ``referencias.bib`` para o gerador de .docx (``build_docx.py``).

O leitor entende o subconjunto de LaTeX usado no artigo (classe abntex2 com ``abntartigo.sty``) e
devolve o documento em blocos. O texto de cada bloco continua em LaTeX até a montagem, quando
``Article.runs`` o converte em trechos formatados (``Run``). Na conversão, as remissões recebem o
número e as citações e referências seguem o formato que o biblatex-abnt imprime no PDF
(NBR 10520:2023 e NBR 6023:2018).

Subconjunto aceito:

* no preâmbulo: ``\titulo``, ``\adicionarautor``, ``\orientacao``, ``\palavraschave``,
  ``\graphicspath`` e ``\addbibresource``; o restante do preâmbulo é ignorado;
* na estrutura:
  - ``\section`` e ``\subsection``, sem número na forma com ``*``;
  - ``\label``, ``\imprimircabecalho``, ``\printbibliography``, ``\newpage`` e ``\clearpage``;
* nos ambientes:
  - ``resumo``, ``equation``/``equation*`` e ``citacao``;
  - ``alineas``, ``itemize`` e ``enumerate``, com ``label=``;
  - ``figure``, ``table`` e ``quadro``, que aceitam ``\caption``, ``\label``,
    ``\includegraphics``, ``\fonte`` e ``\nota``, além de ``tabular``/``tabularx`` com
    ``booktabs`` ou ``\hline``;
* no texto:
  - ``\textbf``, ``\textit``, ``\emph``, ``\textsuperscript``, ``\textsubscript``, ``\enquote``,
    ``\footnote``, ``\url`` e ``\href``;
  - ``\cite``/``\parencite``, ``\textcite``, ``\citeauthor``, ``\citeyear`` e ``\nocite``;
  - ``\ref``, ``\eqref``, fórmulas ``$...$`` e os caracteres especiais usuais (``~``, ``--``,
    ``\%``, ``\,`` etc.).

Qualquer outro comando interrompe a leitura com o nome dele (``LatexError``), para que nada seja
omitido do .docx em silêncio.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

TEXT_WIDTH_CM = 16.0                     # A4 (21 cm) menos as margens de 3 cm e 2 cm
FLOATS = {"figure": "figura", "table": "tabela", "quadro": "quadro"}
ILLUSTRATIONS = {"figura": "Figura", "tabela": "Tabela", "quadro": "Quadro"}
MAX_CITE_NAMES, MAX_BIB_NAMES = 2, 3     # et al. a partir de 3 autores na citação e de 4 na lista
COMMAND = re.compile(r"\\([A-Za-z]+)(\*?)")


class LatexError(ValueError):
    """Construção fora do subconjunto de LaTeX que o gerador de .docx entende."""


@dataclass(frozen=True)
class Run:
    """Trecho de texto com formatação uniforme, uma fórmula em OMML ou uma nota de rodapé."""

    text: str = ""
    bold: bool = False
    italic: bool = False
    sub: bool = False
    sup: bool = False
    math: str = ""                            # fórmula em linha que só o OMML representa
    note: tuple[Run, ...] | None = None       # conteúdo da nota de rodapé


Runs = list[Run]


def plain(runs: Runs) -> str:
    return "".join(r.text for r in runs if r.note is None)


@dataclass
class Block:
    kind: str     # heading | para | list | quote | figura | tabela | quadro | equation | pagebreak | references
    text: str = ""                                   # LaTeX do título, parágrafo, citação ou legenda
    level: int = 0
    numbered: bool = True
    items: list[tuple[str, str]] = field(default_factory=list)   # (rótulo, LaTeX do item)
    rows: list[list[str]] = field(default_factory=list)
    header_rows: int = 1
    aligns: list[str] = field(default_factory=list)
    path: Path | None = None
    width_cm: float = TEXT_WIDTH_CM
    label: str = ""
    number: str = ""
    source: str = ""
    note: str = ""


# --------------------------------------------------------------------------------------------
# Leitura de grupos, comandos e ambientes
# --------------------------------------------------------------------------------------------
def strip_comments(text: str) -> str:
    """Remove os comentários; como no TeX, ``%`` no fim da linha também elimina a quebra."""
    lines = []
    for line in text.splitlines():
        m = re.search(r"(?<!\\)%", line)
        if m is None:
            lines.append(line)
        elif line[:m.start()].strip():
            lines.append(line[:m.start()] + "\x00")
    return "\n".join(lines).replace("\x00\n", "").replace("\x00", "")


def skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def group(s: str, i: int) -> tuple[str, int]:
    """Conteúdo do grupo ``{...}`` que começa em ``s[i]`` (após espaços) e a posição seguinte."""
    i = skip_ws(s, i)
    if i >= len(s) or s[i] != "{":
        raise LatexError(f"esperado '{{' em: {s[i:i + 40]!r}")
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        depth += c == "{"
        depth -= c == "}"
        if depth == 0:
            return s[i + 1:j], j + 1
        j += 1
    raise LatexError(f"grupo sem fechamento: {s[i:i + 40]!r}")


def optional(s: str, i: int) -> tuple[str | None, int]:
    """Argumento opcional ``[...]``, se houver."""
    j = skip_ws(s, i)
    if j >= len(s) or s[j] != "[":
        return None, i
    depth, k = 0, j
    while k < len(s):
        c = s[k]
        if c == "\\":
            k += 2
            continue
        if c == "{":
            _, k = group(s, k)
            continue
        depth += c == "["
        depth -= c == "]"
        if depth == 0:
            return s[j + 1:k], k + 1
        k += 1
    raise LatexError(f"argumento opcional sem fechamento: {s[j:j + 40]!r}")


def environment(s: str, i: int, name: str) -> tuple[str, int]:
    """Conteúdo do ambiente ``name`` aberto antes de ``s[i]`` e a posição após o ``\\end``."""
    pattern = re.compile(r"\\(begin|end)\{" + re.escape(name) + r"\}")
    depth = 1
    for m in pattern.finditer(s, i):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return s[i:m.start()], m.end()
    raise LatexError(f"ambiente sem \\end: {name}")


def split_top(s: str, sep: re.Pattern[str]) -> list[str]:
    """Divide ``s`` nas ocorrências de ``sep`` fora de grupos."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(s):
        c = s[i]
        if depth == 0 and (m := sep.match(s, i)):
            parts.append(s[start:i])
            start = i = m.end()
            continue
        if c == "\\":
            i += 2
            continue
        depth += c == "{"
        depth -= c == "}"
        i += 1
    parts.append(s[start:])
    return parts


def commands(s: str) -> list[tuple[str, str | None, list[str]]]:
    """Sequência de comandos de um corpo de flutuante: (nome, opcional, argumentos)."""
    nargs = {"caption": 1, "label": 1, "includegraphics": 1, "fonte": 1, "nota": 1, "centering": 0,
             "small": 0, "footnotesize": 0}
    out, i = [], 0
    while (i := skip_ws(s, i)) < len(s):
        m = COMMAND.match(s, i)
        if m is None:
            raise LatexError(f"texto solto em ilustração: {s[i:i + 40]!r}")
        name, i = m.group(1), m.end()
        if name == "begin":
            env, i = group(s, i)
            body, i = environment(s, i, env)
            out.append(("begin", None, [env, body]))
            continue
        if name not in nargs:
            raise LatexError(f"comando não suportado em ilustração: \\{name}")
        opt, i = optional(s, i)
        args = []
        for _ in range(nargs[name]):
            arg, i = group(s, i)
            args.append(arg)
        out.append((name, opt, args))
    return out


# --------------------------------------------------------------------------------------------
# Tabelas
# --------------------------------------------------------------------------------------------
ALIGN = {"l": "left", "L": "left", "X": "left", "c": "center", "C": "center", "r": "right", "R": "right"}
ROW_END = re.compile(r"\\\\(?:\s*\[[^\]]*\])?")
CELL_SEP = re.compile(r"(?<!\\)&")
RULE = re.compile(r"\s*\\(toprule|midrule|bottomrule|hline|cmidrule(?:\([^)]*\))?\{[^}]*\}|addlinespace)")


def column_aligns(spec: str) -> list[str]:
    out, i = [], 0
    while i < len(spec):
        c = spec[i]
        if c.isspace() or c == "|":
            i += 1
        elif c in "<>@!":
            _, i = group(spec, i + 1)
        elif c == "*":
            n, i = group(spec, i + 1)
            sub, i = group(spec, i)
            out += column_aligns(sub) * int(n)
        elif c in "pmb":
            _, i = group(spec, i + 1)
            out.append("left")
        elif c in ALIGN:
            out.append(ALIGN[c])
            i += 1
        else:
            raise LatexError(f"tipo de coluna não suportado: {c}")
    return out


def parse_tabular(env: str, body: str, block: Block) -> None:
    i = 0
    if env in ("tabularx", "tabular*"):
        _, i = group(body, i)
    spec, i = group(body, i)
    block.aligns = column_aligns(spec)
    header_rows = None
    for chunk in split_top(body[i:], ROW_END):
        rules = []
        while m := RULE.match(chunk):
            rules.append(m.group(1))
            chunk = chunk[m.end():]
        if header_rows is None and block.rows and "midrule" in rules:
            header_rows = len(block.rows)
        if chunk.strip():
            if "\\multicolumn" in chunk or "\\multirow" in chunk:
                raise LatexError("\\multicolumn e \\multirow não são suportados no .docx")
            cells = [c.strip() for c in split_top(chunk, CELL_SEP)]
            if len(cells) != len(block.aligns):
                raise LatexError(f"linha com {len(cells)} células em tabela de {len(block.aligns)} colunas")
            block.rows.append(cells)
    block.header_rows = header_rows or 1


# --------------------------------------------------------------------------------------------
# Documento
# --------------------------------------------------------------------------------------------
BLOCK_ENVS = {"resumo", "figure", "table", "quadro", "equation", "equation*", "alineas", "itemize",
              "enumerate", "citacao"}
BREAKS = {"newpage", "clearpage", "pagebreak"}
LIST_LABELS = {"alineas": r"\alph*)", "itemize": "•", "enumerate": r"\arabic*."}


def item_label(template: str, k: int) -> str:
    roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]
    label = (template.replace(r"\alph*", chr(ord("a") + k)).replace(r"\Alph*", chr(ord("A") + k))
             .replace(r"\arabic*", str(k + 1)).replace(r"\roman*", roman[k]))
    return label.replace("{", "").replace("}", "")


@dataclass
class Article:
    """Artigo lido do LaTeX; ``runs`` converte o texto de um bloco no momento da montagem."""

    title: str
    authors: list[tuple[str, str]]                  # (nome, nota de rodapé ou "")
    orientation: tuple[str, str, str] | None        # (rótulo, nome, nota de rodapé ou "")
    keywords: str
    abstract: str
    blocks: list[Block]
    bibliography: Bibliography
    base: Path
    refs: dict[str, str] = field(default_factory=dict)

    def runs(self, latex: str, **style: bool) -> Runs:
        return Inline(self).convert(latex, Style(**style))

    def references(self) -> list[Runs]:
        """Referências citadas, na ordem alfabética do biber."""
        bib = self.bibliography
        entries = sorted((bib.entries[k] for k in bib.cited), key=bib.sort_key)
        return [bib.reference(e, self) for e in entries]


def read_article(path: Path) -> Article:
    source = strip_comments(path.read_text(encoding="utf-8"))
    if "\\begin{document}" not in source:
        raise LatexError("\\begin{document} não encontrado")
    preamble, body = source.split("\\begin{document}", 1)
    body = body.split("\\end{document}", 1)[0]
    meta = Preamble(preamble, path.parent)
    reader = BodyReader(meta, path.parent)
    blocks = reader.read(body)
    bib_files = meta.bib_files or [path.with_name("referencias.bib")]
    bibliography = Bibliography.load([path.parent / f for f in bib_files])
    article = Article(meta.title, meta.authors, meta.orientation, meta.keywords, reader.abstract, blocks,
                      bibliography, path.parent)
    article.refs = number(blocks)
    return article


class Preamble:
    def __init__(self, text: str, base: Path) -> None:
        self.title, self.keywords = "", ""
        self.authors: list[tuple[str, str]] = []
        self.orientation: tuple[str, str, str] | None = None
        self.graphics: list[Path] = [base]
        self.bib_files: list[str] = []
        i = 0
        while m := COMMAND.search(text, i):
            name, i = m.group(1), m.end()
            if name == "titulo":
                self.title, i = group(text, i)
            elif name == "adicionarautor":
                note, i = optional(text, i)
                author, i = group(text, i)
                self.authors.append((author, note or ""))
            elif name == "orientacao":
                note, i = optional(text, i)
                label, i = group(text, i)
                person, i = group(text, i)
                self.orientation = (label, person, note or "")
            elif name == "palavraschave":
                self.keywords, i = group(text, i)
            elif name == "graphicspath":
                paths, i = group(text, i)
                self.graphics += [base / p for p in re.findall(r"\{([^{}]*)\}", paths)]
            elif name == "addbibresource":
                bib, i = group(text, i)
                self.bib_files.append(bib)
        if not self.title:
            raise LatexError("\\titulo não encontrado no preâmbulo")


class BodyReader:
    def __init__(self, meta: Preamble, base: Path) -> None:
        self.meta, self.base = meta, base
        self.abstract = ""

    def read(self, body: str) -> list[Block]:
        blocks: list[Block] = []
        buf: list[str] = []

        def flush() -> None:
            text = "".join(buf).strip()
            if text:
                blocks.append(Block("para", text))
            buf.clear()

        i = 0
        while i < len(body):
            c = body[i]
            if c == "\n":
                j = i + 1
                while j < len(body) and body[j] in " \t":
                    j += 1
                if j < len(body) and body[j] == "\n":           # linha em branco: fim do parágrafo
                    flush()
                    i = j
                else:
                    buf.append(c)
                    i += 1
                continue
            if c == "{":
                _, j = group(body, i)
                buf.append(body[i:j])
                i = j
                continue
            m = COMMAND.match(body, i) if c == "\\" else None
            if m is None:
                buf.append(body[i:i + 2] if c == "\\" else c)
                i += 2 if c == "\\" else 1
                continue
            name, star = m.group(1), m.group(2)
            if name == "begin":
                env, j = group(body, m.end())
                if env not in BLOCK_ENVS:
                    raise LatexError(f"ambiente não suportado: {env}")
                flush()
                opt, j = optional(body, j)
                content, i = environment(body, j, env)
                blocks.extend(self.environment(env, opt, content))
            elif name in ("section", "subsection"):
                flush()
                title, i = group(body, m.end())
                blocks.append(Block("heading", title, level=1 if name == "section" else 2, numbered=not star))
            elif name == "label" and not "".join(buf).strip() and blocks and blocks[-1].kind == "heading":
                blocks[-1].label, i = group(body, m.end())
            elif name == "printbibliography":
                flush()
                _, i = optional(body, m.end())
                blocks.append(Block("references"))
            elif name in BREAKS:
                flush()
                blocks.append(Block("pagebreak"))
                i = m.end()
            elif name in ("imprimircabecalho", "maketitle", "noindent"):
                i = m.end()
            else:
                buf.append(m.group(0))
                i = m.end()
        flush()
        return blocks

    def environment(self, env: str, opt: str | None, content: str) -> list[Block]:
        if env == "resumo":
            self.abstract = " ".join(content.split())
            return []
        if env in FLOATS:
            return [self.illustration(FLOATS[env], content)]
        if env.startswith("equation"):
            label = ""
            if m := re.search(r"\\label\{([^}]*)\}", content):
                label = m.group(1)
                content = content[:m.start()] + content[m.end():]
            return [Block("equation", " ".join(content.split()), label=label, numbered=env == "equation")]
        if env == "citacao":
            return [Block("quote", content.strip())]
        template = LIST_LABELS[env]
        if opt and (m := re.search(r"label\s*=\s*(\{[^}]*\}|[^,]+)", opt)):
            template = m.group(1).strip()
        items = []
        for k, raw in enumerate(split_top(content, re.compile(r"\\item\b"))[1:]):
            label, j = optional(raw, 0)
            items.append((label if label is not None else item_label(template, k), raw[j:].strip()))
        return [Block("list", items=items)]

    def illustration(self, kind: str, content: str) -> Block:
        block = Block(kind)
        for name, opt, args in commands(content):
            if name == "caption":
                block.text = args[0]
            elif name == "label":
                block.label = args[0]
            elif name == "fonte":
                block.source = args[0]
            elif name == "nota":
                block.note = args[0]
            elif name == "includegraphics":
                block.path = self.image(args[0])
                if opt and (m := re.search(r"width\s*=\s*([\d.]*)\s*(\\textwidth|\\linewidth|cm|mm)", opt)):
                    value = float(m.group(1) or 1)
                    unit = m.group(2)
                    block.width_cm = value * TEXT_WIDTH_CM if unit.startswith("\\") else (
                        value if unit == "cm" else value / 10)
            elif name == "begin":
                env, body = args
                if env not in ("tabular", "tabularx", "tabular*"):
                    raise LatexError(f"ambiente não suportado em ilustração: {env}")
                parse_tabular(env, body, block)
        if not block.text:
            raise LatexError(f"{kind} sem \\caption")
        if kind == "figura" and block.path is None:
            raise LatexError(f"figura sem \\includegraphics: {block.text[:40]}")
        if kind != "figura" and not block.rows:
            raise LatexError(f"{kind} sem tabular: {block.text[:40]}")
        return block

    def image(self, name: str) -> Path:
        candidates = [name] if Path(name).suffix else [name + ext for ext in (".png", ".jpg", ".jpeg")]
        for directory in self.meta.graphics:
            for candidate in candidates:
                if (path := (directory / candidate).resolve()).exists():
                    if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                        raise LatexError(f"o .docx aceita só imagens PNG ou JPEG: {path.name}")
                    return path
        raise LatexError(f"imagem não encontrada no \\graphicspath: {name}")


def number(blocks: list[Block]) -> dict[str, str]:
    """Numera seções, ilustrações e equações e devolve o texto de cada ``\\ref``."""
    refs: dict[str, str] = {}
    counters = {"figura": 0, "tabela": 0, "quadro": 0, "equation": 0}
    h1 = h2 = 0
    for b in blocks:
        if b.kind == "heading" and b.numbered:
            if b.level == 1:
                h1, h2 = h1 + 1, 0
                b.number = str(h1)
            else:
                h2 += 1
                b.number = f"{h1}.{h2}"
        elif b.kind in counters and b.numbered:
            counters[b.kind] += 1
            b.number = str(counters[b.kind])
        if b.label and b.number:
            if b.label in refs:
                raise LatexError(f"rótulo repetido: {b.label}")
            refs[b.label] = b.number
    return refs


# --------------------------------------------------------------------------------------------
# Texto: LaTeX em linha → trechos formatados
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Style:
    bold: bool = False
    italic: bool = False
    sub: bool = False
    sup: bool = False


SYMBOLS_TEXT = {"ldots": "…", "dots": "…", "textendash": "–", "textemdash": "—", "texteuro": "€",
                "euro": "€", "S": "§", "quad": " ", "qquad": "  ", "LaTeX": "LaTeX", "TeX": "TeX",
                "textquotedblleft": "“", "textquotedblright": "”", "textbullet": "•", "textdegree": "°",
                "textordmasculine": "º", "textordfeminine": "ª", "slash": "/", "textbar": "|"}
ESCAPES = {"%": "%", "&": "&", "#": "#", "$": "$", "_": "_", "{": "{", "}": "}", ",": "\u202f",
           " ": " ", "@": "", "-": "", "/": "", "\\": "\n"}
IGNORED = {"noindent", "centering", "par", "xspace", "allowbreak", "nolinebreak", "relax", "smallskip",
           "medskip", "bigskip", "hfill", "unskip"}
IGNORED_WITH_ARG = {"label", "index", "hspace", "vspace", "phantomsection"}
LIGATURES = (("---", "—"), ("--", "–"), ("``", "“"), ("''", "”"), ("~", "\u00a0"))


class Inline:
    """Conversor do texto LaTeX de um bloco (inclusive notas, citações e fórmulas em linha)."""

    def __init__(self, article: Article) -> None:
        self.article = article
        self.out: Runs = []

    def convert(self, latex: str, style: Style) -> Runs:
        self.out = []
        self.text(latex, style)
        return tidy(self.out)

    def emit(self, text: str, style: Style) -> None:
        if text:
            self.out.append(Run(text, style.bold, style.italic, style.sub, style.sup))

    def text(self, s: str, style: Style) -> None:
        i, chars = 0, []

        def flush() -> None:
            self.emit("".join(chars), style)
            chars.clear()

        while i < len(s):
            c = s[i]
            if c.isspace():
                if not chars or chars[-1] != " ":
                    chars.append(" ")
                i += 1
            elif c == "{":
                inner, i = group(s, i)
                flush()
                self.text(inner, style)
            elif c == "$":
                end = s.find("$", i + 1)
                while end > 0 and s[end - 1] == "\\":
                    end = s.find("$", end + 1)
                if end < 0:
                    raise LatexError(f"fórmula sem '$' de fechamento: {s[i:i + 40]!r}")
                flush()
                self.math(s[i + 1:end], style)
                i = end + 1
            elif c == "\\":
                flush()
                i = self.command(s, i, style)
            else:
                for seq, char in LIGATURES:
                    if s.startswith(seq, i):
                        chars.append(char)
                        i += len(seq)
                        break
                else:
                    chars.append(c)
                    i += 1
        flush()

    def command(self, s: str, i: int, style: Style) -> int:
        m = COMMAND.match(s, i)
        if m is None:                                       # símbolo de controle: \%, \&, \, ...
            char = s[i + 1:i + 2]
            if char == "(":
                end = s.find("\\)", i)
                self.math(s[i + 2:end], style)
                return end + 2
            if char not in ESCAPES:
                raise LatexError(f"comando não suportado no .docx: \\{char}")
            self.emit(ESCAPES[char], style)
            return i + 2
        name, j = m.group(1), m.end()
        if name in SYMBOLS_TEXT or name in IGNORED:
            self.emit(SYMBOLS_TEXT.get(name, ""), style)
            if j < len(s) and s[j] == "{" and s[j + 1:j + 2] == "}":   # \ldots{}
                return j + 2
            return skip_ws(s, j) if name not in ("quad", "qquad") else j
        if name in IGNORED_WITH_ARG:
            _, j = group(s, j)
            return j
        formats = {"textbf": {"bold": True}, "textit": {"italic": True},
                   "emph": {"italic": not style.italic}, "textsuperscript": {"sup": True, "sub": False},
                   "textsubscript": {"sub": True, "sup": False}, "textup": {"italic": False},
                   "textnormal": {"bold": False, "italic": False}, "textrm": {}, "textsf": {}, "texttt": {},
                   "mbox": {}, "text": {}, "textsc": {}}
        if name in formats:
            arg, j = group(s, j)
            self.text(arg, replace(style, **formats[name]))
            return j
        if name == "enquote":
            arg, j = group(s, j)
            self.emit("“", style)
            self.text(arg, style)
            self.emit("”", style)
            return j
        if name == "footnote":
            arg, j = group(s, j)
            self.out.append(Run(note=tuple(Inline(self.article).convert(arg, Style()))))
            return j
        if name == "url":
            arg, j = group(s, j)
            self.emit(arg, style)
            return j
        if name == "href":
            _, j = group(s, j)
            arg, j = group(s, j)
            self.text(arg, style)
            return j
        if name in ("ref", "eqref", "autoref"):
            label, j = group(s, j)
            if label not in self.article.refs:
                raise LatexError(f"remissão sem destino: \\{name}{{{label}}}")
            number = self.article.refs[label]
            self.emit(f"({number})" if name == "eqref" else number, style)
            return j
        if name in CITE_COMMANDS:
            pre, j = optional(s, j)
            post, j = optional(s, j)
            if post is None:
                pre, post = None, pre
            keys, j = group(s, j)
            runs = self.article.bibliography.cite(name, [k.strip() for k in keys.split(",")], pre, post,
                                                  self.article)
            self.out.extend(replace(r, bold=r.bold or style.bold) for r in runs)
            return j
        raise LatexError(f"comando não suportado no .docx: \\{name}")

    def math(self, latex: str, style: Style) -> None:
        runs = MathText(latex, self.article).convert()
        if runs is None:
            self.out.append(Run(math=latex))
        else:
            self.out.extend(replace(r, bold=style.bold) for r in runs)


def tidy(runs: Runs) -> Runs:
    """Junta trechos vizinhos de mesma formatação e apara os espaços das pontas."""
    out: Runs = []
    for r in runs:
        if out and r.note is None and not r.math and out[-1].note is None and not out[-1].math and (
                (r.bold, r.italic, r.sub, r.sup) == (out[-1].bold, out[-1].italic, out[-1].sub, out[-1].sup)):
            text = out[-1].text + r.text
            out[-1] = replace(out[-1], text=re.sub(r" {2,}", " ", text))
        elif r.text or r.note is not None or r.math:
            out.append(r)
    if out and out[0].text:
        out[0] = replace(out[0], text=out[0].text.lstrip())
    if out and out[-1].text:
        out[-1] = replace(out[-1], text=out[-1].text.rstrip())
    return [r for r in out if r.text or r.note is not None or r.math]


# Fórmulas em linha simples viram texto (letras em itálico, índices); as demais, OMML
MATH_TOKEN = re.compile(r"\\[A-Za-z]+|\\[^A-Za-z]|[{}_^]|\s+|[^\\{}_^\s]")
GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
         "kappa": "κ", "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
         "phi": "φ", "varphi": "φ", "omega": "ω", "theta": "θ", "eta": "η", "chi": "χ", "psi": "ψ",
         "Delta": "Δ", "Sigma": "Σ", "Phi": "Φ", "Omega": "Ω", "Gamma": "Γ", "Lambda": "Λ", "Pi": "Π"}
MATH_SYMBOLS = {"cdot": "·", "times": "×", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠",
                "approx": "≈", "in": "∈", "infty": "∞", "to": "→", "ldots": "…", "pm": "±", "%": "%",
                "{": "{", "}": "}", ",": "\u2009", ";": " ", "quad": " ", "qquad": "  ", " ": " ", "!": "",
                "mid": "|", "sim": "~", "propto": "∝", "prime": "′", "lt": "<", "gt": ">"}
MATH_RELATIONS = {"cdot", "times", "le", "leq", "ge", "geq", "neq", "approx", "in", "to", "pm", "mid", "sim",
                  "propto", "lt", "gt"}       # com espaço dos dois lados, como no PDF
MATH_FUNCTIONS = {"ln", "exp", "log", "max", "min", "sin", "cos", "arg", "det", "lim", "Pr"}
MATH_TEXT = {"text": {}, "mathrm": {}, "textrm": {}, "operatorname": {}, "mathit": {"italic": True},
             "textit": {"italic": True}, "mathbf": {"bold": True}, "textbf": {"bold": True}}


class ComplexMath(Exception):
    pass


class MathText:
    def __init__(self, latex: str, article: Article) -> None:
        self.toks = MATH_TOKEN.findall(latex)
        self.i = 0
        self.article = article

    def convert(self) -> Runs | None:
        try:
            runs = self.seq(None, Style())
        except ComplexMath:
            return None
        return tidy(runs)

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def seq(self, stop: str | None, style: Style) -> Runs:
        out: Runs = []
        while (tok := self.peek()) is not None and tok != stop:
            out += self.atom(style)
        if stop is not None:
            self.i += 1
        return out

    def arg(self, style: Style) -> Runs:
        while (tok := self.peek()) is not None and tok.isspace():
            self.i += 1
        if self.peek() == "{":
            self.i += 1
            return self.seq("}", style)
        return self.atom(style)

    def atom(self, style: Style) -> Runs:
        tok = self.toks[self.i]
        self.i += 1
        run = lambda text, italic=False: [Run(text, False, italic, style.sub, style.sup)]  # noqa: E731
        if tok.isspace():
            return run(" ")
        if tok == "{":
            return self.seq("}", style)
        if tok in "_^":
            return self.arg(replace(style, sub=tok == "_", sup=tok == "^"))
        if not tok.startswith("\\"):
            return run("−" if tok == "-" else tok, tok.isalpha())
        name = tok[1:]
        if name.isalpha() and name not in MATH_FUNCTIONS:
            while (nxt := self.peek()) is not None and nxt.isspace():      # espaço após \comando
                self.i += 1
        if name in GREEK:
            return run(GREEK[name], name[0].islower())
        if name in MATH_SYMBOLS:
            symbol = MATH_SYMBOLS[name]
            return run(f" {symbol} " if name in MATH_RELATIONS else symbol)
        if name in MATH_FUNCTIONS:
            return run(name)
        if name in MATH_TEXT:
            if self.peek() != "{":
                raise ComplexMath
            depth, start = 0, self.i
            while True:
                t = self.toks[self.i]
                depth += t == "{"
                depth -= t == "}"
                self.i += 1
                if depth == 0:
                    break
            inner = "".join(self.toks[start + 1:self.i - 1])
            return [replace(r, sub=style.sub, sup=style.sup, italic=r.italic or bool(MATH_TEXT[name].get("italic")),
                            bold=r.bold or bool(MATH_TEXT[name].get("bold")))
                    for r in Inline(self.article).convert(inner, Style())]
        raise ComplexMath


# --------------------------------------------------------------------------------------------
# Bibliografia: .bib, citações (NBR 10520:2023) e referências (NBR 6023:2018)
# --------------------------------------------------------------------------------------------
CITE_COMMANDS = {"cite", "parencite", "autocite", "textcite", "citeauthor", "citeyear", "nocite"}
ET_AL = "et al."


@dataclass
class Name:
    family: str
    given: str = ""                     # iniciais, como as imprime o biblatex (giveninits)


@dataclass
class Entry:
    key: str
    kind: str
    fields: dict[str, str]
    names: list[Name]
    others: bool

    def get(self, name: str) -> str:
        return self.fields.get(name, "")

    @property
    def year(self) -> str:
        return self.get("year") or self.get("date")[:4]


def initials(given: str) -> str:
    """Prenomes em iniciais: "Su-In" → "S.-I.", "R. J." permanece."""
    if given.startswith("{") and given.endswith("}"):
        return given[1:-1]
    out = []
    for part in given.split():
        pieces = [p for p in part.split("-") if p]
        out.append("-".join(p if p.endswith(".") else p[0] + "." for p in pieces))
    return " ".join(out)


def parse_names(value: str) -> tuple[list[Name], bool]:
    names, others = [], False
    for raw in split_top(value, re.compile(r"\s+and\s+")):
        raw = " ".join(raw.split())
        if raw == "others":
            others = True
        elif raw.startswith("{") and raw.endswith("}"):
            names.append(Name(raw[1:-1]))
        elif "," in raw:
            family, given = (p.strip() for p in raw.split(",", 1))
            names.append(Name(family.replace("{", "").replace("}", ""), initials(given)))
        else:
            parts = raw.split()
            names.append(Name(parts[-1].replace("{", "").replace("}", ""), initials(" ".join(parts[:-1]))))
    return names, others


def parse_bib(text: str) -> dict[str, Entry]:
    entries: dict[str, Entry] = {}
    i = 0
    while (at := text.find("@", i)) >= 0:
        m = re.compile(r"@(\w+)\s*\{").match(text, at)
        if m is None:
            i = at + 1
            continue
        kind = m.group(1).lower()
        body, i = group(text, m.end() - 1)
        if kind in ("comment", "preamble", "string"):
            continue
        key, _, rest = body.partition(",")
        fields, j = {}, 0
        while fm := re.compile(r"\s*(\w+)\s*=\s*").match(rest, j):
            name, j = fm.group(1).lower(), fm.end()
            if rest[j] == "{":
                value, j = group(rest, j)
            elif rest[j] == '"':
                end = rest.index('"', j + 1)
                value, j = rest[j + 1:end], end + 1
            else:
                vm = re.compile(r"[^,\s}]+").match(rest, j)
                value, j = vm.group(0), vm.end()
            fields[name] = " ".join(value.split())
            j = skip_ws(rest, j)
            if j < len(rest) and rest[j] == ",":
                j += 1
        if "journal" in fields:
            fields.setdefault("journaltitle", fields.pop("journal"))
        names, others = parse_names(fields.get("author") or fields.get("editor", ""))
        entries[key.strip()] = Entry(key.strip(), kind, fields, names, others)
    return entries


class Bibliography:
    def __init__(self, entries: dict[str, Entry]) -> None:
        self.entries = entries
        self.cited: list[str] = []

    @classmethod
    def load(cls, paths: list[Path]) -> Bibliography:
        entries: dict[str, Entry] = {}
        for path in paths:
            entries.update(parse_bib(path.read_text(encoding="utf-8")))
        return cls(entries)

    def entry(self, key: str) -> Entry:
        if key not in self.entries:
            raise LatexError(f"citação sem referência no .bib: {key}")
        if key not in self.cited:
            self.cited.append(key)
        return self.entries[key]

    @staticmethod
    def label(e: Entry, textual: bool) -> Runs:
        """Autores na citação: "Müller et al.", "Diebold; Mariano" ou "Diebold e Mariano"."""
        families = [n.family for n in e.names]
        if e.others or len(families) > MAX_CITE_NAMES:
            return [Run(families[0] + " "), Run(ET_AL, italic=True)]
        return [Run((" e " if textual else "; ").join(families))]

    def cite(self, command: str, keys: list[str], pre: str | None, post: str | None,
             article: Article) -> Runs:
        entries = [self.entry(k) for k in keys]
        if command == "nocite":
            return []
        post_runs = article.runs(post) if post else []
        if command == "citeauthor":
            return join_runs([self.label(e, True) for e in entries], ", ")
        if command == "citeyear":
            return join_runs([[Run(e.year)] for e in entries], ", ")
        if command == "textcite":
            parts = [self.label(e, True) + [Run(f" ({e.year}")] for e in entries]
            if post_runs:
                parts[-1] += [Run(", ")] + post_runs
            return join_runs([p + [Run(")")] for p in parts], ", ", " e ")
        parts = [self.label(e, False) + [Run(f", {e.year}")] for e in entries]
        if post_runs:
            parts[-1] += [Run(", ")] + post_runs
        pre_runs = article.runs(pre) + [Run(" ")] if pre else []
        return [Run("(")] + pre_runs + join_runs(parts, "; ") + [Run(")")]

    @staticmethod
    def sort_key(e: Entry) -> tuple[str, str, str]:
        """Ordem do biber (nome, ano, título), com o et al. (``~``) depois dos coautores nomeados."""
        shown = e.names[:1] if e.others or len(e.names) > MAX_BIB_NAMES else e.names
        names = " ".join(f"{n.family} {n.given}" for n in shown) + (" ~" if len(shown) < len(e.names) or e.others else "")
        return (ascii_upper(names), e.year, ascii_upper(e.get("title")))

    def reference(self, e: Entry, article: Article) -> Runs:
        out: Runs = []
        field_runs = lambda name, **style: article.runs(e.get(name), **style)  # noqa: E731

        def add(text: str, **style: bool) -> None:
            out.append(Run(text, **style))

        # autoria: SOBRENOME, Iniciais; ...; com quatro ou mais, o primeiro e et al.
        shown = e.names[:1] if e.others or len(e.names) > MAX_BIB_NAMES else e.names
        add("; ".join(f"{n.family.upper()}, {n.given}" if n.given else n.family.upper() for n in shown))
        if len(shown) < len(e.names) or e.others:
            add(" ")
            add("et al", italic=True)
            add(".")
        add(" " if plain(out).endswith(".") else ". ")

        title = field_runs("title", bold=e.kind == "book")
        out += title
        if e.get("subtitle"):
            add(": ")
            out += field_runs("subtitle")
        add(" " if plain(out).endswith((".", "?", "!")) else ". ")

        if e.kind == "article":
            out += field_runs("journaltitle", bold=True)
            for name, prefix in (("volume", "v. "), ("number", "n. "), ("pages", "p. "), ("eid", "")):
                if e.get(name):
                    add(", " + prefix)
                    out += field_runs(name)
            add(f", {e.year}.")
        elif e.kind == "book":
            if e.get("edition"):
                add(f"{e.get('edition')}. ed. ")
            self.imprint(e, article, out)
        elif e.kind in ("inproceedings", "incollection"):
            add("In", italic=True)
            add(": ")
            if e.kind == "inproceedings":
                add(plain(field_runs("eventtitle")).upper())
                for name in ("number", "eventyear", "venue"):
                    if e.get(name):
                        add(", " + plain(field_runs(name)) + ("." if name == "number" else ""))
                add(". ")
                out += field_runs("booktitle", bold=True)
                add(" [...]. ")
            else:
                if e.get("editor"):
                    editors, _ = parse_names(e.get("editor"))
                    add("; ".join(f"{n.family.upper()}, {n.given}" for n in editors) + " (ed.). ")
                out += field_runs("booktitle", bold=True)
                add(". ")
            self.imprint(e, article, out)
            if e.get("pages"):
                add(" p. ")
                out += field_runs("pages")
                add(".")
        elif e.kind in ("online", "misc"):
            add(f"{e.year}.")
            if e.get("url"):
                add(f" Disponível em: {e.get('url')}.")
            if e.get("urldate"):
                add(f" Acesso em: {abnt_date(e.get('urldate'))}.")
        else:
            raise LatexError(f"tipo de referência não suportado no .docx: @{e.kind} ({e.key})")
        if e.get("addendum"):
            add(" ")
            out += field_runs("addendum")
        return tidy(out)

    @staticmethod
    def imprint(e: Entry, article: Article, out: Runs) -> None:
        """Local: Editora, ano."""
        place = plain(article.runs(e.get("location"))) or "[S. l.]"
        publisher = plain(article.runs(e.get("publisher"))) or "[s. n.]"
        out.append(Run(f"{place}: {publisher}, {e.year}."))


def join_runs(parts: list[Runs], sep: str, last: str | None = None) -> Runs:
    out: Runs = []
    for k, part in enumerate(parts):
        if k:
            out.append(Run(last if last is not None and k == len(parts) - 1 else sep))
        out += part
    return out


def ascii_upper(text: str) -> str:
    text = re.sub(r"[{}\\]", "", text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()


MONTHS = ["jan.", "fev.", "mar.", "abr.", "maio", "jun.", "jul.", "ago.", "set.", "out.", "nov.", "dez."]


def abnt_date(iso: str) -> str:
    year, month, day = (iso.split("-") + ["", ""])[:3]
    return " ".join(p for p in (str(int(day)) if day else "", MONTHS[int(month) - 1] if month else "", year) if p)
