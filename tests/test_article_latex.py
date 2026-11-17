import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "article"))
from latex_reader import LatexError, plain, read_article  # noqa: E402

TEX = r"""
\documentclass[article]{abntex2}
\addbibresource{refs.bib}
\titulo{Título com \textit{itálico}}
\adicionarautor[Nota um.]{Autora Um}
\adicionarautor{Autor Dois}
\orientacao{Orientadora}{Profa. Três}
\palavraschave{a; b.}
\begin{document}
\imprimircabecalho
\begin{resumo}
Resumo curto. % comentário
\end{resumo}

\section{Introdução}
\label{sec:intro}

Texto com $V_{T+1}$, $\Delta V\%$ e $k \in \{1, 2\}$ \cite{a2020,b2019}, e \textcite{c2018}.
\begin{equation}
  y = \frac{a}{b}
  \label{eq:y}
\end{equation}

Ver a equação~\eqref{eq:y}, a Tabela~\ref{tab:t} e a seção~\ref{sec:intro}.

\begin{table}[htbp]
  \caption{Tabela}
  \label{tab:t}
  \begin{tabularx}{\textwidth}{Lr}
    \toprule
    A & B \\
    \midrule
    x & 1,0\% \\
    \bottomrule
  \end{tabularx}
  \fonte{Elaboração própria.}
\end{table}

\begin{alineas}[label=H\arabic*:]
  \item primeira;
  \item segunda.
\end{alineas}

\printbibliography[heading=referencias]
\end{document}
"""

BIB = r"""
@article{a2020, author = {Alfa, B. and Beta, C. and Gama, D. and Delta, E.}, title = {Um título},
  journaltitle = {Revista \& Cia}, volume = {2}, number = {1}, pages = {10--20}, year = {2020}}
@book{b2019, author = {Bê, F. and Cê, G.}, title = {Livro}, subtitle = {subtítulo}, edition = {2},
  location = {São Paulo}, publisher = {Editora}, year = {2019}}
@inproceedings{c2018, author = {Cid, Hugo Lima}, title = {{XGBoost}: anais}, eventtitle = {Conference on Things},
  number = {5}, eventyear = {2018}, venue = {Rio}, booktitle = {Anais}, location = {Rio}, publisher = {SBC},
  pages = {1--9}, year = {2018}}
"""


@pytest.fixture
def article(tmp_path):
    (tmp_path / "refs.bib").write_text(BIB, encoding="utf-8")
    tex = tmp_path / "artigo.tex"
    tex.write_text(TEX, encoding="utf-8")
    return read_article(tex)


def test_structure_numbering_and_tables(article):
    kinds = [b.kind for b in article.blocks]
    assert kinds == ["heading", "para", "equation", "para", "tabela", "list", "references"]
    assert article.abstract == "Resumo curto."
    assert article.authors == [("Autora Um", "Nota um."), ("Autor Dois", "")]
    assert article.orientation == ("Orientadora", "Profa. Três", "")
    assert article.refs == {"sec:intro": "1", "eq:y": "1", "tab:t": "1"}
    table = article.blocks[4]
    assert table.rows == [["A", "B"], ["x", r"1,0\%"]] and table.aligns == ["left", "right"]
    assert table.header_rows == 1 and table.source == "Elaboração própria."
    assert article.blocks[5].items == [("H1:", "primeira;"), ("H2:", "segunda.")]


def test_inline_text_math_and_citations(article):
    runs = article.runs(article.blocks[1].text)
    assert plain(runs) == "Texto com VT+1, ΔV% e k ∈ {1, 2} (Alfa et al., 2020; Bê; Cê, 2019), e Cid (2018)."
    sub = next(r for r in runs if r.sub)
    assert sub.text.startswith("T") and sub.italic
    assert any(r.text == "et al." and r.italic for r in runs)
    ref_runs = article.runs(article.blocks[3].text)
    assert plain(ref_runs) == "Ver a equação (1), a Tabela 1 e a seção 1."


def test_references_follow_abnt(article):
    for block in article.blocks:                 # as citações entram na lista ao serem convertidas
        if block.kind == "para":
            article.runs(block.text)
    refs = [plain(r) for r in article.references()]
    assert refs == [
        "ALFA, B. et al. Um título. Revista & Cia, v. 2, n. 1, p. 10–20, 2020.",
        "BÊ, F.; CÊ, G. Livro: subtítulo. 2. ed. São Paulo: Editora, 2019.",
        "CID, H. L. XGBoost: anais. In: CONFERENCE ON THINGS, 5., 2018, Rio. Anais [...]. Rio: SBC, 2018. p. 1–9.",
    ]
    journal = next(r for r in article.references()[0] if r.text.startswith("Revista"))
    assert journal.bold


def test_unsupported_command_is_reported(article):
    with pytest.raises(LatexError, match=r"\\vfill"):
        article.runs(r"texto \vfill")


def test_english_abstract_and_appendix(tmp_path):
    (tmp_path / "refs.bib").write_text(BIB, encoding="utf-8")
    tex = TEX.replace(r"\palavraschave{a; b.}", "\\palavraschave{a; b.}\n\\palavraschaveingles{x; y.}")
    tex = tex.replace("\\end{resumo}", "\\end{resumo}\n\\begin{abstract}\nShort abstract.\n\\end{abstract}")
    tex = tex.replace("\\printbibliography[heading=referencias]",
                      "\\printbibliography[heading=referencias]\n\\apendice{Robustez}\n\\label{ap:rob}\n\nVer o"
                      " Apêndice~\\ref{ap:rob}.")
    path = tmp_path / "artigo.tex"
    path.write_text(tex, encoding="utf-8")
    art = read_article(path)
    assert art.abstract == "Resumo curto." and art.abstract_en == "Short abstract."
    assert art.keywords_en == "x; y."
    heading = next(b for b in art.blocks if b.kind == "heading" and not b.numbered)
    assert heading.number == "A" and "Robustez" in heading.text
    assert art.refs["ap:rob"] == "A"
    assert plain(art.runs(art.blocks[-1].text)) == "Ver o Apêndice\u00a0A."
