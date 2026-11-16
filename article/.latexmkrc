# Compila artigo.tex com LuaLaTeX e biber: `latexmk` (ou `latexmk -pvc` para recompilar ao salvar).
# Os arquivos auxiliares ficam em build/; o PDF final, em article/artigo.pdf.
@default_files = ('artigo.tex');
$pdf_mode = 4;                 # lualatex
$bibtex_use = 2;               # roda o biber quando o .bib ou as citações mudam
$aux_dir = 'build';
$out_dir = '.';
$lualatex = 'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error %O %S';
