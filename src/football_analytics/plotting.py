"""Estilo gráfico para as figuras do artigo e registro das figuras exportadas.

As figuras saem sem título embutido: pela ABNT, o título ("Figura N – ...") fica acima da imagem no
texto e a fonte abaixo. Cada figura é registrada com uma legenda sugerida em um índice
(``figures_index.csv`` / ``.md``) ao lado dos PNGs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .logging_utils import get_logger

log = get_logger("plotting")

# Paleta de referência (validada para daltonismo nos três primeiros slots, em todos os pares).
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"
NEUTRAL_LIGHT = "#f0efec"
SURFACE = "#ffffff"

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
POSITION_COLORS: dict[str, str] = {"DEF": BLUE, "MEI": ORANGE, "ATA": AQUA}
# Rampa ordinal de um só matiz para as faixas etárias (ordem natural: mais jovem = mais claro).
AGE_BAND_COLORS: tuple[str, ...] = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("div_red_blue", [RED, NEUTRAL_LIGHT, BLUE])

# Largura útil de uma página A4 com margens ABNT (3 cm + 2 cm): 16 cm.
TEXT_WIDTH_IN = 16 / 2.54


def setup_style() -> None:
    """Fonte métrica-compatível com Arial (exigência ABNT), traços finos e grade discreta."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "patch.linewidth": 0,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.color": INK,
    })


def figsize(height_cm: float, width_cm: float = 16) -> tuple[float, float]:
    return width_cm / 2.54, height_cm / 2.54


def br(x: float, decimals: int = 1) -> str:
    """Número no padrão brasileiro: ponto de milhar e vírgula decimal (``12.408,5``)."""
    text = f"{x:,.{decimals}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".").replace("-", "−")


def br_pct(x: float, decimals: int = 0) -> str:
    return f"{br(x, decimals)}%"


def br_formatter(decimals: int = 0) -> mticker.FuncFormatter:
    return mticker.FuncFormatter(lambda x, _pos: br(x, decimals))


def pct_formatter(decimals: int = 0, scale: float = 100) -> mticker.FuncFormatter:
    """Rótulos de eixo em %, para dados em [0, scale]."""
    return mticker.FuncFormatter(lambda x, _pos: br_pct(x / scale * 100, decimals))


def annotations(df: pd.DataFrame, decimals: int = 1) -> pd.DataFrame:
    """Matriz de rótulos para `sns.heatmap(annot=..., fmt="")` no padrão brasileiro."""
    return df.map(lambda v: "" if pd.isna(v) else br(v, decimals))


def euro_millions(x: float, _pos: object = None) -> str:
    if x >= 1e6:
        return f"€{br(x / 1e6, 0 if x >= 10e6 or x % 1e6 == 0 else 1)} mi"
    if x >= 1e3:
        return f"€{br(x / 1e3, 0)} mil"
    return f"€{br(x, 0)}"


@dataclass
class FigureRegistry:
    """Salva PNGs numerados e mantém o índice de legendas para o texto do artigo."""

    directory: Path
    source: str
    entries: list[dict[str, str]] = field(default_factory=list)

    def save(self, fig: plt.Figure, name: str, caption: str, note: str = "") -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{name}.png"
        fig.savefig(path)
        entry = {"arquivo": path.name, "legenda": caption, "fonte": self.source, "nota": note}
        for existing in self.entries:  # reexecução da célula: mantém o número da figura
            if existing["arquivo"] == path.name:
                existing.update(entry)
                break
        else:
            self.entries.append({"figura": str(len(self.entries) + 1), **entry})
        log.info("figura gravada", extra={"path": str(path)})
        return path

    def write_index(self) -> pd.DataFrame:
        index = pd.DataFrame(self.entries)
        index.to_csv(self.directory / "figures_index.csv", index=False)
        lines = ["# Índice de figuras", "",
                 "Legendas sugeridas (ABNT: título acima da figura, fonte abaixo).", ""]
        for e in self.entries:
            lines += [f"**Figura {e['figura']} – {e['legenda']}**  ", f"`{e['arquivo']}`  ",
                      f"Fonte: {e['fonte']}." + (f" Nota: {e['nota']}" if e["nota"] else ""), ""]
        (self.directory / "figures_index.md").write_text("\n".join(lines), encoding="utf-8")
        return index


def beeswarm(ax: plt.Axes, phi: pd.DataFrame, values: pd.DataFrame, features: list[str], labels: list[str],
             max_points: int = 2500, seed: int = 0) -> mpl.cm.ScalarMappable:
    """Gráfico de enxame SHAP no estilo do projeto (substitui `shap.summary_plot`).

    Cada linha é uma variável; a dispersão vertical é proporcional à densidade local dos valores
    SHAP e a cor é o percentil do valor da variável (vermelho = baixo, azul = alto).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(phi), size=min(max_points, len(phi)), replace=False)
    for row, f in enumerate(features):
        x = phi[f].to_numpy()[idx]
        v = values[f].to_numpy()[idx]
        hist, edges = np.histogram(x, bins=60)
        dens = hist[np.clip(np.digitize(x, edges) - 1, 0, len(hist) - 1)] / max(hist.max(), 1)
        y = row + rng.uniform(-1, 1, size=len(x)) * 0.38 * dens
        rank = pd.Series(v).rank(pct=True).to_numpy()
        missing = np.isnan(v)
        ax.scatter(x[missing], y[missing], s=3, color=NEUTRAL, linewidths=0, alpha=0.5, rasterized=True)
        ax.scatter(x[~missing], y[~missing], s=3, c=rank[~missing], cmap=DIVERGING, vmin=0, vmax=1,
                   linewidths=0, alpha=0.6, rasterized=True)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_yticks(range(len(features)), labels)
    ax.set_ylim(len(features) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    return mpl.cm.ScalarMappable(cmap=DIVERGING, norm=mpl.colors.Normalize(0, 1))
