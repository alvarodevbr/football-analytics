"""FBref: estatísticas de jogador por temporada a partir das páginas combinadas das Big 5.

A busca usa `soccerdata.FBref.get()` (Chrome + rate limit de 7 s + cache em disco), única via
que passa pelo Cloudflare. O parsing é próprio: lê as células pelo atributo ``data-stat``
(nomes de máquina estáveis) e preserva o id do jogador e do clube no FBref, que o soccerdata
descarta e que são a chave longitudinal do painel.

Desde 20/01/2026 o FBref não publica mais as métricas avançadas licenciadas pela Opta: as
páginas `passing`, `possession`, `gca` e grande parte de `defense` mantêm o cabeçalho, mas com
todas as células vazias, inclusive nas temporadas antigas. `column_fill_rates` documenta isso.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import html as lxml_html

from .config import ExtractionConfig, Season
from .logging_utils import get_logger

log = get_logger("fbref")

BIG5_URL = "https://fbref.com/en/comps/Big5/{label}/{slug}/players/{label}-Big-5-European-Leagues-Stats"

# página lógica -> (slug na URL, id da tabela no HTML)
PAGE_SPECS: dict[str, tuple[str, str]] = {
    "standard": ("stats", "stats_standard"),
    "shooting": ("shooting", "stats_shooting"),
    "playing_time": ("playingtime", "stats_playing_time"),
    "misc": ("misc", "stats_misc"),
    "passing": ("passing", "stats_passing"),
    "defense": ("defense", "stats_defense"),
    "possession": ("possession", "stats_possession"),
    "gca": ("gca", "stats_gca"),
}

IDENTITY_STATS = ("player", "nationality", "position", "team", "comp_level", "age", "birth_year")
_SKIP_STATS = {"ranker", "matches"}
KEY = ["season", "fbref_player_id", "fbref_team_id"]

_PLAYER_HREF = re.compile(r"/en/players/([0-9a-f]{8})/")
_TEAM_HREF = re.compile(r"/en/squads/([0-9a-f]{8})/")


class FBrefParseError(ValueError):
    pass


def page_url(season: Season, page: str) -> str:
    slug, _ = PAGE_SPECS[page]
    return BIG5_URL.format(label=season.fbref_label, slug=slug)


def page_cache_path(cache_dir: Path, season: Season, page: str) -> Path:
    return cache_dir / f"big5_{season.code}_{page}.html"


def _to_number(text: str) -> float:
    cleaned = text.replace(",", "").strip()
    if cleaned in ("", "-"):
        return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def parse_player_table(page_html: str | bytes, page: str) -> pd.DataFrame:
    """Uma linha por (jogador, clube) da tabela ``stats_<page>``; colunas = valores de ``data-stat``.

    Linhas de cabeçalho repetido (``class="thead"``) são ignoradas. Valores numéricos viram float;
    vazio vira NaN (não zero), preservando a distinção entre "não publicado" e "zero".
    """
    _, table_id = PAGE_SPECS[page]
    tree = lxml_html.fromstring(page_html)
    tables = tree.xpath(f"//table[@id='{table_id}']")
    if not tables:
        # Páginas por liga escondem tabelas secundárias em comentários HTML.
        for comment in tree.xpath(f"//comment()[contains(., 'id=\"{table_id}\"')]"):
            tables = lxml_html.fromstring(comment.text).xpath(f"//table[@id='{table_id}']")
            if tables:
                break
    if not tables:
        raise FBrefParseError(f"tabela {table_id} não encontrada")

    records: list[dict[str, object]] = []
    for tr in tables[0].xpath("./tbody/tr"):
        if "thead" in (tr.get("class") or ""):
            continue
        cells = {td.get("data-stat"): td for td in tr.xpath("./td") if td.get("data-stat")}
        player_td = cells.get("player")
        if player_td is None:
            continue
        player_link = player_td.xpath(".//a/@href")
        player_id = player_td.get("data-append-csv") or (
            _PLAYER_HREF.search(player_link[0]).group(1) if player_link and _PLAYER_HREF.search(player_link[0]) else None
        )
        team_td = cells.get("team")
        team_href = team_td.xpath(".//a/@href") if team_td is not None else []
        team_match = _TEAM_HREF.search(team_href[0]) if team_href else None
        comp_td = cells.get("comp_level")

        record: dict[str, object] = {
            "fbref_player_id": player_id,
            "player": player_td.text_content().strip(),
            "fbref_team_id": team_match.group(1) if team_match else None,
            "team": team_td.text_content().strip() if team_td is not None else None,
            # O texto da célula traz o código do país antes do nome ("eng Premier League").
            "comp": (comp_td.xpath(".//a/text()") or [comp_td.text_content()])[0].strip() if comp_td is not None else None,
            "nationality": (cells["nationality"].text_content().split() or [None])[-1] if "nationality" in cells else None,
            "position": cells["position"].text_content().strip() if "position" in cells else None,
            "age": _to_number(cells["age"].text_content().split("-")[0]) if "age" in cells else np.nan,
            "birth_year": _to_number(cells["birth_year"].text_content()) if "birth_year" in cells else np.nan,
        }
        for stat, td in cells.items():
            if stat in IDENTITY_STATS or stat in _SKIP_STATS:
                continue
            record[stat] = _to_number(td.text_content())
        records.append(record)

    if not records:
        raise FBrefParseError(f"tabela {table_id} sem linhas de jogador")
    return pd.DataFrame.from_records(records)


def column_fill_rates(df: pd.DataFrame) -> pd.Series:
    """Fração de células não vazias por coluna de estatística (diagnóstico da remoção Opta)."""
    stat_cols = [c for c in df.columns if c not in {"fbref_player_id", "player", "fbref_team_id", "team", "comp",
                                                    "nationality", "position", "age", "birth_year"}]
    return df[stat_cols].notna().mean().sort_values()


class FBrefFetcher:
    """Busca páginas Big 5 com cache em disco; o Chrome só é aberto se faltar alguma página."""

    def __init__(self, cfg: ExtractionConfig) -> None:
        self.cfg = cfg
        self.cache_dir = cfg.paths.fbref_html
        self._reader = None

    def _get_reader(self):  # noqa: ANN202 - tipo do soccerdata
        if self._reader is None:
            import soccerdata as sd

            from .browser import find_chrome

            self._reader = sd.FBref(
                leagues="Big 5 European Leagues Combined",
                seasons=[s.code for s in self.cfg.seasons],
                data_dir=self.cfg.paths.soccerdata_dir / "FBref",
                path_to_browser=find_chrome(),
                headless=self.cfg.browser_headless,
            )
        return self._reader

    def fetch(self, season: Season, page: str) -> str:
        path = page_cache_path(self.cache_dir, season, page)
        if path.exists() and path.stat().st_size > 50_000:
            return path.read_text(encoding="utf-8")
        url = page_url(season, page)
        log.info("baixando página FBref", extra={"season": season.code, "page": page, "url": url})
        with self._get_reader().get(url, path) as handle:
            content = handle.read()
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        if f'id="{PAGE_SPECS[page][1]}"' not in text:
            path.unlink(missing_ok=True)
            raise FBrefParseError(f"{url}: resposta sem a tabela esperada (desafio Cloudflare?)")
        return text

    def close(self) -> None:
        if self._reader is not None and hasattr(self._reader, "_driver"):
            try:
                self._reader._driver.quit()  # noqa: SLF001
            except Exception:  # noqa: BLE001 - encerramento best-effort
                pass
        self._reader = None


def merge_pages(frames: dict[str, pd.DataFrame], comp_to_league: dict[str, str]) -> pd.DataFrame:
    """Une as páginas de uma temporada pela chave (jogador, clube).

    `standard` é a base; colunas que reaparecem em outras páginas (gols, cartões, minutos...) só
    entram se ainda não existirem, e a coincidência dos valores é verificada.
    """
    base = frames["standard"].copy()
    identity = ["fbref_player_id", "player", "fbref_team_id", "team", "comp", "nationality",
                "position", "age", "birth_year"]
    for page, frame in frames.items():
        if page == "standard":
            continue
        new_cols = [c for c in frame.columns if c not in identity and c not in base.columns]
        shared = [c for c in frame.columns if c not in identity and c in base.columns]
        joined = base.merge(frame[["fbref_player_id", "fbref_team_id", *new_cols, *shared]],
                            on=["fbref_player_id", "fbref_team_id"], how="left",
                            suffixes=("", f"__{page}"), validate="one_to_one")
        for col in shared:
            other = joined.pop(f"{col}__{page}")
            both = joined[col].notna() & other.notna()
            mismatch = int((joined.loc[both, col] != other[both]).sum())
            if mismatch:
                log.warning("coluna repetida diverge entre páginas",
                            extra={"column": col, "page": page, "rows": mismatch})
        base = joined

    base["league"] = base["comp"].map(comp_to_league)
    unmapped = base.loc[base["league"].isna(), "comp"].unique().tolist()
    if unmapped:
        raise FBrefParseError(f"competições sem mapeamento para liga: {unmapped}")
    return base.drop(columns=["comp"])


def build_player_season_table(
    fetcher: FBrefFetcher,
    seasons: Iterable[Season],
    pages: Iterable[str],
    comp_to_league: dict[str, str],
    leagues: Iterable[str],
) -> pd.DataFrame:
    """Tabela jogador-temporada-clube para todas as temporadas, com prefixo ``fb_`` nas métricas."""
    pages = list(pages)
    out: list[pd.DataFrame] = []
    for season in seasons:
        frames = {page: parse_player_table(fetcher.fetch(season, page), page) for page in pages}
        merged = merge_pages(frames, comp_to_league)
        merged.insert(0, "season", season.code)
        out.append(merged)
        log.info("temporada FBref consolidada", extra={"season": season.code, "rows": len(merged)})

    df = pd.concat(out, ignore_index=True)
    df = df[df["league"].isin(list(leagues))].reset_index(drop=True)

    dupes = df.duplicated(KEY, keep=False)
    if dupes.any():
        raise FBrefParseError(f"{int(dupes.sum())} linhas com chave {KEY} duplicada")

    # Coluna integralmente vazia = métrica não publicada; descartá-la evita falsas features.
    stat_cols = [c for c in df.columns if c not in {"season", "league", *KEY, "player", "team",
                                                    "nationality", "position", "age", "birth_year"}]
    empty = [c for c in stat_cols if df[c].notna().sum() == 0]
    if empty:
        log.info("colunas FBref sem nenhum valor descartadas", extra={"columns": empty})
        df = df.drop(columns=empty)
        stat_cols = [c for c in stat_cols if c not in empty]

    df["n_teams_in_season"] = df.groupby(["season", "fbref_player_id"])["fbref_team_id"].transform("nunique")
    df["is_goalkeeper"] = df["position"].fillna("").str.contains("GK")
    df = df.rename(columns={c: f"fb_{c}" for c in stat_cols})
    df = df.rename(columns={"player": "fb_player", "team": "fb_team", "nationality": "fb_nationality",
                            "position": "fb_position", "age": "fb_age", "birth_year": "fb_birth_year"})
    return df
