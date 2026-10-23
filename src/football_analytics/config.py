"""Configuração central da extração: ligas, temporadas, caminhos e parâmetros de coleta.

Todas as convenções de temporada partem do código FBref/soccerdata de quatro dígitos
(``"2324"`` = 2023/24); os formatos das outras fontes são derivados por `Season`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Primeiro diretório (subindo a partir de `start`) que contém o ``pyproject.toml`` do projeto."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "football_analytics").exists():
            return candidate
    raise FileNotFoundError(f"pyproject.toml do projeto não encontrado acima de {here}")


@dataclass(frozen=True)
class Season:
    """Uma temporada europeia, identificada pelo código de quatro dígitos (``"2324"``)."""

    code: str

    def __post_init__(self) -> None:
        if len(self.code) != 4 or not self.code.isdigit():
            raise ValueError(f"código de temporada inválido: {self.code!r}")
        if (int(self.code[:2]) + 1) % 100 != int(self.code[2:]):
            raise ValueError(f"temporada não consecutiva: {self.code!r}")

    @property
    def start_year(self) -> int:
        return 2000 + int(self.code[:2])

    @property
    def fbref_label(self) -> str:
        """``"2023-2024"``, usado nas URLs do FBref."""
        return f"{self.start_year}-{self.start_year + 1}"

    @property
    def tm_saison_id(self) -> str:
        """Transfermarkt indexa a temporada pelo ano de início."""
        return str(self.start_year)

    @property
    def sofascore_year(self) -> str:
        """``"23/24"``, rótulo do campo ``year`` na lista de temporadas do Sofascore."""
        return f"{self.code[:2]}/{self.code[2:]}"

    @property
    def end_anchor(self) -> date:
        """Data-âncora do valor de mercado da temporada: 30/jun do ano de término."""
        return date(self.start_year + 1, 6, 30)

    def offset(self, n: int) -> "Season":
        start = self.start_year + n
        return Season(f"{start % 100:02d}{(start + 1) % 100:02d}")

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True)
class League:
    """Identificadores de uma liga em cada fonte."""

    key: str                 # chave canônica (padrão soccerdata), ex.: "ENG-Premier League"
    fbref_comp: str          # texto da coluna `comp_level` nas páginas Big 5 do FBref
    tm_slug: str
    tm_competition_id: str
    sofascore_tournament_id: int


BIG5_LEAGUES: tuple[League, ...] = (
    League("ENG-Premier League", "Premier League", "premier-league", "GB1", 17),
    League("ESP-La Liga", "La Liga", "laliga", "ES1", 8),
    League("ITA-Serie A", "Serie A", "serie-a", "IT1", 23),
    League("GER-Bundesliga", "Bundesliga", "bundesliga", "L1", 35),
    League("FRA-Ligue 1", "Ligue 1", "ligue-1", "FR1", 34),
)

# Temporadas de referência T do estudo (CLAUDE.md §1) e a temporada anterior, coletada apenas
# para permitir a construção de lags de desempenho de T = 2018/19 no notebook 03.
FEATURE_SEASONS: tuple[str, ...] = ("1819", "1920", "2021", "2122", "2223", "2324", "2425")
LAG_ONLY_SEASONS: tuple[str, ...] = ("1718",)

# Última data-âncora com valor de mercado já observado na coleta (out/2026): o alvo de T+k só existe
# se 30/jun de T+k for até esta data. Com ela, 2024/25 tem alvo em T+1 (30/06/2026), mas não em T+2.
VALUE_CUTOFF: date = date(2026, 6, 30)
# Teste final de cada horizonte. O de T+1 é pré-registrado: nenhuma decisão de modelagem pode usar
# seus alvos antes do commit de `reports/protocol_v3.json`. O de T+2 é semi-independente (consultado
# nas versões 1 e 2 do estudo).
HOLDOUT_SEASONS: dict[int, str] = {1: "2425", 2: "2324"}


def target_available(season: str, horizon: int, cutoff: date = VALUE_CUTOFF) -> bool:
    """O valor em T+k da temporada já foi observado até `cutoff`?"""
    return Season(season).offset(horizon).end_anchor <= cutoff

# Páginas do FBref que ainda carregam valores após a remoção dos dados Opta (jan/2026).
FBREF_PAGES: tuple[str, ...] = ("standard", "shooting", "playing_time", "misc")
# Páginas que o FBref ainda publica, mas com todas as colunas avançadas vazias. São baixadas
# para uma única temporada apenas para documentar a ausência (seção de Metodologia).
FBREF_EMPTY_PROBE_PAGES: tuple[str, ...] = ("passing", "defense", "possession", "gca")
# Temporada usada para documentar as páginas vazias (já em cache desde a v1).
FBREF_PROBE_SEASON = "2324"

# Métricas solicitadas à API de estatísticas por temporada do Sofascore.
SOFASCORE_FIELDS: tuple[str, ...] = (
    "minutesPlayed", "appearances", "matchesStarted", "rating",
    "goals", "assists", "expectedGoals", "expectedAssists",
    "totalShots", "shotsOnTarget", "bigChancesCreated", "bigChancesMissed",
    "totalPasses", "accuratePasses", "accuratePassesPercentage", "keyPasses",
    "accurateLongBalls", "accurateLongBallsPercentage", "accurateFinalThirdPasses", "accurateCrosses",
    "successfulDribbles", "successfulDribblesPercentage", "touches", "possessionLost", "dispossessed",
    "tackles", "interceptions", "clearances", "ballRecovery", "outfielderBlocks", "dribbledPast",
    "aerialDuelsWon", "aerialDuelsWonPercentage", "groundDuelsWon", "groundDuelsWonPercentage",
    "totalDuelsWon", "totalDuelsWonPercentage", "errorLeadToShot", "errorLeadToGoal",
    "wasFouled", "fouls", "yellowCards", "redCards",
)


@dataclass(frozen=True)
class Paths:
    """Layout de diretórios. `data_root` é separável para que o smoke test não toque os dados reais."""

    root: Path
    data_root: Path | None = None

    @property
    def data(self) -> Path:
        return self.data_root or (self.root / "data")

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def external(self) -> Path:
        """Tabelas manuais versionadas (aliases, overrides), compartilhadas com o smoke test."""
        return self.root / "data" / "external"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def soccerdata_dir(self) -> Path:
        """Cache próprio do soccerdata (isola o ``~/soccerdata`` global)."""
        return self.raw / "soccerdata"

    @property
    def fbref_html(self) -> Path:
        return self.raw / "fbref"

    @property
    def sofascore(self) -> Path:
        return self.raw / "sofascore"

    @property
    def transfermarkt(self) -> Path:
        return self.raw / "transfermarkt"

    @property
    def final_dataset(self) -> Path:
        return self.raw / "football_raw_dataset.parquet"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def reports(self) -> Path:
        """Artefatos para o artigo (figuras e tabelas), versionáveis."""
        return self.root / "reports"

    def figures(self, notebook: str) -> Path:
        return self.reports / "figures" / notebook

    def tables(self, notebook: str) -> Path:
        return self.reports / "tables" / notebook

    def ensure(self) -> None:
        for directory in (self.raw, self.interim, self.external, self.logs, self.soccerdata_dir,
                          self.fbref_html, self.sofascore, self.transfermarkt):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExtractionConfig:
    """Parâmetros de uma execução do notebook 01."""

    paths: Paths
    leagues: tuple[League, ...] = BIG5_LEAGUES
    feature_seasons: tuple[str, ...] = FEATURE_SEASONS
    lag_only_seasons: tuple[str, ...] = LAG_ONLY_SEASONS
    fbref_pages: tuple[str, ...] = FBREF_PAGES
    fbref_empty_probe_pages: tuple[str, ...] = FBREF_EMPTY_PROBE_PAGES
    fbref_probe_season: str = FBREF_PROBE_SEASON
    sofascore_fields: tuple[str, ...] = SOFASCORE_FIELDS
    # Cloudflare (FBref) e o anti-bot do Sofascore reprovam o Chrome headless.
    browser_headless: bool = False
    tm_delay_seconds: float = 1.0
    sofascore_delay_seconds: float = 1.5
    request_jitter_seconds: float = 0.5
    # Limite de requisições de rede por endpoint e por execução (None = sem limite); o cache torna
    # a coleta retomável em várias sessões.
    max_fetches_per_run: int | None = None
    # Validade máxima (dias) do último valor de mercado anterior à data-âncora.
    value_max_staleness_days: int = 183
    value_cutoff: date = VALUE_CUTOFF
    holdout_seasons: dict[int, str] = field(default_factory=lambda: dict(HOLDOUT_SEASONS))
    smoke_test: bool = False
    smoke_max_players: int = 40
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    )
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def seasons(self) -> tuple[Season, ...]:
        codes = sorted(set(self.feature_seasons) | set(self.lag_only_seasons))
        return tuple(Season(c) for c in codes)

    def target_seasons(self, horizon: int) -> tuple[str, ...]:
        """Temporadas de referência cujo alvo em T+k já é observável."""
        return tuple(s for s in sorted(self.feature_seasons) if target_available(s, horizon, self.value_cutoff))

    @property
    def league_keys(self) -> tuple[str, ...]:
        return tuple(league.key for league in self.leagues)

    def league(self, key: str) -> League:
        for league in self.leagues:
            if league.key == key:
                return league
        raise KeyError(key)

    def as_smoke_test(self) -> "ExtractionConfig":
        """Recorte mínimo (Premier League 2023/24) para validar o pipeline ponta a ponta."""
        return replace(
            self,
            leagues=self.leagues[:1],
            feature_seasons=("2324",),
            lag_only_seasons=(),
            smoke_test=True,
            paths=Paths(self.paths.root, data_root=self.paths.root / "data" / "smoke_test"),
        )

    def describe(self) -> dict[str, object]:
        return {
            "root": str(self.paths.root),
            "leagues": list(self.league_keys),
            "seasons": [s.code for s in self.seasons],
            "feature_seasons": list(self.feature_seasons),
            "lag_only_seasons": list(self.lag_only_seasons),
            "value_cutoff": self.value_cutoff.isoformat(),
            "holdout_seasons": {f"T+{k}": v for k, v in self.holdout_seasons.items()},
            "fbref_pages": list(self.fbref_pages),
            "smoke_test": self.smoke_test,
            "max_fetches_per_run": self.max_fetches_per_run,
            "browser_headless": self.browser_headless,
        }


def default_config(root: Path | None = None, **overrides: object) -> ExtractionConfig:
    """Configuração padrão ancorada na raiz do projeto; `overrides` substitui campos pontuais."""
    cfg = ExtractionConfig(paths=Paths(root or find_project_root()))
    return replace(cfg, **overrides) if overrides else cfg  # type: ignore[arg-type]
