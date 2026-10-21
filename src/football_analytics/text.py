"""Normalização de nomes de jogadores e clubes para a resolução de entidades.

Remove as três fontes de ruído descritas em specs/verificacao_dados.md §5.2: diacríticos e
codificação (ø, ß, ł, ı…), pontuação/hífens e caixa.
"""

from __future__ import annotations

import re
import unicodedata

# Letras que a decomposição NFKD não reduz a ASCII.
_TRANSLITERATION = str.maketrans({
    "ø": "o", "Ø": "o", "ß": "ss", "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ð": "d", "Ð": "d",
    "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "þ": "th", "Þ": "th", "ı": "i", "İ": "i",
    "ħ": "h", "Ħ": "h", "ŧ": "t", "ŋ": "n", "ſ": "s",
    "’": "'", "‘": "'", "`": "'", "´": "'",
})
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.translate(_TRANSLITERATION))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_name(name: object) -> str:
    """``"Martin Ødegaard"`` → ``"martin odegaard"``; ``"N'Golo Kanté"`` → ``"ngolo kante"``.

    Apóstrofos são removidos (não viram espaço) porque as fontes divergem entre "N'Golo" e
    "Ngolo"; hífens e pontos viram espaço ("Jean-Philippe" ≡ "Jean Philippe").
    """
    if name is None or (isinstance(name, float) and name != name):
        return ""
    text = strip_accents(str(name)).lower().replace("'", "")
    text = _NON_ALNUM.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


_CLUB_NOISE = re.compile(
    r"\b(fc|cf|afc|ac|as|ss|ssc|sc|sv|vfb|vfl|tsg|rc|rcd|ud|cd|sd|ca|club|calcio|ogc|osc|"
    r"stade|olympique|de|del|la|1|04|05|1899|1900|1907|1909|1910|1913|1846|hotspur|wanderers|"
    r"albion)\b"
)


def normalize_club(name: object) -> str:
    """Forma reduzida de nome de clube para comparação difusa ("AC Milan" ~ "Milan")."""
    base = normalize_name(name)
    reduced = _SPACES.sub(" ", _CLUB_NOISE.sub(" ", base)).strip()
    return reduced or base
