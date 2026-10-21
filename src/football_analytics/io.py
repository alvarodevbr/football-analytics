"""Persistência: Parquet com sidecar de schema, caches JSON/HTML atômicos."""

from __future__ import annotations

import gzip
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .logging_utils import get_logger

log = get_logger("io")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Escreve via arquivo temporário + rename, para que uma interrupção nunca deixe cache corrompido."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def read_json(path: Path) -> Any | None:
    """Conteúdo do cache, ou None se a chave nunca foi buscada."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_gz(path: Path, text: str) -> None:
    _atomic_write_bytes(path, gzip.compress(text.encode("utf-8")))


def read_text_gz(path: Path) -> str | None:
    if not path.exists():
        return None
    return gzip.decompress(path.read_bytes()).decode("utf-8")


def save_table(
    df: pd.DataFrame,
    path: Path,
    source: str,
    description: dict[str, str] | None = None,
) -> Path:
    """Grava Parquet + ``<nome>_schema.json`` (colunas, dtypes, n de linhas, proveniência).

    O sidecar permite que o notebook seguinte valide o contrato da tabela antes de qualquer join.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index(drop=True)
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(path, index=False)

    schema = {
        "path": path.name,
        "source": source,
        "n_rows": int(len(out)),
        "n_columns": int(out.shape[1]),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "columns": {
            col: {"dtype": str(dtype), **({"description": description[col]} if description and col in description else {})}
            for col, dtype in out.dtypes.items()
        },
    }
    sidecar = path.with_name(path.stem + "_schema.json")
    sidecar.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("tabela gravada", extra={"path": str(path), "rows": len(out), "cols": out.shape[1]})
    return path


def load_table(path: Path, expect_columns: list[str] | None = None) -> pd.DataFrame:
    """Lê uma tabela gravada por `save_table` e confere o número de linhas e as colunas esperadas."""
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe — execute a etapa que a produz.")
    df = pd.read_parquet(path)
    sidecar = path.with_name(path.stem + "_schema.json")
    if sidecar.exists():
        schema = json.loads(sidecar.read_text(encoding="utf-8"))
        if len(df) != schema["n_rows"]:
            raise ValueError(f"{path.name}: {len(df)} linhas em disco, sidecar registra {schema['n_rows']}")
    missing = [c for c in (expect_columns or []) if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: colunas ausentes {missing}")
    return df
