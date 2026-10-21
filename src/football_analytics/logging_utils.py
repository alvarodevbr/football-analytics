"""Logging estruturado: JSON-lines em arquivo (auditável) e texto legível no console/notebook."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonLinesFormatter(logging.Formatter):
    """Um objeto JSON por linha; campos passados em ``extra=`` viram chaves de primeiro nível."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in vars(record).items() if k not in _RESERVED}
        suffix = "  " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        return f"{record.levelname[0]} {record.name.split('.')[-1]:>16s} | {record.getMessage()}{suffix}"


def setup_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Configura o logger raiz do pacote (idempotente: reexecutar a célula não duplica handlers)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("football_analytics")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JsonLinesFormatter())
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"football_analytics.{name}")
