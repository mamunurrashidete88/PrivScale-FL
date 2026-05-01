"""Structured JSONL logging for experiment runs."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


def setup_stdout_logger(name: str = "privscale", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


class JsonlLogger:
    """Append-only JSONL logger keyed by a run directory."""

    def __init__(self, run_dir: str | Path, filename: str = "metrics.jsonl"):
        self.path = Path(run_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)

    def log(self, record: dict[str, Any]) -> None:
        record = {"_t": time.time(), **record}
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
