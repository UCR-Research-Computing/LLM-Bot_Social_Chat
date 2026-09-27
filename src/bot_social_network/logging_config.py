"""Structured JSONL logging, one folder per run under the data dir."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter

from . import settings


def new_run_dir(base: Path | None = None) -> Path:
    base = base or settings.RUNS_DIR
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = base / f"sim_{stamp}"
    n = 1
    while run.exists():
        n += 1
        run = base / f"sim_{stamp}_{n}"
    (run / "audio").mkdir(parents=True)
    return run


def setup_logging(run_dir: Path | None = None, level: int = logging.INFO) -> Path:
    """Send all logging to <run>/simulation.jsonl (and nothing to the terminal)."""
    run = run_dir or new_run_dir()
    (run / "audio").mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    fh = logging.FileHandler(run / "simulation.jsonl")
    fh.setFormatter(JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root.addHandler(fh)
    for noisy in ("httpx", "httpcore", "google_genai", "google.genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.info("logging started", extra={"event": "system.init", "run_dir": str(run)})
    return run
