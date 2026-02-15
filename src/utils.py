# src/utils.py

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_logger(log_file: str, name: str = "train") -> logging.Logger:
    """
    Logger that logs to BOTH console and a file.
    Safe to call multiple times (won't duplicate handlers).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create log dir
    log_dir = os.path.dirname(log_file)
    if log_dir:
        ensure_dir(log_dir)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def write_json(path: str, obj: Dict[str, Any]) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        ensure_dir(dirpath)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(path: str, text: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        ensure_dir(dirpath)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_latest_run(artifacts_root: str, run_id: str) -> None:
    ensure_dir(artifacts_root)
    latest_path = os.path.join(artifacts_root, "latest.txt")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(run_id)
