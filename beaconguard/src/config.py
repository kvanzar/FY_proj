"""NFR-8: single YAML config, loaded once and passed explicitly through the
pipeline rather than read ad hoc from disk in each module."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    set_global_seed(cfg["seed"])
    _resolve_relative_paths(cfg, path)
    return cfg


def _resolve_relative_paths(cfg: dict[str, Any], config_path: Path) -> None:
    """Paths written in the YAML (data.raw_paths, paths.*) are meant to be
    relative to the project root — e.g. "data/iot23_combined_new.csv" —
    not to whatever directory the calling process happens to have as its
    cwd. Left unresolved, the exact same config works from
    `python run_pipeline.py` (cwd = project root) but breaks with a
    FileNotFoundError from a notebook in notebooks/ (cwd = notebooks/).
    Resolve everything here, once, against the config file's own location
    instead — this assumes the config lives one directory level under the
    project root (config/default.yaml), matching DEFAULT_CONFIG_PATH and
    this project's layout; a custom --config path should follow the same
    convention.
    """
    project_root = config_path.resolve().parent.parent

    def _resolve(p: str) -> str:
        pp = Path(p).expanduser()
        if not pp.is_absolute():
            pp = (project_root / pp).resolve()
        return str(pp)

    if "data" in cfg and cfg["data"].get("raw_paths"):
        cfg["data"]["raw_paths"] = [_resolve(p) for p in cfg["data"]["raw_paths"]]
    if "paths" in cfg:
        cfg["paths"] = {k: _resolve(v) for k, v in cfg["paths"].items()}


def set_global_seed(seed: int) -> None:
    """NFR-5: fixed seeds for reproducibility across numpy, python's random,
    and (if importable) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_logger(name: str, cfg: dict[str, Any] | None = None) -> logging.Logger:
    """NFR-9: structured logging at every pipeline stage."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)

    if cfg is not None:
        logs_dir = Path(cfg["paths"]["logs_dir"])
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / "pipeline.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)
    return logger
