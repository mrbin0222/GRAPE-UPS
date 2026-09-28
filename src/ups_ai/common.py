from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(path.resolve())
    return config


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dataset_dir(config: dict[str, Any]) -> Path:
    return project_root() / config.get("data_path", "data/" + config["dataset_name"])


def artifact_dir(config: dict[str, Any]) -> Path:
    return Path(os.environ.get("GRAPE_CACHE_DIR", project_root() / ".cache")) / config["dataset_name"]


def ensure_directories(config: dict[str, Any]) -> None:
    if not dataset_dir(config).is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir(config)}")
    artifact_dir(config).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = max(1.4826 * mad, 1e-8)
    return center, scale


def robust_positive_z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    center, scale = robust_center_scale(reference)
    return np.maximum(0.0, (np.asarray(values, dtype=float) - center) / scale)
