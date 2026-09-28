from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    artifact_dir,
    project_root,
    robust_positive_z,
    set_seed,
    write_json,
)


FUSION_COMPONENT_ORDER = [
    "relative_temporal",
    "relative_physics",
    "group_relative",
    "drift",
    "absolute_temporal",
    "absolute_physics",
]
FUSION_COMPONENT_KEYS = {
    "relative_temporal": "sequence_error_z",
    "relative_physics": "physics_error_z",
    "group_relative": "group_score_z",
    "drift": "drift_score_z",
    "absolute_temporal": "abs_sequence_error_z",
    "absolute_physics": "abs_physics_error_z",
}
DEFAULT_FUSION_WEIGHTS = {
    "relative_temporal": 0.35,
    "relative_physics": 0.30,
    "group_relative": 0.25,
    "drift": 0.10,
    "absolute_temporal": 0.0,
    "absolute_physics": 0.0,
}
GROUP_SCORE_CONTRACT = "directed_top3_mean_v1"
GROUP_DIRECTIONAL_FEATURES = {
    # A weak/offset cell is expected to be low in voltage and recovery.
    "pre_voltage_v_relz": -1.0,
    "voltage_sag_v_relz": 1.0,
    "dynamic_resistance_mohm_relz": 1.0,
    "discharge_slope_v_per_s_relz": -1.0,
    "minimum_voltage_v_relz": -1.0,
    "recovery_30s_v_relz": -1.0,
    "recovery_ratio_relz": -1.0,
    "temperature_rise_c_relz": 1.0,
}


@dataclass
class PreparedData:
    features: pd.DataFrame
    sequences: np.ndarray
    model_features: list[str]
    absolute_features: list[str]
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_seen_mask: np.ndarray
    test_external_mask: np.ndarray
    x_all: np.ndarray
    x_abs_all: np.ndarray
    seq_all: np.ndarray
    seq_abs_all: np.ndarray
    imputer: SimpleImputer
    scaler: StandardScaler
    abs_imputer: SimpleImputer
    abs_scaler: StandardScaler
    seq_mean: np.ndarray
    seq_std: np.ndarray
    seq_abs_mean: np.ndarray
    seq_abs_std: np.ndarray




class HybridAutoencoder(nn.Module):
    def __init__(
        self,
        sequence_channels: int,
        sequence_length: int,
        physics_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        kernels = [3, 7, 15]
        self.multiscale = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(sequence_channels, 8, kernel, padding=kernel // 2),
                    nn.ReLU(),
                )
                for kernel in kernels
            ]
        )
        self.seq_down = nn.Sequential(
            nn.Conv1d(24, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        reduced_length = sequence_length // 4
        self.reduced_length = reduced_length
        self.seq_latent = nn.Linear(8 * reduced_length, latent_dim)
        self.seq_expand = nn.Linear(latent_dim, 8 * reduced_length)
        self.seq_up = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(24, sequence_channels, kernel_size=3, padding=1),
        )
        physics_hidden = max(24, min(96, physics_dim * 2))
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_dim, physics_hidden),
            nn.ReLU(),
            nn.Linear(physics_hidden, max(6, latent_dim // 2)),
        )
        self.physics_decoder = nn.Sequential(
            nn.Linear(max(6, latent_dim // 2), physics_hidden),
            nn.ReLU(),
            nn.Linear(physics_hidden, physics_dim),
        )

    def forward(
        self, sequence: torch.Tensor, physics: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        views = [branch(sequence) for branch in self.multiscale]
        merged = torch.cat(views, dim=1)
        down = self.seq_down(merged)
        latent = self.seq_latent(down.flatten(1))
        expanded = self.seq_expand(latent).reshape(
            sequence.shape[0], 8, self.reduced_length
        )
        reconstructed_sequence = self.seq_up(expanded)
        if reconstructed_sequence.shape[-1] != self.sequence_length:
            reconstructed_sequence = nn.functional.interpolate(
                reconstructed_sequence,
                size=self.sequence_length,
                mode="linear",
                align_corners=False,
            )
        physics_latent = self.physics_encoder(physics)
        reconstructed_physics = self.physics_decoder(physics_latent)
        return reconstructed_sequence, reconstructed_physics


def _prepare_data(config: dict[str, Any]) -> PreparedData:
    artifacts = artifact_dir(config)
    features = pd.read_csv(artifacts / "event_features.csv")
    sequence_bundle = np.load(artifacts / "event_sequences.npz")
    sequences = sequence_bundle["sequences"].astype(np.float32)
    with (artifacts / "feature_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    model_features = list(manifest["model_feature_columns"])
    absolute_features = list(manifest["absolute_feature_columns"]) + [
        "load_current_a",
        "discharge_duration_s",
        "ambient_temperature_c",
        "ups_load_pct",
    ]
    train_mask = (
        (features["split"].to_numpy() == "train")
        & (features["label_anomaly"].to_numpy() == 0)
    )
    validation_mask = features["split"].to_numpy() == "validation"
    test_seen_mask = features["split"].to_numpy() == "test_seen"
    test_external_mask = features["split"].to_numpy() == "test_external"

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = imputer.fit_transform(features.loc[train_mask, model_features])
    scaler.fit(x_train)
    x_all = scaler.transform(imputer.transform(features[model_features])).astype(
        np.float32
    )

    abs_imputer = SimpleImputer(strategy="median")
    abs_scaler = StandardScaler()
    x_abs_train = abs_imputer.fit_transform(
        features.loc[train_mask, absolute_features]
    )
    abs_scaler.fit(x_abs_train)
    x_abs_all = abs_scaler.transform(
        abs_imputer.transform(features[absolute_features])
    ).astype(np.float32)

    seq_mean = np.nanmean(sequences[train_mask], axis=(0, 2), keepdims=True)
    seq_std = np.nanstd(sequences[train_mask], axis=(0, 2), keepdims=True)
    seq_std = np.maximum(seq_std, 1e-4)
    seq_all = np.nan_to_num((sequences - seq_mean) / seq_std).astype(np.float32)

    seq_abs = sequences[:, :3, :]
    seq_abs_mean = np.nanmean(seq_abs[train_mask], axis=(0, 2), keepdims=True)
    seq_abs_std = np.nanstd(seq_abs[train_mask], axis=(0, 2), keepdims=True)
    seq_abs_std = np.maximum(seq_abs_std, 1e-4)
    seq_abs_all = np.nan_to_num(
        (seq_abs - seq_abs_mean) / seq_abs_std
    ).astype(np.float32)

    return PreparedData(
        features=features,
        sequences=sequences,
        model_features=model_features,
        absolute_features=absolute_features,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_seen_mask=test_seen_mask,
        test_external_mask=test_external_mask,
        x_all=x_all,
        x_abs_all=x_abs_all,
        seq_all=seq_all,
        seq_abs_all=seq_abs_all,
        imputer=imputer,
        scaler=scaler,
        abs_imputer=abs_imputer,
        abs_scaler=abs_scaler,
        seq_mean=seq_mean,
        seq_std=seq_std,
        seq_abs_mean=seq_abs_mean,
        seq_abs_std=seq_abs_std,
    )




def _train_hybrid_autoencoder(
    sequence_train: np.ndarray,
    physics_train: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[HybridAutoencoder, list[float]]:
    model_cfg = config["model"]
    model = HybridAutoencoder(
        sequence_channels=sequence_train.shape[1],
        sequence_length=sequence_train.shape[2],
        physics_dim=physics_train.shape[1],
        latent_dim=int(model_cfg["latent_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg["learning_rate"]))
    dataset = TensorDataset(
        torch.from_numpy(sequence_train), torch.from_numpy(physics_train)
    )
    loader = DataLoader(
        dataset,
        batch_size=int(model_cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
    )
    history: list[float] = []
    model.train()
    for _ in range(int(model_cfg["epochs"])):
        epoch_losses = []
        for sequence, physics in loader:
            sequence = sequence.to(device)
            physics = physics.to(device)
            optimizer.zero_grad()
            reconstructed_sequence, reconstructed_physics = model(sequence, physics)
            sequence_loss = nn.functional.mse_loss(
                reconstructed_sequence, sequence
            )
            physics_loss = nn.functional.mse_loss(reconstructed_physics, physics)
            loss = sequence_loss + 0.55 * physics_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(epoch_losses)))
    return model, history




def _hybrid_errors(
    model: HybridAutoencoder,
    sequences: np.ndarray,
    physics: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    sequence_scores: list[np.ndarray] = []
    physics_scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            sequence = torch.from_numpy(sequences[start : start + batch_size]).to(
                device
            )
            physical = torch.from_numpy(physics[start : start + batch_size]).to(
                device
            )
            reconstructed_sequence, reconstructed_physics = model(sequence, physical)
            sequence_error = torch.mean(
                (reconstructed_sequence - sequence) ** 2, dim=(1, 2)
            )
            physics_error = torch.mean(
                (reconstructed_physics - physical) ** 2, dim=1
            )
            sequence_scores.append(sequence_error.cpu().numpy())
            physics_scores.append(physics_error.cpu().numpy())
    return np.concatenate(sequence_scores), np.concatenate(physics_scores)


def _threshold_from_validation(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float(np.quantile(scores, 0.95))
    candidates = np.unique(np.quantile(scores, np.linspace(0.50, 0.995, 250)))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        prediction = (scores >= threshold).astype(int)
        value = f1_score(y_true, prediction, zero_division=0)
        if value > best_f1:
            best_f1 = value
            best_threshold = float(threshold)
    return best_threshold


def _binary_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = (scores >= threshold).astype(int)
    result = {
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "threshold": float(threshold),
        "average_precision": float("nan"),
        "roc_auc": float("nan"),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "false_alarms_per_1000_normal": float(
            1000
            * np.sum((prediction == 1) & (y_true == 0))
            / max(1, np.sum(y_true == 0))
        ),
    }
    if len(np.unique(y_true)) == 2:
        result["average_precision"] = float(average_precision_score(y_true, scores))
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
    return result


def _ranking_metrics(
    frame: pd.DataFrame, score_column: str, label_column: str
) -> dict[str, float]:
    """Return event-wise ranking quality for events containing anomalies.

    ``topK_normalized_capture`` is the number of anomalous units in the first
    K ranks divided by ``min(K, n_anomalous)``. It measures how fully the
    available maintenance shortlist is used and is deliberately not named
    recall@K, whose denominator would be ``n_anomalous``.
    """

    capture3: list[float] = []
    capture5: list[float] = []
    ndcg_values: list[float] = []
    for _, group in frame.groupby(["string_id", "event_id"]):
        labels = group[label_column].to_numpy(int)
        if labels.sum() == 0:
            continue
        scores = group[score_column].to_numpy(float)
        order = np.argsort(-scores)
        capture3.append(float(labels[order[:3]].sum() / min(3, labels.sum())))
        capture5.append(float(labels[order[:5]].sum() / min(5, labels.sum())))
        ndcg_values.append(float(ndcg_score(labels[None, :], scores[None, :])))
    return {
        "top3_normalized_capture": (
            float(np.mean(capture3)) if capture3 else float("nan")
        ),
        "top5_normalized_capture": (
            float(np.mean(capture5)) if capture5 else float("nan")
        ),
        "ndcg": float(np.mean(ndcg_values)) if ndcg_values else float("nan"),
    }


def _score_components(
    data: PreparedData,
    hybrid_model: HybridAutoencoder,
    abs_hybrid_model: HybridAutoencoder,
    device: torch.device,
) -> dict[str, np.ndarray]:
    sequence_error, physics_error = _hybrid_errors(
        hybrid_model, data.seq_all, data.x_all, device
    )
    abs_sequence_error, abs_physics_error = _hybrid_errors(
        abs_hybrid_model, data.seq_abs_all, data.x_abs_all, device
    )
    group_score = _directed_group_score(data.features)
    relz_columns = [
        column for column in data.features.columns if column.endswith("_relz")
    ]
    group_undirected_score = np.nanmax(
        np.abs(data.features[relz_columns].to_numpy(float)), axis=1
    )
    train_reference = data.features.loc[data.train_mask]
    resistance_scale = max(
        1e-6,
        float(train_reference["resistance_drift_mohm"].abs().median()) * 1.4826,
    )
    voltage_scale = max(
        1e-6,
        float(train_reference["pre_voltage_drift_v"].abs().median()) * 1.4826,
    )
    temp_scale = max(
        1e-6,
        float(train_reference["temperature_rise_drift_c"].abs().median()) * 1.4826,
    )
    drift_score = np.maximum.reduce(
        [
            np.maximum(
                0.0,
                data.features["resistance_drift_mohm"].to_numpy(float)
                / resistance_scale,
            ),
            np.maximum(
                0.0,
                -data.features["pre_voltage_drift_v"].to_numpy(float)
                / voltage_scale,
            ),
            np.maximum(
                0.0,
                data.features["temperature_rise_drift_c"].to_numpy(float)
                / temp_scale,
            ),
        ]
    )
    train = data.train_mask
    components = {
        "sequence_error": sequence_error,
        "physics_error": physics_error,
        "group_score": group_score,
        "group_undirected_score": group_undirected_score,
        "drift_score": drift_score,
        "abs_sequence_error": abs_sequence_error,
        "abs_physics_error": abs_physics_error,
    }
    for name in list(components):
        components[f"{name}_z"] = robust_positive_z(
            components[name], components[name][train]
        )
    return components


def _directed_group_score(frame: pd.DataFrame, top_k: int = 3) -> np.ndarray:
    """Aggregate physically directed group deviations without max-noise inflation."""
    available = [
        column for column in GROUP_DIRECTIONAL_FEATURES if column in frame.columns
    ]
    if not available:
        relz_columns = [column for column in frame.columns if column.endswith("_relz")]
        if not relz_columns:
            raise ValueError("No group-relative z-score columns are available.")
        return np.nanmax(
            np.abs(frame[relz_columns].to_numpy(float)), axis=1
        )
    directed = np.column_stack(
        [
            GROUP_DIRECTIONAL_FEATURES[column]
            * frame[column].to_numpy(float)
            for column in available
        ]
    )
    directed = np.nan_to_num(directed, nan=0.0, posinf=0.0, neginf=0.0)
    directed = np.maximum(0.0, directed)
    k = max(1, min(int(top_k), directed.shape[1]))
    return np.sort(directed, axis=1)[:, -k:].mean(axis=1)


def _mean_sd_group_scores(
    frame: pd.DataFrame,
    top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack-mean comparator for the robust median/MAD peer reference."""
    group_columns = ["string_id", "event_id"]
    directed_columns = []
    for relz_column, direction in GROUP_DIRECTIONAL_FEATURES.items():
        base_column = relz_column.removesuffix("_relz")
        if base_column not in frame.columns:
            continue
        mean = frame.groupby(group_columns)[base_column].transform("mean")
        std = frame.groupby(group_columns)[base_column].transform("std").clip(
            lower=1e-6
        )
        z_score = (frame[base_column] - mean) / std
        directed_columns.append(direction * z_score.to_numpy(float))
    if not directed_columns:
        raise ValueError("No physical columns available for mean/SD peer scores.")
    directed = np.nan_to_num(
        np.column_stack(directed_columns),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    undirected = np.max(np.abs(directed), axis=1)
    positive = np.maximum(0.0, directed)
    k = max(1, min(int(top_k), positive.shape[1]))
    directed_score = np.sort(positive, axis=1)[:, -k:].mean(axis=1)
    return undirected, directed_score


def _normalize_fusion_weights(
    weights: dict[str, float] | list[float] | np.ndarray,
) -> np.ndarray:
    if isinstance(weights, dict):
        unknown = set(weights) - set(FUSION_COMPONENT_ORDER)
        if unknown:
            raise ValueError(f"Unknown GRAPE fusion components: {sorted(unknown)}")
        values = np.asarray(
            [float(weights.get(name, 0.0)) for name in FUSION_COMPONENT_ORDER],
            dtype=float,
        )
    else:
        values = np.asarray(weights, dtype=float)
    if values.shape != (len(FUSION_COMPONENT_ORDER),):
        raise ValueError(
            f"Expected {len(FUSION_COMPONENT_ORDER)} fusion weights, "
            f"got shape {values.shape}."
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Fusion weights must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("At least one fusion weight must be positive.")
    return values / total


def _load_fusion_weights(
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    model_cfg = config.get("model", {})
    if "fusion_weights" in model_cfg:
        weights = _normalize_fusion_weights(model_cfg["fusion_weights"])
        return weights, {
            "source": "config_inline",
            "selection_scope": "declared_by_config",
            "used_splits": [],
        }
    weights_file = model_cfg.get("fusion_weights_file")
    if weights_file:
        path = Path(weights_file)
        if not path.is_absolute():
            path = project_root() / path
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("component_order") != FUSION_COMPONENT_ORDER:
            raise ValueError(
                f"Fusion component order mismatch in {path}: "
                f"{payload.get('component_order')}"
            )
        weights = _normalize_fusion_weights(payload["weights"])
        provenance = dict(payload)
        provenance["source"] = "validation_weights_file"
        provenance["path"] = str(path.resolve())
        return weights, provenance
    return _normalize_fusion_weights(DEFAULT_FUSION_WEIGHTS), {
        "source": "fixed_default",
        "selection_scope": "fixed_prior",
        "used_splits": [],
    }


def _fusion_matrix(components: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [components[FUSION_COMPONENT_KEYS[name]] for name in FUSION_COMPONENT_ORDER]
    )


def _renormalized_without(weights: np.ndarray, removed: set[str]) -> np.ndarray:
    retained = weights.copy()
    for name in removed:
        retained[FUSION_COMPONENT_ORDER.index(name)] = 0.0
    if retained.sum() <= 0:
        retained = np.asarray(
            [
                1.0 if name not in removed else 0.0
                for name in FUSION_COMPONENT_ORDER
            ],
            dtype=float,
        )
    return _normalize_fusion_weights(retained)


def _make_fused_scores(
    components: dict[str, np.ndarray],
    fusion_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    matrix = _fusion_matrix(components)
    group_only = np.zeros(len(FUSION_COMPONENT_ORDER), dtype=float)
    group_only[FUSION_COMPONENT_ORDER.index("group_relative")] = 1.0
    absolute_only = _normalize_fusion_weights(
        {
            "absolute_temporal": 0.60,
            "absolute_physics": 0.40,
        }
    )
    return {
        "GRAPE-UPS": matrix @ fusion_weights,
        "GRAPE-no-temporal": matrix
        @ _renormalized_without(
            fusion_weights, {"relative_temporal", "absolute_temporal"}
        ),
        "GRAPE-no-physics": matrix
        @ _renormalized_without(
            fusion_weights, {"relative_physics", "absolute_physics"}
        ),
        "GRAPE-no-group": matrix
        @ _renormalized_without(fusion_weights, {"group_relative"}),
        "GRAPE-no-drift": matrix
        @ _renormalized_without(fusion_weights, {"drift"}),
        "GRAPE-absolute-only": matrix @ absolute_only,
        "Group-only": matrix @ group_only,
    }




