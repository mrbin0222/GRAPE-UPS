from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import artifact_dir, dataset_dir, ensure_directories, write_json


ABSOLUTE_FEATURES = [
    "pre_voltage_v",
    "early_voltage_v",
    "voltage_sag_v",
    "dynamic_resistance_mohm",
    "discharge_slope_v_per_s",
    "minimum_voltage_v",
    "recovery_30s_v",
    "recovery_ratio",
    "pre_temperature_c",
    "temperature_rise_c",
    "mean_rssi_dbm",
    "missing_rate",
]

GROUP_RELATIVE_BASES = [
    "pre_voltage_v",
    "voltage_sag_v",
    "dynamic_resistance_mohm",
    "discharge_slope_v_per_s",
    "minimum_voltage_v",
    "recovery_30s_v",
    "recovery_ratio",
    "temperature_rise_c",
]


def _target_grid(sequence_length: int, max_time_s: float) -> np.ndarray:
    n_pre = max(16, int(round(sequence_length * 0.25)))
    n_mid = max(16, int(round(sequence_length * 0.25)))
    n_recovery = sequence_length - n_pre - n_mid
    if n_recovery < 16:
        n_recovery = 16
        n_mid = sequence_length - n_pre - n_recovery
    pre = np.linspace(-5.0, 5.0, n_pre, endpoint=True)
    mid = np.linspace(5.5, 60.0, n_mid, endpoint=True)
    recovery = np.linspace(61.0, max_time_s, n_recovery, endpoint=True)
    return np.concatenate([pre, mid, recovery]).astype(np.float32)


def _interp(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return np.full_like(target, np.nan, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    if unique_x.size == 1:
        return np.full_like(target, unique_y[0], dtype=float)
    return np.interp(target, unique_x, unique_y).astype(float)


def _mean_in_window(values: np.ndarray, grid: np.ndarray, start: float, end: float) -> float:
    mask = (grid >= start) & (grid <= end) & np.isfinite(values)
    if not mask.any():
        return float("nan")
    return float(np.mean(values[mask]))


def _feature_row(
    voltage: np.ndarray,
    temperature: np.ndarray,
    current: np.ndarray,
    grid: np.ndarray,
    duration_s: float,
    mean_rssi: float,
    missing_rate: float,
) -> dict[str, float]:
    pre_v = _mean_in_window(voltage, grid, -5, -1)
    early_v = _mean_in_window(voltage, grid, 0, 3)
    pre_i = _mean_in_window(current, grid, -5, -1)
    early_i = _mean_in_window(current, grid, 0, 3)
    sag = pre_v - early_v
    delta_i = abs(pre_i - early_i)
    dynamic_r = 1000.0 * sag / max(delta_i, 1e-6)
    discharge_mask = (grid >= 0) & (grid <= duration_s) & np.isfinite(voltage)
    minimum_voltage = (
        float(np.min(voltage[discharge_mask])) if discharge_mask.any() else float("nan")
    )
    slope_mask = (
        (grid >= 10)
        & (grid <= min(duration_s - 5, 60))
        & np.isfinite(voltage)
    )
    if slope_mask.sum() >= 3:
        discharge_slope = float(np.polyfit(grid[slope_mask], voltage[slope_mask], 1)[0])
    else:
        discharge_slope = float("nan")
    recovery_30 = _mean_in_window(
        voltage, grid, duration_s + 20, duration_s + 40
    )
    recovery_gain = recovery_30 - minimum_voltage
    recovery_ratio = recovery_gain / max(sag, 1e-6)
    pre_temp = _mean_in_window(temperature, grid, -5, -1)
    temp_mask = (
        (grid >= 0)
        & (grid <= min(float(grid.max()), duration_s + 60))
        & np.isfinite(temperature)
    )
    max_temp = float(np.max(temperature[temp_mask])) if temp_mask.any() else pre_temp
    return {
        "pre_voltage_v": pre_v,
        "early_voltage_v": early_v,
        "voltage_sag_v": sag,
        "dynamic_resistance_mohm": dynamic_r,
        "discharge_slope_v_per_s": discharge_slope,
        "minimum_voltage_v": minimum_voltage,
        "recovery_30s_v": recovery_30,
        "recovery_ratio": recovery_ratio,
        "pre_temperature_c": pre_temp,
        "temperature_rise_c": max_temp - pre_temp,
        "mean_rssi_dbm": mean_rssi,
        "missing_rate": missing_rate,
        "load_current_a": abs(early_i),
        "discharge_duration_s": duration_s,
    }


def _add_group_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    group_cols = ["string_id", "event_id"]
    for column in GROUP_RELATIVE_BASES:
        med = frame.groupby(group_cols)[column].transform("median")
        abs_dev = (frame[column] - med).abs()
        mad = abs_dev.groupby([frame[c] for c in group_cols]).transform("median")
        robust_scale = (1.4826 * mad).clip(lower=1e-6)
        frame[f"{column}_relz"] = (frame[column] - med) / robust_scale
        frame[f"{column}_rank_pct"] = frame.groupby(group_cols)[column].rank(pct=True)
    return frame


def _add_drift_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["string_id", "battery_id", "event_index"]).copy()
    for column, output in [
        ("dynamic_resistance_mohm", "resistance_drift_mohm"),
        ("pre_voltage_v", "pre_voltage_drift_v"),
        ("temperature_rise_c", "temperature_rise_drift_c"),
    ]:
        previous = frame.groupby(["string_id", "battery_id"])[column].transform(
            lambda values: values.shift(1).rolling(3, min_periods=1).median()
        )
        frame[output] = (frame[column] - previous).fillna(0.0)
    return frame.sort_values("_row_id").reset_index(drop=True)


def _split_name(
    string_id: str,
    event_index: int,
    n_events: int,
    train_fraction: float = 0.60,
    split_protocol: str = "temporal_plus_external_v1",
    train_string_id: str = "SIM-S1",
    validation_string_id: str = "SIM-S2",
) -> str:
    if string_id == "SIM-S3":
        return "test_external"
    train_end = max(2, int(round(n_events * train_fraction)))
    if split_protocol == "cross_string_validation_v1":
        if string_id == validation_string_id:
            return "validation"
        if string_id == train_string_id:
            return "train" if event_index <= train_end else "test_seen"
        raise ValueError(
            f"Unexpected development string {string_id!r} for "
            f"{split_protocol!r}."
        )
    validation_end = max(train_end + 1, int(round(n_events * 0.78)))
    if event_index <= train_end:
        return "train"
    if event_index <= validation_end:
        return "validation"
    return "test_seen"


def build_features(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    ensure_directories(config)
    data_root = dataset_dir(config)
    artifact_root = artifact_dir(config)
    feature_path = artifact_root / "event_features.csv"
    sequence_path = artifact_root / "event_sequences.npz"
    manifest_path = artifact_root / "feature_manifest.json"
    if feature_path.exists() and sequence_path.exists() and manifest_path.exists() and not force:
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    battery_usecols = [
        "string_id",
        "event_id",
        "event_index",
        "battery_id",
        "sensor_id",
        "elapsed_s",
        "voltage_v",
        "temperature_c",
        "rssi_dbm",
    ]
    current_usecols = [
        "string_id",
        "event_id",
        "event_index",
        "elapsed_s",
        "current_a",
    ]
    battery = pd.read_csv(
        data_root / "events" / "battery_event_raw.csv.gz",
        usecols=battery_usecols,
        dtype={
            "string_id": "category",
            "event_id": "category",
            "battery_id": "category",
            "sensor_id": "category",
            "event_index": "int16",
            "elapsed_s": "float32",
            "voltage_v": "float32",
            "temperature_c": "float32",
            "rssi_dbm": "float32",
        },
    )
    current = pd.read_csv(
        data_root / "events" / "current_event_raw.csv.gz",
        usecols=current_usecols,
        dtype={
            "string_id": "category",
            "event_id": "category",
            "event_index": "int16",
            "elapsed_s": "float32",
            "current_a": "float32",
        },
    )
    metadata = pd.read_csv(data_root / "events" / "event_metadata.csv")
    synthetic_truth_path = data_root / "events" / "synthetic_ground_truth.csv"
    field_labels_path = data_root / "events" / "labels.csv"
    if synthetic_truth_path.exists():
        truth = pd.read_csv(synthetic_truth_path)
    elif field_labels_path.exists():
        truth = pd.read_csv(field_labels_path)
    else:
        truth = (
            battery[
                ["string_id", "event_id", "event_index", "battery_id", "sensor_id"]
            ]
            .drop_duplicates()
            .copy()
        )
        truth["active_fault_type"] = "unlabeled"
        truth["fault_severity"] = np.nan
        truth["label_anomaly"] = -1
        truth["label_battery_weak"] = -1
        truth["label_sensor_fault"] = -1
        truth["true_resistance_mohm"] = np.nan
        truth["true_capacity_ah"] = np.nan
        truth["sensor_voltage_offset_v"] = np.nan
        truth["recovery_tau_fast_s"] = np.nan
        truth["recovery_tau_slow_s"] = np.nan
        truth["recovery_weight_fast"] = np.nan
        truth["thermal_extra_c"] = np.nan
    with (data_root / "manifest.json").open("r", encoding="utf-8") as handle:
        data_manifest = json.load(handle)

    max_time_s = float(config["event_grid"]["recovery_end_s"])
    target_grid = _target_grid(int(config["feature_sequence_length"]), max_time_s)
    expected_points = int(data_manifest["event_grid_points"])
    event_meta = metadata.set_index(["string_id", "event_id"])

    current_by_event: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (string_id, event_id), group in current.groupby(
        ["string_id", "event_id"], observed=True, sort=False
    ):
        current_by_event[(str(string_id), str(event_id))] = (
            group["elapsed_s"].to_numpy(float),
            group["current_a"].to_numpy(float),
        )

    feature_rows: list[dict[str, Any]] = []
    sequence_rows: list[np.ndarray] = []
    row_keys: list[str] = []
    quality_rows: list[dict[str, Any]] = []
    row_id = 0

    for (string_id_value, event_id_value), event_frame in battery.groupby(
        ["string_id", "event_id"], observed=True, sort=False
    ):
        string_id = str(string_id_value)
        event_id = str(event_id_value)
        current_x, current_y = current_by_event[(string_id, event_id)]
        current_sequence = _interp(current_x, current_y, target_grid)
        meta = event_meta.loc[(string_id, event_id)]
        duration_s = float(meta["discharge_duration_s"])

        temporary: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
        for battery_id_value, group in event_frame.groupby(
            "battery_id", observed=True, sort=True
        ):
            battery_id = str(battery_id_value)
            x = group["elapsed_s"].to_numpy(float)
            voltage = _interp(x, group["voltage_v"].to_numpy(float), target_grid)
            temperature = _interp(
                x, group["temperature_c"].to_numpy(float), target_grid
            )
            missing_rate = max(0.0, 1.0 - len(group) / expected_points)
            row = {
                "_row_id": row_id,
                "site_id": config["site_id"],
                "string_id": string_id,
                "event_id": event_id,
                "event_index": int(group["event_index"].iloc[0]),
                "battery_id": battery_id,
                "sensor_id": str(group["sensor_id"].iloc[0]),
                "ambient_temperature_c": float(meta["ambient_temperature_c"]),
                "ups_load_pct": float(meta["ups_load_pct"]),
                "load_stratum": str(meta["load_stratum"]),
                **_feature_row(
                    voltage,
                    temperature,
                    current_sequence,
                    target_grid,
                    duration_s,
                    float(group["rssi_dbm"].mean()),
                    missing_rate,
                ),
            }
            temporary.append((row, voltage, temperature))
            row_id += 1

        voltage_matrix = np.stack([item[1] for item in temporary])
        temperature_matrix = np.stack([item[2] for item in temporary])
        voltage_median = np.nanmedian(voltage_matrix, axis=0)
        voltage_mad = 1.4826 * np.nanmedian(
            np.abs(voltage_matrix - voltage_median), axis=0
        )
        voltage_mad = np.maximum(voltage_mad, 0.003)
        temperature_median = np.nanmedian(temperature_matrix, axis=0)
        temperature_mad = 1.4826 * np.nanmedian(
            np.abs(temperature_matrix - temperature_median), axis=0
        )
        temperature_mad = np.maximum(temperature_mad, 0.03)

        for row, voltage, temperature in temporary:
            relative_voltage = (voltage - voltage_median) / voltage_mad
            relative_temperature = (temperature - temperature_median) / temperature_mad
            channels = np.stack(
                [
                    voltage,
                    temperature,
                    current_sequence,
                    relative_voltage,
                    relative_temperature,
                ],
                axis=0,
            ).astype(np.float32)
            feature_rows.append(row)
            sequence_rows.append(channels)
            row_keys.append(
                f"{row['string_id']}|{row['event_id']}|{row['battery_id']}"
            )

        node_count = event_frame["battery_id"].nunique()
        actual_rows = len(event_frame)
        battery_completeness = actual_rows / max(
            1, expected_points * int(config["n_batteries"])
        )
        current_rows = len(current_x)
        current_completeness = current_rows / max(1, expected_points)
        if (
            node_count == int(config["n_batteries"])
            and battery_completeness >= 0.95
            and current_completeness >= 0.98
        ):
            quality_grade = "A"
        elif (
            node_count >= 38
            and battery_completeness >= 0.90
            and current_completeness >= 0.90
        ):
            quality_grade = "B"
        else:
            quality_grade = "C"
        quality_rows.append(
            {
                "string_id": string_id,
                "event_id": event_id,
                "event_index": int(meta["event_index"]),
                "node_count": int(node_count),
                "battery_rows": int(actual_rows),
                "current_rows": int(current_rows),
                "battery_completeness": battery_completeness,
                "current_completeness": current_completeness,
                "quality_grade": quality_grade,
            }
        )

    features = pd.DataFrame(feature_rows)
    features = _add_group_relative_features(features)
    features = _add_drift_features(features)
    truth_merge_columns = [
        "string_id",
        "event_id",
        "battery_id",
        "active_fault_type",
        "fault_severity",
        "label_anomaly",
        "label_battery_weak",
        "label_sensor_fault",
        "true_resistance_mohm",
        "true_capacity_ah",
        "sensor_voltage_offset_v",
        "recovery_tau_fast_s",
        "recovery_tau_slow_s",
        "recovery_weight_fast",
        "thermal_extra_c",
    ]
    features = features.merge(
        truth[truth_merge_columns],
        on=["string_id", "event_id", "battery_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    features = features.sort_values("_row_id").reset_index(drop=True)
    n_events = int(config["n_events_per_string"])
    features["split"] = [
        _split_name(
            string_id,
            int(event_index),
            n_events,
            float(config.get("train_fraction", 0.60)),
            str(config.get("split_protocol", "temporal_plus_external_v1")),
            str(config.get("train_string_id", "SIM-S1")),
            str(config.get("validation_string_id", "SIM-S2")),
        )
        for string_id, event_index in zip(features["string_id"], features["event_index"])
    ]

    sequence_array = np.stack(sequence_rows).astype(np.float32)
    if len(features) != len(sequence_array):
        raise RuntimeError("Feature and sequence row counts diverged.")
    features.to_csv(feature_path, index=False)
    np.savez_compressed(
        sequence_path,
        sequences=sequence_array,
        target_grid_s=target_grid,
        row_keys=np.asarray(row_keys, dtype="U64"),
        channel_names=np.asarray(
            ["voltage_v", "temperature_c", "current_a", "relative_voltage", "relative_temperature"],
            dtype="U32",
        ),
    )
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(artifact_root / "quality_summary.csv", index=False)

    relative_columns = [
        column
        for column in features.columns
        if column.endswith("_relz") or column.endswith("_rank_pct")
    ]
    drift_columns = [
        "resistance_drift_mohm",
        "pre_voltage_drift_v",
        "temperature_rise_drift_c",
    ]
    model_feature_columns = (
        ABSOLUTE_FEATURES
        + ["load_current_a", "discharge_duration_s", "ambient_temperature_c", "ups_load_pct"]
        + relative_columns
        + drift_columns
    )
    feature_manifest = {
        "dataset_name": config["dataset_name"],
        "simulation_only": bool(data_manifest.get("simulation_only", False)),
        "n_samples": int(len(features)),
        "n_sequence_channels": int(sequence_array.shape[1]),
        "sequence_length": int(sequence_array.shape[2]),
        "absolute_feature_columns": ABSOLUTE_FEATURES,
        "relative_feature_columns": relative_columns,
        "drift_feature_columns": drift_columns,
        "model_feature_columns": model_feature_columns,
        "split_counts": {
            str(key): int(value)
            for key, value in features["split"].value_counts().to_dict().items()
        },
        "label_counts": {
            "all_anomaly": int((features["label_anomaly"] == 1).sum()),
            "battery_weak": int((features["label_battery_weak"] == 1).sum()),
            "sensor_fault": int((features["label_sensor_fault"] == 1).sum()),
        },
        "quality_grade_counts": {
            str(key): int(value)
            for key, value in quality["quality_grade"].value_counts().to_dict().items()
        },
    }
    write_json(manifest_path, feature_manifest)
    return feature_manifest
