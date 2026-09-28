#!/usr/bin/env python3
"""Generate the primary synthetic UPS dataset and export its labeled records."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

TZ_CHINA = timezone(timedelta(hours=8))
UTC = timezone.utc

# ── 物理参数 ──────────────────────────────────────────────
N_BATTERIES = 40
SITE_ID = "SITE_SIM_01"
STRING_ID = "SIM-S1"
FIRMWARE_VERSION = "sim-1.2.0"
CALIBRATION_VERSION = "SIM-CAL-202607"
GENERATOR_VERSION = "v2_p1_full_contract_v3"
P1_TRAIN_FRACTION = 0.60

FAULT_PLAN_BASE: dict[int, tuple[str, float]] = {
    # battery_no: (fault_type, onset_fraction)
    7:  ("low_capacity",   0.30),
    12: ("sensor_offset",  0.38),
    19: ("thermal_drift",  0.42),
    31: ("slow_recovery",  0.34),
    38: ("high_resistance", 0.24),
}

# P1 扩展故障类型（medium/hard 额外添加）
FAULT_PLAN_EXTRA: dict[int, tuple[str, float]] = {
    5:  ("voltage_drift",        0.30),
    15: ("intermittent_fault",   0.35),
    25: ("accelerated_aging",    0.25),
    33: ("sulfation",            0.32),
}

# 难度等级配置
DIFFICULTY_CONFIGS = {
    "p0": {
        "onset_multiplier": 1.0,
        "severity_cap": 1.0,
        "extra_faults": False,
        "packet_loss_rate": 0.0,
        "upload_latency_sigma_s": 0.0,
    },
    "easy": {
        "onset_multiplier": 1.5,   # 故障出现更晚
        "severity_cap": 0.70,      # 最大严重度更低
        "extra_faults": False,     # 不添加额外故障
        "packet_loss_rate": 0.02,  # 2% 丢包
        "upload_latency_sigma_s": 0.3,
    },
    "medium": {
        "onset_multiplier": 1.0,
        "severity_cap": 1.0,
        "extra_faults": True,      # 添加 3 个额外故障
        "packet_loss_rate": 0.05,  # 5% 丢包
        "upload_latency_sigma_s": 0.5,
    },
    "hard": {
        "onset_multiplier": 0.65,  # 故障出现更早
        "severity_cap": 1.0,
        "extra_faults": True,      # 添加 4 个额外故障
        "packet_loss_rate": 0.10,  # 10% 丢包
        "upload_latency_sigma_s": 1.0,
    },
}


def get_fault_plan(difficulty: str = "medium", string_idx: int = 0) -> dict[int, tuple[str, float]]:
    """根据难度等级返回故障计划

    B5: 不同串使用不同故障电池编号（旋转偏移），确保跨串独立。
    B10: 对 P1 难度，故障 onset 晚于 60% train split，保证训练期无异常。
    """
    plan = dict(FAULT_PLAN_BASE)
    cfg = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["medium"])
    if cfg["extra_faults"]:
        plan.update(FAULT_PLAN_EXTRA)
        if difficulty == "hard":
            plan[28] = ("sulfation", 0.28)

    if string_idx > 0:
        offset = string_idx * 4  # 每串偏移4个位置
        rotated: dict[int, tuple[str, float]] = {}
        for bno, (ftype, ofrac) in plan.items():
            new_bno = ((bno - 1 + offset) % N_BATTERIES) + 1
            rotated[new_bno] = (ftype, ofrac)
        plan = rotated
    return plan

# 正常电池基线参数
BASE_FLOAT_V = 13.63
BASE_RESISTANCE_MOHM = 9.10
BASE_CAPACITY_AH = 100.0

# 放电事件参数 — 固定时长保证 grid 一致
LOAD_LEVELS = [
    (25.0, 22.0),
    (45.0, 38.0),
    (65.0, 52.0),
]
FIXED_DURATION_S = 120.0       # 固定放电时长
PRE_DURATION_S = 30.0
RECOVERY_DURATION_S = 120.0
SAMPLE_INTERVAL_S = 2.0

# 事件网格总点数（PRE 15 + DISCHARGE 61 + RECOVERY 60 = 136）
EVENT_GRID_POINTS = 136


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()



def severity_progression(event_index: int, onset_event: int, n_events: int,
                         severity_cap: float = 1.0) -> float:
    """故障严重度随事件递进：0 -> severity_cap"""
    if event_index < onset_event:
        return 0.0
    span = max(1, n_events - onset_event)
    return float(min(severity_cap, 0.18 + 0.82 * (event_index - onset_event) / span))


def p1_train_end_event(n_events: int) -> int:
    """Return the last event used by the pipeline's 60% training split."""
    return max(2, int(round(n_events * P1_TRAIN_FRACTION)))


def p1_min_fault_onset_event(n_events: int) -> int:
    """Keep the entire training split physically fault-free."""
    return p1_train_end_event(n_events) + 1


def get_fault(battery_no: int, n_events: int,
              fault_plan: dict[int, tuple[str, float]] | None = None,
              onset_multiplier: float = 1.0,
              min_onset: int = 0) -> tuple[str, int]:
    """返回 (fault_type, onset_event_index)，onset 使用实际 n_events

    B10: min_onset 参数确保故障不在训练期出现。
    """
    plan = fault_plan if fault_plan is not None else FAULT_PLAN_BASE
    if battery_no in plan:
        fault_type, onset_frac = plan[battery_no]
        onset_frac = min(0.95, onset_frac * onset_multiplier)
        onset_event = max(2, int(round(n_events * onset_frac)))
        # B10: 确保不早于 min_onset
        if min_onset > 0:
            onset_event = max(onset_event, min_onset)
        return fault_type, onset_event
    return "normal", n_events + 1


def battery_response(
    elapsed_s: float,
    current_a: float,
    duration_s: float,
    ambient_c: float,
    battery_no: int,
    fault_type: str,
    severity: float,
    rng: random.Random,
    event_index: int = 0,
    n_events: int = 30,
    string_idx: int = 0,
    load_current_a: float | None = None,
) -> tuple[float, float, dict[str, float]]:
    """计算单节电池在某一时刻的电压和温度

    P1 扩展：
    - 新增 voltage_drift, intermittent_fault, accelerated_aging, sulfation 故障类型
    - 恢复模型：双指数恢复（快速+慢速），温度依赖 tau
    - 容量模型：跨事件渐进退化
    - string_idx: 用于独立基线（不同串的相同 battery_no 有不同基线参数）
    - load_current_a: 本次放电负载幅值，用于固定放电切除状态
    """

    # 个体差异（固定种子可复现，string_idx 确保跨串独立）
    indiv_seed = battery_no * 7919 + string_idx * 100003
    indiv_rng = random.Random(indiv_seed)
    float_v = BASE_FLOAT_V + indiv_rng.gauss(0, 0.022)
    base_r = BASE_RESISTANCE_MOHM + indiv_rng.gauss(0, 0.48)
    base_r = max(7.7, min(10.5, base_r))
    capacity = BASE_CAPACITY_AH + indiv_rng.gauss(0, 2.8)
    capacity = max(92.0, min(106.0, capacity))
    temp_offset = indiv_rng.gauss(0, 0.18)

    # 跨事件渐进退化（所有电池都有微小退化，故障电池更明显）
    aging_frac = event_index / max(1, n_events)
    base_r += 0.3 * aging_frac  # 正常电池也有微小内阻增长
    capacity -= 1.5 * aging_frac  # 正常电池也有微小容量衰减

    # 故障参数调整
    resistance_mult = 1.0
    capacity_mult = 1.0
    recovery_mult = 1.0
    thermal_extra = 0.0
    sensor_offset_v = 0.0
    voltage_drift_v = 0.0
    intermittent_drop_v = 0.0

    if fault_type == "high_resistance":
        resistance_mult += 0.28 * severity
        float_v -= 0.012 * severity
    elif fault_type == "low_capacity":
        capacity_mult -= 0.28 * severity
        resistance_mult += 0.25 * severity
        float_v -= 0.07 * severity
    elif fault_type == "slow_recovery":
        recovery_mult += 3.2 * severity
        resistance_mult += 0.12 * severity
    elif fault_type == "thermal_drift":
        thermal_extra = 1.4 * severity
        resistance_mult += 0.10 * severity
    elif fault_type == "sensor_offset":
        sensor_offset_v = -0.13 * severity
    elif fault_type == "voltage_drift":
        # 渐进电压基线漂移：随事件递进，浮充电压持续下降
        voltage_drift_v = -0.08 * severity * aging_frac
        float_v += voltage_drift_v
        resistance_mult += 0.08 * severity
    elif fault_type == "intermittent_fault":
        # 间歇性接触不良：随机电压骤降
        if rng.random() < 0.15 * severity:
            intermittent_drop_v = -rng.uniform(0.05, 0.25) * severity
        resistance_mult += 0.15 * severity
    elif fault_type == "accelerated_aging":
        # 加速老化：容量快速下降 + 内阻快速上升
        capacity_mult -= 0.35 * severity
        resistance_mult += 0.40 * severity
        float_v -= 0.05 * severity
        recovery_mult += 1.0 * severity
    elif fault_type == "sulfation":
        # 硫化：充电接受能力下降，恢复极慢 + 容量损失
        recovery_mult += 4.5 * severity
        capacity_mult -= 0.22 * severity
        resistance_mult += 0.20 * severity
        float_v -= 0.03 * severity

    true_r = base_r * resistance_mult
    true_capacity = capacity * capacity_mult
    r_ohm = true_r / 1000.0
    delta_i = current_a - 4.2  # 相对浮充电流的变化

    temp_factor = 1.0 + 0.003 * (ambient_c - 25.0)  # 温度越高恢复越快
    tau_fast = 32.0 * recovery_mult / temp_factor
    tau_slow = 180.0 * recovery_mult / temp_factor

    # ── 电压计算 ──
    if elapsed_s < 0:
        voltage = float_v + delta_i * r_ohm + intermittent_drop_v
    elif elapsed_s < duration_s:
        discharge_frac = elapsed_s / duration_s
        polarization = 0.065 * (1.0 - math.exp(-elapsed_s / 7.0))
        capacity_sag = 0.20 * discharge_frac * (100.0 / max(true_capacity, 45.0))
        voltage = float_v + delta_i * r_ohm - polarization * (1.0 + 0.15 * severity) - capacity_sag + intermittent_drop_v
    else:
        t_rec = elapsed_s - duration_s
        cutoff_load_a = abs(load_current_a) if load_current_a is not None else abs(current_a)
        cutoff_delta_i = -cutoff_load_a - 4.2
        cutoff_equilibrium_v = float_v + cutoff_delta_i * r_ohm
        polarization_end_v = (
            0.065
            * (1.0 - math.exp(-duration_s / 7.0))
            * (1.0 + 0.15 * severity)
        )
        capacity_sag_end_v = 0.20 * (100.0 / max(true_capacity, 45.0))
        # The cutoff state is fixed by the load immediately before switching.
        # The recovery current may vary, but it must not redefine this state.
        discharge_end_v = (
            cutoff_equilibrium_v
            - polarization_end_v
            - capacity_sag_end_v
        )
        recovery_target = float_v + delta_i * r_ohm
        recovery_lag_v = cutoff_equilibrium_v - discharge_end_v
        lag_fast = recovery_lag_v * 0.75 * math.exp(-t_rec / tau_fast)
        lag_slow = recovery_lag_v * 0.25 * math.exp(-t_rec / tau_slow)
        # A transient contact drop is applied exactly once at the current sample.
        voltage = recovery_target - lag_fast - lag_slow + intermittent_drop_v

    # 传感器偏置只加到在线读数，不加到真实电压
    voltage += sensor_offset_v
    voltage += rng.gauss(0, 0.0045)

    # ── 温度计算 ──
    temperature = ambient_c + temp_offset
    if elapsed_s >= 0 and elapsed_s < duration_s:
        discharge_frac = elapsed_s / duration_s
        heat = 0.32 * (true_r / max(base_r, 1e-6)) + thermal_extra
        temperature += heat * discharge_frac
    elif elapsed_s >= duration_s:
        t_rec = elapsed_s - duration_s
        heat = 0.32 * (true_r / max(base_r, 1e-6)) + thermal_extra
        temperature += heat * math.exp(-t_rec / 180.0)
    temperature += rng.gauss(0, 0.025)

    truth = {
        "true_resistance_mohm": round(true_r, 3),
        "true_capacity_ah": round(true_capacity, 2),
        "sensor_voltage_offset_v": round(sensor_offset_v, 4),
        "recovery_tau_fast_s": round(tau_fast, 1),
        "recovery_tau_slow_s": round(tau_slow, 1),
        "recovery_weight_fast": 0.75,
        "discharge_cutoff_voltage_v": round(
            float_v
            + (-abs(load_current_a or current_a) - 4.2) * r_ohm
            - 0.065 * (1.0 - math.exp(-duration_s / 7.0)) * (1.0 + 0.15 * severity)
            - 0.20 * (100.0 / max(true_capacity, 45.0)),
            4,
        ),
        "thermal_extra_c": round(thermal_extra, 2),
    }
    return round(voltage, 6), round(temperature, 5), truth


def phase_of(elapsed_s: float, duration_s: float) -> str:
    if elapsed_s < 0:
        return "PRE"
    if elapsed_s < duration_s:
        return "DISCHARGE"
    return "RECOVERY"


def event_current(elapsed_s: float, load_current: float, duration_s: float) -> float:
    if elapsed_s < 0:
        return 4.2 + 0.10 * math.sin(elapsed_s * 0.45)
    if elapsed_s < duration_s:
        return -load_current + 0.10 * math.sin(elapsed_s * 0.45)
    t_rec = elapsed_s - duration_s
    return 4.2 + 16.0 * math.exp(-t_rec / 45.0) + 0.10 * math.sin(elapsed_s * 0.45)


def generate_event_grid(duration_s: float) -> list[float]:
    """生成事件时间轴：前30s + 放电段 + 恢复120s"""
    pre = [-PRE_DURATION_S + i * SAMPLE_INTERVAL_S for i in range(int(PRE_DURATION_S / SAMPLE_INTERVAL_S))]
    discharge = [i * SAMPLE_INTERVAL_S for i in range(int(duration_s / SAMPLE_INTERVAL_S) + 1)]
    recovery_start = duration_s + SAMPLE_INTERVAL_S
    recovery = [recovery_start + i * SAMPLE_INTERVAL_S for i in range(int(RECOVERY_DURATION_S / SAMPLE_INTERVAL_S))]
    return [round(t, 3) for t in pre + discharge + recovery]


FLOAT_INTERVAL_S = 600  # 10 分钟

def generate_float_monitoring(
    conn: sqlite3.Connection,
    devices: list[dict[str, str]],
    str_id: str,
    str_idx: int,
    start_time: datetime,
    end_time: datetime,
    rng: random.Random,
    seq_counters: dict[str, int],
    fault_plan: dict[int, tuple[str, float]],
    n_events: int,
    onset_multiplier: float,
    severity_cap: float,
    packet_loss_rate: float,
    upload_latency_sigma_s: float,
    min_onset: int = 0,
) -> int:
    """生成连续浮充监测数据（10分钟间隔）

    浮充期间电池处于 float 模式，电压 ~13.6V，温度 ~环境温度。
    故障电池在浮充期间也有特征表现（电压偏低、温度偏高）。

    B3: raw_message_id=NULL, simulation_only=1
    B5: 使用 str_idx 确保独立基线
    B7: 丢包时序列号仍递增，产生序列间隙
    """
    float_samples = 0
    current_time = start_time
    sample_interval = timedelta(seconds=FLOAT_INTERVAL_S)

    while current_time < end_time:
        # 近似 event_index：按时间进度推算
        elapsed_frac = (current_time - start_time).total_seconds() / max(1, (end_time - start_time).total_seconds())
        approx_event_index = max(1, int(round(elapsed_frac * n_events)))

        # 环境温度日变化
        day_frac = (current_time.hour * 3600 + current_time.minute * 60) / 86400
        ambient_c = 25.0 + 2.5 * math.sin(2 * math.pi * day_frac) + rng.gauss(0, 0.3)

        for device in devices:
            battery_no = int(device["battery_id"][1:])
            battery_id = device["battery_id"]
            sensor_id = device["sensor_id"]

            batt_key = f"battery|{sensor_id}"
            seq = seq_counters[batt_key]
            seq_counters[batt_key] += 1

            if rng.random() < packet_loss_rate:
                continue

            fault_type, onset_event = get_fault(battery_no, n_events, fault_plan, onset_multiplier,
                                                min_onset=min_onset)
            severity = severity_progression(approx_event_index, onset_event, n_events, severity_cap)
            active_fault = fault_type if severity > 0 else "normal"

            # 浮充电压计算
            voltage, temperature, _ = battery_response(
                -1.0, 4.2, FIXED_DURATION_S, ambient_c,
                battery_no, active_fault, severity, rng,
                event_index=approx_event_index, n_events=n_events,
                string_idx=str_idx,
            )

            # 上传延迟（非抖动，是上传时延）
            upload_latency = (
                abs(rng.gauss(0, upload_latency_sigma_s))
                if upload_latency_sigma_s > 0
                else 0
            )
            upload_time = current_time + timedelta(seconds=upload_latency)
            rssi = (-62.0 + (battery_no - 1) * 0.18) + rng.gauss(0, 2.5)

            # 浮充数据 event_id 为空
            sample_key = f"battery|{SITE_ID}|{str_id}||{battery_id}|{sensor_id}|{seq}"

            conn.execute(
                """INSERT OR IGNORE INTO battery_samples(
                    raw_message_id, site_id, string_id, event_id, event_index,
                    battery_id, sensor_id, sequence_no, elapsed_s,
                    sample_timestamp, upload_timestamp, voltage_v, temperature_c,
                    rssi_dbm, status_code, phase, firmware_version,
                    calibration_version, schema_version, quality_flags,
                    simulation_only, sample_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    None,  # B3: float data has no raw MQTT message
                    SITE_ID, str_id, "", None,
                    battery_id, sensor_id, seq, None,
                    _utc_iso(current_time), _utc_iso(upload_time),
                    voltage, temperature, round(rssi, 2),
                    "0", "FLOAT", FIRMWARE_VERSION, CALIBRATION_VERSION,
                    "json_v1", "[]", 1,  # simulation_only=1
                    sample_key,
                ),
            )
            float_samples += 1

        current_time += sample_interval

    return float_samples


def load_device_mapping() -> list[dict[str, str]]:
    """Return synthetic device identities; no field identifiers are used."""
    return [dict(kind="battery", battery_id=f"B{i:02d}", sensor_id=f"SENSOR{i:02d}", topic_device_id=f"DEVICE{i:02d}", string_id="SIM-S1", site_id=SITE_ID) for i in range(1, 41)]


def make_string_devices(str_id: str, base_devices: list[dict[str, str]]) -> list[dict[str, str]]:
    """为指定串生成全局唯一的设备列表（B5: 跨串传感器 ID 独立）

    将 sensor_id 和 topic_device_id 加上串前缀，确保 S1/S2/S3 的传感器身份全局唯一。
    battery_id 保持不变（在 string_id 作用域内唯一即可）。
    """
    result = []
    for d in base_devices:
        new_d = dict(d)
        new_d["sensor_id"] = f"{str_id}-{d['sensor_id']}"
        if "topic_device_id" in d:
            new_d["topic_device_id"] = f"{str_id}-{d['topic_device_id']}"
        result.append(new_d)
    return result


def init_database(db_path: str) -> sqlite3.Connection:
    """初始化 SQLite 数据库（使用与采集系统相同的 schema + synthetic_truth 表）"""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS raw_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_received_at TEXT NOT NULL, last_received_at TEXT NOT NULL,
        topic TEXT NOT NULL, payload_text TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
        qos INTEGER NOT NULL, retained INTEGER NOT NULL, parse_status TEXT NOT NULL,
        parse_error TEXT, sample_count INTEGER NOT NULL DEFAULT 0,
        duplicate_count INTEGER NOT NULL DEFAULT 0,
        UNIQUE(topic, payload_sha256)
    );
    CREATE TABLE IF NOT EXISTS battery_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_message_id INTEGER REFERENCES raw_messages(id),
        site_id TEXT NOT NULL, string_id TEXT NOT NULL,
        event_id TEXT NOT NULL DEFAULT '', event_index INTEGER,
        battery_id TEXT NOT NULL, sensor_id TEXT NOT NULL,
        sequence_no INTEGER, elapsed_s REAL,
        sample_timestamp TEXT NOT NULL, upload_timestamp TEXT NOT NULL,
        voltage_v REAL NOT NULL, temperature_c REAL, rssi_dbm REAL,
        status_code TEXT NOT NULL, phase TEXT NOT NULL DEFAULT '',
        firmware_version TEXT NOT NULL DEFAULT '', calibration_version TEXT NOT NULL DEFAULT '',
        schema_version TEXT NOT NULL, quality_flags TEXT NOT NULL DEFAULT '[]',
        simulation_only INTEGER NOT NULL DEFAULT 0,
        sample_key TEXT NOT NULL UNIQUE
    );
    CREATE TABLE IF NOT EXISTS current_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_message_id INTEGER REFERENCES raw_messages(id),
        site_id TEXT NOT NULL, string_id TEXT NOT NULL,
        event_id TEXT NOT NULL DEFAULT '', event_index INTEGER,
        sequence_no INTEGER, elapsed_s REAL,
        sample_timestamp TEXT NOT NULL, upload_timestamp TEXT NOT NULL,
        current_a REAL NOT NULL, string_voltage_v REAL, ups_load_pct REAL,
        ups_mode TEXT NOT NULL DEFAULT '', alarm_code TEXT NOT NULL DEFAULT '',
        phase TEXT NOT NULL DEFAULT '', schema_version TEXT NOT NULL,
        quality_flags TEXT NOT NULL DEFAULT '[]',
        simulation_only INTEGER NOT NULL DEFAULT 0,
        sample_key TEXT NOT NULL UNIQUE
    );
    CREATE TABLE IF NOT EXISTS environment_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_message_id INTEGER REFERENCES raw_messages(id),
        site_id TEXT NOT NULL, string_id TEXT NOT NULL,
        event_id TEXT NOT NULL DEFAULT '', sequence_no INTEGER,
        sample_timestamp TEXT NOT NULL, upload_timestamp TEXT NOT NULL,
        ambient_temperature_c REAL NOT NULL, ambient_humidity_pct REAL,
        cabinet_id TEXT NOT NULL DEFAULT '', source_id TEXT NOT NULL DEFAULT '',
        schema_version TEXT NOT NULL, quality_flags TEXT NOT NULL DEFAULT '[]',
        simulation_only INTEGER NOT NULL DEFAULT 0,
        sample_key TEXT NOT NULL UNIQUE
    );
    CREATE TABLE IF NOT EXISTS event_metadata (
        event_id TEXT PRIMARY KEY, site_id TEXT NOT NULL, string_id TEXT NOT NULL,
        event_type TEXT NOT NULL, planned_start_timestamp TEXT,
        actual_start_timestamp TEXT, actual_end_timestamp TEXT,
        approved_by TEXT NOT NULL DEFAULT '', operated_by TEXT NOT NULL DEFAULT '',
        recorded_by TEXT NOT NULL DEFAULT '', pre_ups_mode TEXT NOT NULL DEFAULT '',
        pre_load_pct REAL, operation_description TEXT NOT NULL DEFAULT '',
        post_status TEXT NOT NULL DEFAULT '', ambient_temperature_start_c REAL,
        ambient_temperature_end_c REAL, sampling_profile TEXT NOT NULL DEFAULT '',
        firmware_version TEXT NOT NULL DEFAULT '', calibration_version TEXT NOT NULL DEFAULT '',
        field_notes TEXT NOT NULL DEFAULT '', include_status TEXT NOT NULL DEFAULT 'PENDING_QC',
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS reference_measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        measurement_timestamp TEXT NOT NULL, site_id TEXT NOT NULL, string_id TEXT NOT NULL,
        battery_id TEXT NOT NULL, sensor_id TEXT NOT NULL DEFAULT '',
        event_id TEXT NOT NULL DEFAULT '', event_index INTEGER,
        repetition INTEGER NOT NULL,
        reference_voltage_v REAL, reference_ac_resistance_mohm REAL,
        reference_conductance_s REAL, capacity_ah REAL,
        online_dynamic_resistance_mohm REAL, ups_mode TEXT NOT NULL DEFAULT '',
        ambient_temperature_c REAL, instrument_brand TEXT NOT NULL DEFAULT '',
        instrument_model TEXT NOT NULL DEFAULT '', instrument_serial TEXT NOT NULL DEFAULT '',
        calibration_status TEXT NOT NULL DEFAULT '', contact_position TEXT NOT NULL DEFAULT '',
        operator_id TEXT NOT NULL DEFAULT '', maintenance_conclusion TEXT NOT NULL DEFAULT '',
        replacement_reason TEXT NOT NULL DEFAULT '', appearance_notes TEXT NOT NULL DEFAULT '',
        notes TEXT NOT NULL DEFAULT '',
        UNIQUE(measurement_timestamp, string_id, battery_id, repetition, instrument_serial)
    );
    CREATE TABLE IF NOT EXISTS system_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_timestamp TEXT NOT NULL, severity TEXT NOT NULL, source TEXT NOT NULL,
        event_type TEXT NOT NULL, details_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE IF NOT EXISTS synthetic_truth (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        site_id TEXT NOT NULL, string_id TEXT NOT NULL,
        event_id TEXT NOT NULL, event_index INTEGER,
        battery_id TEXT NOT NULL, sensor_id TEXT NOT NULL,
        assigned_fault_type TEXT NOT NULL,
        active_fault_type TEXT NOT NULL,
        fault_severity REAL NOT NULL,
        label_anomaly INTEGER NOT NULL,
        label_battery_weak INTEGER NOT NULL,
        label_sensor_fault INTEGER NOT NULL,
        true_resistance_mohm REAL,
        true_capacity_ah REAL,
        sensor_voltage_offset_v REAL,
        recovery_tau_fast_s REAL,
        recovery_tau_slow_s REAL,
        recovery_weight_fast REAL,
        discharge_cutoff_voltage_v REAL,
        thermal_extra_c REAL,
        UNIQUE(string_id, event_id, battery_id)
    );
    CREATE INDEX IF NOT EXISTS idx_battery_time ON battery_samples(string_id, battery_id, sample_timestamp);
    CREATE INDEX IF NOT EXISTS idx_battery_event ON battery_samples(event_id, battery_id, elapsed_s);
    CREATE INDEX IF NOT EXISTS idx_current_time ON current_samples(string_id, sample_timestamp);
    CREATE INDEX IF NOT EXISTS idx_current_event ON current_samples(event_id, elapsed_s);
    CREATE INDEX IF NOT EXISTS idx_environment_time ON environment_samples(string_id, sample_timestamp);
    CREATE INDEX IF NOT EXISTS idx_reference_battery ON reference_measurements(string_id, battery_id, measurement_timestamp);
    CREATE INDEX IF NOT EXISTS idx_system_event_time ON system_events(event_timestamp);
    CREATE INDEX IF NOT EXISTS idx_raw_received ON raw_messages(first_received_at);
    CREATE INDEX IF NOT EXISTS idx_truth_event ON synthetic_truth(event_id, battery_id);
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(schema_sql)
    conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', '2')")
    conn.commit()
    return conn


def generate_simulation(
    db_path: str,
    n_events: int = 30,
    span_days: int = 15,
    seed: int = 20260725,
    clean: bool = False,
    strings: list[dict[str, Any]] | None = None,
    difficulty: str = "medium",
    float_days: int = 0,
    packet_loss_rate: float = -1.0,
    upload_latency_sigma_s: float = -1.0,
) -> dict[str, Any]:
    """生成完整模拟数据集

    如果 strings 不为 None，则按多串模式生成。
    每个 string dict: {"string_id": "SIM-S1", "n_events": 5, "seed_offset": 0}

    P1 参数：
    - difficulty: easy/medium/hard，影响故障数量、onset 和严重度
    - float_days: 连续浮充监测天数（0=不生成）
    - packet_loss_rate: 事件采样丢包率
    - upload_latency_sigma_s: 非负半正态上传时延的 sigma
    """

    generation_started_at = datetime.now(UTC)

    if clean and Path(db_path).exists():
        Path(db_path).unlink()
        for suffix in ("-wal", "-shm"):
            p = Path(db_path + suffix)
            if p.exists():
                p.unlink()

    conn = init_database(db_path)
    rng = random.Random(seed)

    devices = load_device_mapping()
    assert len(devices) == N_BATTERIES, f"expected {N_BATTERIES} battery devices, got {len(devices)}"

    # 多串配置
    if strings is None:
        strings = [{"string_id": STRING_ID, "n_events": n_events, "seed_offset": 0}]

    # 难度配置
    diff_cfg = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["medium"])
    onset_mult = diff_cfg["onset_multiplier"]
    sev_cap = diff_cfg["severity_cap"]
    # 如果调用方未显式指定（-1），使用难度默认值
    if packet_loss_rate < 0:
        packet_loss_rate = diff_cfg["packet_loss_rate"]
    if upload_latency_sigma_s < 0:
        upload_latency_sigma_s = diff_cfg["upload_latency_sigma_s"]

    stats = {
        "events": 0, "battery_samples": 0, "current_samples": 0,
        "environment_samples": 0, "reference_measurements": 0,
        "truth_rows": 0, "float_samples": 0,
        "fault_batteries": 0,  # 更新为跨串并集
        "n_strings": len(strings),
        "difficulty": difficulty,
    }

    # 连续序列号计数器（跨事件、跨串连续）
    seq_counters: dict[str, int] = {}
    seq_counters["current"] = 0
    seq_counters["env"] = 0

    now = datetime.now(TZ_CHINA)
    # 生成截止时间（留 1 小时余量避免 upload 延迟越过 now）
    gen_cutoff = now - timedelta(hours=1)

    all_fault_batteries: set[int] = set()
    fault_plan_by_string: dict[str, dict[str, dict[str, Any]]] = {}
    train_contract_by_string: dict[str, dict[str, int]] = {}

    for str_idx, str_config in enumerate(strings):
        str_id = str_config["string_id"]
        str_n_events = str_config.get("n_events", n_events)
        str_train_end = p1_train_end_event(str_n_events)
        str_min_onset = (
            p1_min_fault_onset_event(str_n_events)
            if difficulty in ("easy", "medium", "hard")
            else 0
        )
        str_seed = seed + str_config.get("seed_offset", str_idx * 1000)
        str_rng = random.Random(str_seed)

        str_devices = make_string_devices(str_id, devices)
        for d in str_devices:
            key = f"battery|{d['sensor_id']}"
            seq_counters[key] = 0

        str_fault_plan = get_fault_plan(difficulty, string_idx=str_idx)
        all_fault_batteries.update(str_fault_plan.keys())
        fault_plan_by_string[str_id] = {}
        for battery_no, (fault_type, onset_fraction) in sorted(str_fault_plan.items()):
            _, actual_onset_event = get_fault(
                battery_no,
                str_n_events,
                str_fault_plan,
                onset_mult,
                min_onset=str_min_onset,
            )
            fault_plan_by_string[str_id][f"B{battery_no:02d}"] = {
                "fault_type": fault_type,
                "onset_fraction": onset_fraction,
                "onset_event": actual_onset_event,
            }
        train_contract_by_string[str_id] = {
            "train_end_event": str_train_end,
            "minimum_fault_onset_event": str_min_onset,
        }

        str_span = span_days

        start_date = now - timedelta(days=str_span)
        event_times: list[datetime] = []
        for i in range(str_n_events):
            fraction = i / max(1, str_n_events - 1)
            event_time = start_date + timedelta(
                days=fraction * str_span,
                hours=str_rng.gauss(0, 3),
                minutes=str_rng.gauss(0, 20),
            )
            if event_time > gen_cutoff:
                event_time = gen_cutoff - timedelta(seconds=str_rng.randint(60, 600))
            event_times.append(event_time)

        reference_event_indices = sorted({1, str_n_events // 2 + 1, str_n_events})

        print(f"\n  ── 串 {str_id} ({str_n_events} 事件, span={str_span}天, "
              f"故障电池: {sorted(str_fault_plan.keys())}) ──")

        for event_index in range(1, str_n_events + 1):
            event_time = event_times[event_index - 1]
            event_id = f"{str_id}-E{event_index:03d}"
            load_pct, nominal_current = LOAD_LEVELS[(event_index - 1) % 3]
            load_current = nominal_current + str_rng.gauss(0, 1.3)
            duration_s = FIXED_DURATION_S
            ambient_c = 25.0 + 2.2 * math.sin(2 * math.pi * event_index / str_n_events) + str_rng.gauss(0, 0.45)

            grid = generate_event_grid(duration_s)
            event_start = event_time + timedelta(seconds=grid[0])

            # ── 写入事件元数据 ──
            now_iso = _utc_iso(datetime.now(UTC))
            actual_end = event_time + timedelta(seconds=duration_s + RECOVERY_DURATION_S)
            conn.execute(
                """INSERT OR REPLACE INTO event_metadata(
                    event_id, site_id, string_id, event_type,
                    actual_start_timestamp, actual_end_timestamp,
                    approved_by, operated_by, recorded_by,
                    pre_ups_mode, pre_load_pct, operation_description,
                    ambient_temperature_start_c, ambient_temperature_end_c,
                    sampling_profile, firmware_version, calibration_version,
                    field_notes, include_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id, SITE_ID, str_id, "SIMULATED_DISCHARGE",
                    _utc_iso(event_start), _utc_iso(actual_end),
                    "SIMULATION", "SIMULATION", "SIMULATION",
                    "FLOAT", load_pct, "Simulated maintenance discharge event",
                    round(ambient_c, 2), round(ambient_c + 0.5, 2),
                    f"{SAMPLE_INTERVAL_S}s event grid, {EVENT_GRID_POINTS} points",
                    FIRMWARE_VERSION, CALIBRATION_VERSION,
                    "SIMULATION DATA -- for pipeline validation only, not field evidence",
                    "EXCLUDE_SIM", now_iso, now_iso,
                ),
            )

            # ── 预计算电流和环境采样 ──
            current_samples_data = []
            env_samples_data = []
            for gi, elapsed in enumerate(grid):
                cur = event_current(elapsed, load_current, duration_s)
                sample_time = event_time + timedelta(seconds=elapsed)
                upload_time = sample_time + timedelta(seconds=abs(str_rng.gauss(0, 0.3)))
                ph = phase_of(elapsed, duration_s)
                mode = "FLOAT" if ph == "PRE" else ("BATTERY" if ph == "DISCHARGE" else "RECOVERY")
                current_samples_data.append({
                    "elapsed": elapsed, "current": cur, "sample_time": sample_time,
                    "upload_time": upload_time, "phase": ph, "mode": mode,
                })
                env_temp = ambient_c + 0.15 * math.sin(gi / 10)
                env_humid = 52.0 + 0.3 * math.cos(gi / 12)
                env_samples_data.append({
                    "elapsed": elapsed, "temp": env_temp, "humid": env_humid,
                    "sample_time": sample_time, "upload_time": upload_time,
                })

            # ── 写入电流原始报文 ──
            current_topic = f"ups/{SITE_ID}/{str_id}/current"
            current_payload_obj = {
                "schema_version": "1.0", "kind": "current",
                "site_id": SITE_ID, "string_id": str_id,
                "event_id": event_id, "event_index": event_index,
                "samples": [
                    {"t": _utc_iso(s["sample_time"]), "i": round(s["current"], 5),
                     "ph": s["phase"], "mode": s["mode"]}
                    for s in current_samples_data
                ],
            }
            current_payload_bytes = json.dumps(current_payload_obj, separators=(",", ":")).encode("utf-8")
            current_payload_sha = _sha256_bytes(current_payload_bytes)
            current_payload_text = current_payload_bytes.decode("utf-8")
            raw_cursor = conn.execute(
                """INSERT INTO raw_messages(
                    first_received_at, last_received_at, topic, payload_text,
                    payload_sha256, qos, retained, parse_status, parse_error, sample_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (_utc_iso(event_time), _utc_iso(event_time), current_topic,
                 current_payload_text, current_payload_sha, 1, 0, "ok", None, len(grid)),
            )
            current_raw_id = raw_cursor.lastrowid

            # ── 写入环境原始报文 ──
            env_topic = f"ups/{SITE_ID}/{str_id}/environment"
            env_payload_obj = {
                "schema_version": "1.0", "kind": "environment",
                "site_id": SITE_ID, "string_id": str_id,
                "event_id": event_id,
                "samples": [
                    {"t": _utc_iso(s["sample_time"]), "tc": round(s["temp"], 3), "rh": round(s["humid"], 3)}
                    for s in env_samples_data
                ],
            }
            env_payload_bytes = json.dumps(env_payload_obj, separators=(",", ":")).encode("utf-8")
            env_payload_sha = _sha256_bytes(env_payload_bytes)
            env_payload_text = env_payload_bytes.decode("utf-8")
            raw_cursor = conn.execute(
                """INSERT INTO raw_messages(
                    first_received_at, last_received_at, topic, payload_text,
                    payload_sha256, qos, retained, parse_status, parse_error, sample_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (_utc_iso(event_time), _utc_iso(event_time), env_topic,
                 env_payload_text, env_payload_sha, 1, 0, "ok", None, len(grid)),
            )
            env_raw_id = raw_cursor.lastrowid

            # ── 写入电流采样 ──
            for s in current_samples_data:
                seq = seq_counters["current"]
                seq_counters["current"] += 1
                sample_key = f"current|{SITE_ID}|{str_id}|{event_id}|||{seq}"
                conn.execute(
                    """INSERT OR IGNORE INTO current_samples(
                        raw_message_id, site_id, string_id, event_id, event_index,
                        sequence_no, elapsed_s, sample_timestamp, upload_timestamp,
                        current_a, string_voltage_v, ups_load_pct, ups_mode, alarm_code,
                        phase, schema_version, quality_flags,
                        simulation_only, sample_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        current_raw_id, SITE_ID, str_id, event_id, event_index,
                        seq, s["elapsed"],
                        _utc_iso(s["sample_time"]), _utc_iso(s["upload_time"]),
                        round(s["current"], 5), round(40 * (13.63 + (s["current"] - 4.2) * 0.0091), 4),
                        load_pct, s["mode"], "NONE", s["phase"], "json_v1", "[]",
                        1,  # simulation_only=1
                        sample_key,
                    ),
                )
                stats["current_samples"] += 1

            # ── 写入环境采样 ──
            for s in env_samples_data:
                seq = seq_counters["env"]
                seq_counters["env"] += 1
                env_key = f"environment|{SITE_ID}|{str_id}|{event_id}|||{seq}"
                conn.execute(
                    """INSERT OR IGNORE INTO environment_samples(
                        raw_message_id, site_id, string_id, event_id, sequence_no,
                        sample_timestamp, upload_timestamp, ambient_temperature_c,
                        ambient_humidity_pct, cabinet_id, source_id, schema_version,
                        quality_flags, simulation_only, sample_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        env_raw_id, SITE_ID, str_id, event_id, seq,
                        _utc_iso(s["sample_time"]), _utc_iso(s["upload_time"]),
                        round(s["temp"], 3), round(s["humid"], 3),
                        "CABINET-01", "ENV-SIM", "json_v1", "[]",
                        1,  # simulation_only=1
                        env_key,
                    ),
                )
                stats["environment_samples"] += 1

            # ── 写入 40 节电池采样 ──
            for device in str_devices:
                battery_no = int(device["battery_id"][1:])
                battery_id = device["battery_id"]
                sensor_id = device["sensor_id"]
                topic_device_id = device["topic_device_id"]

                fault_type, onset_event = get_fault(
                    battery_no,
                    str_n_events,
                    str_fault_plan,
                    onset_mult,
                    min_onset=str_min_onset,
                )
                severity = severity_progression(event_index, onset_event, str_n_events, sev_cap)
                active_fault = fault_type if severity > 0 else "normal"

                batt_key = f"battery|{sensor_id}"
                battery_samples_data = []
                for gi, elapsed in enumerate(grid):
                    seq = seq_counters[batt_key]
                    seq_counters[batt_key] += 1

                    if str_rng.random() < packet_loss_rate:
                        continue
                    cur = event_current(elapsed, load_current, duration_s)
                    voltage, temperature, truth = battery_response(
                        elapsed, cur, duration_s, ambient_c,
                        battery_no, active_fault, severity, str_rng,
                        event_index=event_index, n_events=str_n_events,
                        string_idx=str_idx,
                        load_current_a=load_current,
                    )
                    sample_time = event_time + timedelta(seconds=elapsed)
                    upload_latency = abs(
                        str_rng.gauss(0, upload_latency_sigma_s)
                    )
                    upload_time = sample_time + timedelta(seconds=upload_latency)
                    ph = phase_of(elapsed, duration_s)
                    rssi = (-62.0 + (battery_no - 1) * 0.18) + str_rng.gauss(0, 2.1)
                    battery_samples_data.append({
                        "elapsed": elapsed, "voltage": voltage, "temperature": temperature,
                        "sample_time": sample_time, "upload_time": upload_time,
                        "phase": ph, "rssi": rssi, "truth": truth, "seq": seq,
                    })

                # ── 写入电池原始报文 ──
                batt_topic = f"ups/{SITE_ID}/{str_id}/battery/{topic_device_id}"
                batt_payload_obj = {
                    "schema_version": "1.0", "kind": "battery",
                    "site_id": SITE_ID, "string_id": str_id,
                    "battery_id": battery_id, "sensor_id": sensor_id,
                    "event_id": event_id, "event_index": event_index,
                    "samples": [
                        {"t": _utc_iso(s["sample_time"]), "v": s["voltage"],
                         "tc": s["temperature"], "ph": s["phase"], "rssi": round(s["rssi"], 2)}
                        for s in battery_samples_data
                    ],
                }
                batt_payload_bytes = json.dumps(batt_payload_obj, separators=(",", ":")).encode("utf-8")
                batt_payload_sha = _sha256_bytes(batt_payload_bytes)
                batt_payload_text = batt_payload_bytes.decode("utf-8")
                raw_cursor = conn.execute(
                    """INSERT INTO raw_messages(
                        first_received_at, last_received_at, topic, payload_text,
                        payload_sha256, qos, retained, parse_status, parse_error, sample_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (_utc_iso(event_time), _utc_iso(event_time), batt_topic,
                     batt_payload_text, batt_payload_sha, 1, 0, "ok", None,
                     len(battery_samples_data)),  # B4: 实际样本数（丢包后）
                )
                batt_raw_id = raw_cursor.lastrowid

                # ── 写入电池采样 ──
                for s in battery_samples_data:
                    seq = s["seq"]
                    sample_key = f"battery|{SITE_ID}|{str_id}|{event_id}|{battery_id}|{sensor_id}|{seq}"
                    conn.execute(
                        """INSERT OR IGNORE INTO battery_samples(
                            raw_message_id, site_id, string_id, event_id, event_index,
                            battery_id, sensor_id, sequence_no, elapsed_s,
                            sample_timestamp, upload_timestamp, voltage_v, temperature_c,
                            rssi_dbm, status_code, phase, firmware_version,
                            calibration_version, schema_version, quality_flags,
                            simulation_only, sample_key
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            batt_raw_id, SITE_ID, str_id, event_id, event_index,
                            battery_id, sensor_id, seq, s["elapsed"],
                            _utc_iso(s["sample_time"]), _utc_iso(s["upload_time"]),
                            s["voltage"], s["temperature"], round(s["rssi"], 2),
                            "0", s["phase"], FIRMWARE_VERSION, CALIBRATION_VERSION,
                            "json_v1", "[]", 1,  # simulation_only=1
                            sample_key,
                        ),
                    )
                    stats["battery_samples"] += 1

                last_truth = battery_samples_data[-1]["truth"] if battery_samples_data else {}
                is_sensor_fault = int(active_fault == "sensor_offset" and severity >= 0.20)
                is_battery_weak = int(
                    active_fault in {
                        "high_resistance", "low_capacity", "slow_recovery", "thermal_drift",
                        "voltage_drift", "intermittent_fault", "accelerated_aging", "sulfation",
                    }
                    and severity >= 0.20
                )
                label_anomaly = int(is_sensor_fault or is_battery_weak)
                conn.execute(
                    """INSERT OR IGNORE INTO synthetic_truth(
                        site_id, string_id, event_id, event_index,
                        battery_id, sensor_id,
                        assigned_fault_type, active_fault_type, fault_severity,
                        label_anomaly, label_battery_weak, label_sensor_fault,
                        true_resistance_mohm, true_capacity_ah, sensor_voltage_offset_v,
                        recovery_tau_fast_s, recovery_tau_slow_s, recovery_weight_fast,
                        discharge_cutoff_voltage_v, thermal_extra_c
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        SITE_ID, str_id, event_id, event_index,
                        battery_id, sensor_id,
                        fault_type, active_fault, round(severity, 3),
                        label_anomaly, is_battery_weak, is_sensor_fault,
                        last_truth.get("true_resistance_mohm"),
                        last_truth.get("true_capacity_ah"),
                        last_truth.get("sensor_voltage_offset_v"),
                        last_truth.get("recovery_tau_fast_s"),
                        last_truth.get("recovery_tau_slow_s"),
                        last_truth.get("recovery_weight_fast"),
                        last_truth.get("discharge_cutoff_voltage_v"),
                        last_truth.get("thermal_extra_c"),
                    ),
                )
                stats["truth_rows"] += 1

                # ── 参考测量 ──
                if event_index in reference_event_indices:
                    ref_truth = battery_response(
                        0, 4.2, duration_s, ambient_c, battery_no, fault_type, severity, str_rng,
                        event_index=event_index, n_events=str_n_events,
                        string_idx=str_idx,
                        load_current_a=load_current,
                    )[2]
                    true_float_v = BASE_FLOAT_V + random.Random(
                        battery_no * 7919 + str_idx * 100003
                    ).gauss(0, 0.022)
                    if fault_type == "high_resistance":
                        true_float_v -= 0.012 * severity
                    elif fault_type == "low_capacity":
                        true_float_v -= 0.07 * severity
                    elif fault_type == "accelerated_aging":
                        true_float_v -= 0.05 * severity
                    elif fault_type == "sulfation":
                        true_float_v -= 0.03 * severity
                    elif fault_type == "voltage_drift":
                        true_float_v -= 0.08 * severity * (event_index / max(1, str_n_events))

                    for rep in range(1, 4):
                        measured_r = ref_truth["true_resistance_mohm"] * str_rng.gauss(1.0, 0.015)
                        measured_cap = ref_truth["true_capacity_ah"] * str_rng.gauss(1.0, 0.008)
                        ref_voltage = true_float_v + str_rng.gauss(0, 0.003)
                        ref_time = event_time - timedelta(minutes=20 - rep)
                        conn.execute(
                            """INSERT OR IGNORE INTO reference_measurements(
                                measurement_timestamp, site_id, string_id, battery_id, sensor_id,
                                event_id, event_index,
                                repetition, reference_voltage_v, reference_ac_resistance_mohm,
                                reference_conductance_s, capacity_ah, online_dynamic_resistance_mohm,
                                ups_mode, ambient_temperature_c, instrument_brand, instrument_model,
                                instrument_serial, calibration_status, contact_position, operator_id,
                                maintenance_conclusion, replacement_reason, appearance_notes, notes
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                _utc_iso(ref_time), SITE_ID, str_id, battery_id, sensor_id,
                                event_id, event_index, rep,
                                round(ref_voltage, 4), round(measured_r, 3),
                                round(1000.0 / max(measured_r, 1e-6), 2), round(measured_cap, 2),
                                round(ref_truth["true_resistance_mohm"] * str_rng.gauss(1.0, 0.035), 3),
                                "FLOAT", round(ambient_c, 2),
                                "SIM", "SIM-REF-01", "SIM-SERIAL-001",
                                "calibrated", "terminal", "SIM-OPERATOR",
                                "", "", "", "Simulated reference measurement (independent instrument)",
                            ),
                        )
                        stats["reference_measurements"] += 1

            stats["events"] += 1
            conn.commit()

            # 进度报告
            fault_summary = []
            for bno in sorted(str_fault_plan.keys()):
                ft, oe = get_fault(
                    bno,
                    str_n_events,
                    str_fault_plan,
                    onset_mult,
                    min_onset=str_min_onset,
                )
                sev = severity_progression(event_index, oe, str_n_events, sev_cap)
                if sev > 0:
                    fault_summary.append(f"B{bno:02d}({ft}:{sev:.1%})")
            print(f"  事件 {event_index:3d}/{str_n_events}  {event_id}  "
                  f"负载={load_pct:.0f}%  放电={duration_s:.0f}s  "
                  f"故障: {', '.join(fault_summary) if fault_summary else '无'}")

        # ── 生成连续浮充监测数据 ──
        if float_days > 0:
            float_start = now - timedelta(days=float_days)
            float_end = now - timedelta(hours=1)  # 留 1 小时余量
            print(f"\n  ── 串 {str_id} 浮充监测 ({float_days} 天, {FLOAT_INTERVAL_S}s 间隔) ──")
            float_count = generate_float_monitoring(
                conn, str_devices, str_id, str_idx,
                float_start, float_end,
                str_rng, seq_counters,
                str_fault_plan, str_n_events, onset_mult, sev_cap,
                packet_loss_rate, upload_latency_sigma_s,
                min_onset=str_min_onset,
            )
            stats["float_samples"] += float_count
            conn.commit()
            print(f"  浮充样本: {float_count:,}")

    stats["fault_batteries"] = len(all_fault_batteries)

    generation_completed_at = datetime.now(UTC)

    # ── 写入系统事件 ──
    conn.execute(
        """INSERT INTO system_events(event_timestamp, severity, source, event_type, details_json)
           VALUES (?, ?, ?, ?, ?)""",
        (_utc_iso(now), "INFO", "simulator", "SIMULATION_GENERATED",
         json.dumps({
             "n_strings": len(strings),
             "n_batteries": N_BATTERIES,
             "strings": [s["string_id"] for s in strings],
             "fault_batteries": sorted(all_fault_batteries),
             "fault_plan_by_string": fault_plan_by_string,
             "seed": seed,
             "difficulty": difficulty,
             "event_span_days": span_days,
             "float_days": float_days,
             "packet_loss_rate": packet_loss_rate,
             "upload_latency_distribution": "half_normal_nonnegative",
             "upload_latency_sigma_s": upload_latency_sigma_s,
             "train_fraction": P1_TRAIN_FRACTION,
             "train_contract_by_string": train_contract_by_string,
             "generation_started_at": _utc_iso(generation_started_at),
             "generation_completed_at": _utc_iso(generation_completed_at),
             "generator_version": GENERATOR_VERSION,
         }, sort_keys=True)),
    )

    conn.commit()
    conn.close()
    return stats


# ── synthetic_v2 导出 ─────────────────────────────────────

BATTERY_CSV_COLUMNS = [
    "site_id", "string_id", "event_id", "event_index",
    "battery_id", "sensor_id", "sequence_no", "elapsed_s",
    "sample_timestamp", "upload_timestamp",
    "voltage_v", "temperature_c", "rssi_dbm",
    "status_code", "phase", "firmware_version", "calibration_version",
]

CURRENT_CSV_COLUMNS = [
    "site_id", "string_id", "event_id", "event_index",
    "sequence_no", "elapsed_s", "sample_timestamp", "upload_timestamp",
    "current_a", "string_voltage_v", "ups_load_pct", "ups_mode", "alarm_code",
]

TRUTH_CSV_COLUMNS = [
    "site_id", "string_id", "event_id", "event_index",
    "battery_id", "sensor_id",
    "assigned_fault_type", "active_fault_type", "fault_severity",
    "label_anomaly", "label_battery_weak", "label_sensor_fault",
    "true_resistance_mohm", "true_capacity_ah", "sensor_voltage_offset_v",
    "recovery_tau_fast_s", "recovery_tau_slow_s", "recovery_weight_fast",
    "discharge_cutoff_voltage_v", "thermal_extra_c",
]

EVENT_METADATA_CSV_COLUMNS = [
    "site_id", "string_id", "event_id", "event_index",
    "event_timestamp", "event_type", "load_stratum",
    "ups_load_pct", "discharge_duration_s", "ambient_temperature_c",
    "expected_grid_points", "include_status",
    "approved_by", "operated_by", "recorded_by",
    "simulation_only",
]

REFERENCE_CSV_COLUMNS = [
    "measurement_timestamp", "site_id", "string_id",
    "battery_id", "sensor_id", "event_id", "event_index", "repetition",
    "reference_voltage_v", "reference_ac_resistance_mohm",
    "reference_conductance_s", "capacity_ah",
    "online_dynamic_resistance_mohm", "ups_mode",
    "ambient_temperature_c", "instrument_id",
]

INVENTORY_CSV_COLUMNS = [
    "site_id", "string_id", "battery_id", "sensor_id",
    "battery_model", "batch_id", "manufacture_date", "install_date",
    "firmware_version", "calibration_version",
]

MAPPING_CSV_COLUMNS = [
    "site_id", "string_id", "battery_id", "sensor_id",
    "mapping_valid_from", "mapping_valid_to", "mapping_reason",
]

FLOAT_CSV_COLUMNS = [
    "site_id", "string_id", "battery_id", "sensor_id",
    "sequence_no", "sample_timestamp", "upload_timestamp",
    "voltage_v", "temperature_c", "rssi_dbm",
    "status_code", "firmware_version", "calibration_version",
]


def _write_gzip_csv(path: Path, columns: list[str], rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
            count += 1
    return count


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
            count += 1
    return count


def export_synthetic_v2(db_path: str, output_root: str, seed: int = 20260725,
                        dataset_name: str = "synthetic_v2",
                        difficulty: str = "medium",
                        packet_loss_rate: float = 0.0,
                        upload_latency_sigma_s: float = 0.0,
                        generator_version: str = GENERATOR_VERSION,
                        ) -> dict[str, Any]:
    """从 SQLite 导出为 ups_ai_pipeline/data/{dataset_name}/ 标准目录

    B9: manifest 包含完整溯源信息（dataset_name, difficulty, generator_version 等）
    B5: 设备清单使用每串独立 sensor_id
    """

    root = Path(output_root).resolve()
    if root.exists():
        import shutil
        shutil.rmtree(root)
    for d in ("assets", "events", "continuous", "reference"):
        (root / d).mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    provenance_row = conn.execute(
        """
        SELECT details_json FROM system_events
        WHERE event_type = 'SIMULATION_GENERATED'
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    provenance = json.loads(provenance_row["details_json"]) if provenance_row else {}

    # 电池采样
    battery_rows = [dict(r) for r in conn.execute(
        f"SELECT {', '.join(BATTERY_CSV_COLUMNS)} FROM battery_samples WHERE event_id <> '' ORDER BY event_id, battery_id, sample_timestamp"
    )]
    # 电流采样
    current_rows = [dict(r) for r in conn.execute(
        f"SELECT {', '.join(CURRENT_CSV_COLUMNS)} FROM current_samples WHERE event_id <> '' ORDER BY event_id, sample_timestamp"
    )]
    # ground truth
    truth_rows = [dict(r) for r in conn.execute(
        f"SELECT {', '.join(TRUTH_CSV_COLUMNS)} FROM synthetic_truth ORDER BY event_id, battery_id"
    )]
    # 参考测量
    ref_rows = []
    for r in conn.execute("SELECT * FROM reference_measurements ORDER BY measurement_timestamp, battery_id, repetition"):
        row = dict(r)
        ref_rows.append({
            "measurement_timestamp": row["measurement_timestamp"],
            "site_id": row["site_id"], "string_id": row["string_id"],
            "battery_id": row["battery_id"], "sensor_id": row["sensor_id"],
            "event_id": row.get("event_id", ""),
            "event_index": row.get("event_index", ""),
            "repetition": row["repetition"],
            "reference_voltage_v": row["reference_voltage_v"],
            "reference_ac_resistance_mohm": row["reference_ac_resistance_mohm"],
            "reference_conductance_s": row["reference_conductance_s"],
            "capacity_ah": row["capacity_ah"],
            "online_dynamic_resistance_mohm": row["online_dynamic_resistance_mohm"],
            "ups_mode": row["ups_mode"],
            "ambient_temperature_c": row["ambient_temperature_c"],
            "instrument_id": f"{row.get('instrument_brand', '')}-{row.get('instrument_model', '')}-{row.get('instrument_serial', '')}",
        })

    # 事件元数据（映射为 pipeline 格式）
    # 先建 event_index 映射，供 load_stratum 推导使用
    event_index_map = {}
    for r in conn.execute("SELECT DISTINCT event_id, event_index FROM battery_samples"):
        event_index_map[r["event_id"]] = r["event_index"]

    event_meta_rows = []
    for r in conn.execute("SELECT * FROM event_metadata ORDER BY event_id"):
        row = dict(r)
        # event_timestamp = 电流边沿时刻 (elapsed_s=0), 即 actual_start + PRE_DURATION_S
        actual_start_str = row.get("actual_start_timestamp", "")
        event_ts = actual_start_str
        if actual_start_str:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(actual_start_str.replace("Z", "+00:00"))
                dt = dt + timedelta(seconds=PRE_DURATION_S)
                event_ts = _utc_iso(dt)
            except Exception:
                pass
        ev_idx = event_index_map.get(row["event_id"], 0)
        event_meta_rows.append({
            "site_id": row["site_id"],
            "string_id": row["string_id"],
            "event_id": row["event_id"],
            "event_index": ev_idx,
            "event_timestamp": event_ts,  # 电流边沿时刻 (elapsed_s=0)
            "event_type": row["event_type"],
            "load_stratum": f"L{1 + (ev_idx - 1) % 3}" if ev_idx else "L1",
            "ups_load_pct": row.get("pre_load_pct"),
            "discharge_duration_s": FIXED_DURATION_S,
            "ambient_temperature_c": row.get("ambient_temperature_start_c"),
            "expected_grid_points": EVENT_GRID_POINTS,
            "include_status": row.get("include_status", "EXCLUDE_SIM"),
            "approved_by": row.get("approved_by", ""),
            "operated_by": row.get("operated_by", ""),
            "recorded_by": row.get("recorded_by", ""),
            "simulation_only": 1,
        })

    base_devices = load_device_mapping()
    string_ids_in_data = sorted(set(r["string_id"] for r in event_meta_rows))
    inventory_rows = []
    mapping_rows = []
    batch_ids_by_string: dict[str, list[str]] = {}
    manufacture_dates = ["2021-06-01", "2021-09-15", "2022-01-10"]
    install_dates = ["2022-01-15", "2022-04-20", "2022-08-05"]
    for string_idx, sid in enumerate(string_ids_in_data):
        str_devices = make_string_devices(sid, base_devices)
        batch_ids_by_string[sid] = []
        for d in str_devices:
            bid = d["battery_id"]
            sensor_id = d["sensor_id"]  # 已包含串前缀
            bno = int(bid[1:])
            batch_id = f"{sid}-LOT-{1 + (bno - 1) // 10:02d}"
            if batch_id not in batch_ids_by_string[sid]:
                batch_ids_by_string[sid].append(batch_id)
            inventory_rows.append({
                "site_id": SITE_ID, "string_id": sid,
                "battery_id": bid, "sensor_id": sensor_id,
                "battery_model": "SIM-VRLA-12V-100Ah",
                "batch_id": batch_id,
                "manufacture_date": manufacture_dates[string_idx % len(manufacture_dates)],
                "install_date": install_dates[string_idx % len(install_dates)],
                "firmware_version": FIRMWARE_VERSION,
                "calibration_version": CALIBRATION_VERSION,
            })
            mapping_rows.append({
                "site_id": SITE_ID, "string_id": sid,
                "battery_id": bid, "sensor_id": sensor_id,
                "mapping_valid_from": "2026-01-01T00:00:00+08:00",
                "mapping_valid_to": "",
                "mapping_reason": "initial",
            })

    # 浮充监测数据（event_id 为空的电池采样）
    float_rows = [dict(r) for r in conn.execute(
        f"""SELECT site_id, string_id, battery_id, sensor_id,
                  sequence_no, sample_timestamp, upload_timestamp,
                  voltage_v, temperature_c, rssi_dbm,
                  status_code, firmware_version, calibration_version
           FROM battery_samples WHERE event_id = '' ORDER BY string_id, battery_id, sample_timestamp"""
    )]

    # 写入文件
    battery_count = _write_gzip_csv(root / "events/battery_event_raw.csv.gz", BATTERY_CSV_COLUMNS, battery_rows)
    current_count = _write_gzip_csv(root / "events/current_event_raw.csv.gz", CURRENT_CSV_COLUMNS, current_rows)
    _write_csv(root / "events/event_metadata.csv", EVENT_METADATA_CSV_COLUMNS, event_meta_rows)
    _write_csv(root / "events/synthetic_ground_truth.csv", TRUTH_CSV_COLUMNS, truth_rows)
    _write_csv(root / "reference/reference_measurements.csv", REFERENCE_CSV_COLUMNS, ref_rows)
    _write_csv(root / "assets/battery_inventory.csv", INVENTORY_CSV_COLUMNS, inventory_rows)
    _write_csv(root / "assets/sensor_mapping.csv", MAPPING_CSV_COLUMNS, mapping_rows)
    float_count = _write_gzip_csv(root / "continuous/float_monitoring_raw.csv.gz", FLOAT_CSV_COLUMNS, float_rows)

    # manifest
    n_truth = len(truth_rows)
    # 按串统计事件数
    events_per_string: dict[str, int] = {}
    for r in event_meta_rows:
        sid = r["string_id"]
        events_per_string[sid] = events_per_string.get(sid, 0) + 1
    n_strings = len(events_per_string)
    max_events_per_string = max(events_per_string.values()) if events_per_string else 0
    manifest = {
        "dataset_name": dataset_name,
        "simulation_only": True,
        "warning": "Synthetic labels and metrics are for pipeline validation only.",
        "seed": seed,
        "difficulty": difficulty,
        "packet_loss_rate": packet_loss_rate,
        "upload_latency_distribution": provenance.get(
            "upload_latency_distribution", "half_normal_nonnegative"
        ),
        "upload_latency_sigma_s": provenance.get(
            "upload_latency_sigma_s", upload_latency_sigma_s
        ),
        "generator_version": generator_version,
        "generation_started_at": provenance.get("generation_started_at"),
        "generation_completed_at": provenance.get("generation_completed_at"),
        "schema_version": "2",
        "event_span_days": provenance.get("event_span_days"),
        "float_days": provenance.get("float_days"),
        "train_fraction": provenance.get("train_fraction", P1_TRAIN_FRACTION),
        "train_contract_by_string": provenance.get("train_contract_by_string", {}),
        "fault_plan_by_string": provenance.get("fault_plan_by_string", {}),
        "batch_ids_by_string": batch_ids_by_string,
        "n_strings": n_strings,
        "string_ids": string_ids_in_data,
        "n_batteries_per_string": N_BATTERIES,
        "n_events_per_string": max_events_per_string,
        "events_per_string": events_per_string,
        "n_event_battery_samples": n_truth,
        "battery_event_raw_rows": battery_count,
        "current_event_raw_rows": current_count,
        "continuous_float_raw_rows": float_count,
        "event_grid_points": EVENT_GRID_POINTS,
        "continuous_interval_s": 600,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": Path(db_path).name,

    }
    with (root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    conn.close()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GRAPE-UPS 论文实验模拟数据生成器 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", default="var/ups_simulation.sqlite3",
                        help="输出 SQLite 数据库路径")
    parser.add_argument("--events", type=int, default=30, help="事件数量 (默认: 30)")
    parser.add_argument("--days", type=int, default=15, help="时间跨度天数 (默认: 15)")
    parser.add_argument("--seed", type=int, default=20260725, help="随机种子 (默认: 20260725)")
    parser.add_argument("--clean", action="store_true", help="清空已有数据库后重新生成")
    parser.add_argument("--export", action="store_true",
                        help="同时导出为 ups_ai_pipeline/data/synthetic_v2/ 标准目录")
    parser.add_argument("--p1-smoke", action="store_true",
                        help="生成 P1 smoke 数据 (3 strings x 5 events x 40 batteries)")
    parser.add_argument("--p1-full", action="store_true",
                        help="生成 P1 全量数据 (3 strings x 30 events x 40 batteries + 30天浮充)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium",
                        help="难度等级 (默认: medium)")
    parser.add_argument("--float-days", type=int, default=0,
                        help="连续浮充监测天数 (默认: 0, --p1-full 时自动设为 30)")
    parser.add_argument("--packet-loss", type=float, default=-1.0,
                        help="事件采样丢包率 (默认: 按难度等级)")
    parser.add_argument(
        "--upload-latency-sigma",
        "--jitter",
        dest="upload_latency_sigma",
        type=float,
        default=-1.0,
        help="半正态上传时延 sigma，--jitter 为兼容别名 (默认: 按难度等级)",
    )
    parser.add_argument("--export-dir", default=None,
                        help="导出目录 (默认: ups_ai_pipeline/data/synthetic_v2)")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="覆盖导出 manifest 中的数据集名称；用于不覆盖正式数据的确认实验",
    )
    args = parser.parse_args()

    db_path = Path(__file__).parent / args.db

    if args.p1_smoke:
        # P1 smoke: 3 strings x 5 events x 40 batteries
        # 5 events 保证 S1/S2 有 train/validation/test_seen 三个分割
        strings = [
            {"string_id": "SIM-S1", "n_events": 5, "seed_offset": 0},
            {"string_id": "SIM-S2", "n_events": 5, "seed_offset": 1000},
            {"string_id": "SIM-S3", "n_events": 5, "seed_offset": 2000},
        ]
        print(f" GRAPE-UPS P1 SMOKE 数据生成")
        print(f"{'='*60}")
        print(f"  数据库: {db_path}")
        print(f"  串数:   {len(strings)} ({', '.join(s['string_id'] for s in strings)})")
        print(f"  事件:   每串 {strings[0]['n_events']} 事件")
        print(f"  电池:   {N_BATTERIES} 节/串")
        print(f"  种子:   {args.seed}")
        print(f"  固定放电时长: {FIXED_DURATION_S}s, 网格点数: {EVENT_GRID_POINTS}")
        print(f"{'='*60}")
        print()

        stats = generate_simulation(
            str(db_path), n_events=5, span_days=args.days,
            seed=args.seed, clean=True, strings=strings,
            difficulty="medium", float_days=0,
            packet_loss_rate=0.0, upload_latency_sigma_s=0.0,
        )

        print()
        print(f"{'='*60}")
        print(f" P1 SMOKE 生成完成")
        print(f"{'='*60}")
        print(f"  串数:           {stats['n_strings']}")
        print(f"  事件数:         {stats['events']}")
        print(f"  电池样本:       {stats['battery_samples']:,}")
        print(f"  电流样本:       {stats['current_samples']:,}")
        print(f"  环境样本:       {stats['environment_samples']:,}")
        print(f"  参考测量:       {stats['reference_measurements']:,}")
        print(f"  Ground truth:   {stats['truth_rows']:,}")
        print()

        if args.export or args.export_dir:
            export_name = "p1_smoke"
            if args.export_dir:
                pipeline_data = Path(args.export_dir)
            else:
                pipeline_data = Path(__file__).resolve().parents[1] / "data" / export_name
            print(f" 导出 {export_name} -> {pipeline_data}")
            manifest = export_synthetic_v2(str(db_path), str(pipeline_data), seed=args.seed,
                                           dataset_name=export_name, difficulty="medium",
                                           packet_loss_rate=0.0, upload_latency_sigma_s=0.0)
            print(f"  n_strings: {manifest['n_strings']}")
            print(f"  string_ids: {manifest.get('string_ids')}")
            print(f"  events_per_string: {manifest.get('events_per_string')}")
            print(f"  battery_event_raw_rows: {manifest['battery_event_raw_rows']}")
            print(f"  n_event_battery_samples: {manifest['n_event_battery_samples']}")
            print(f"  simulation_only: {manifest['simulation_only']}")
            print()
        return 0

    if args.p1_full:
        # P1 full: 3 strings x 30 events x 40 batteries + 30-day float monitoring
        n_ev = 30
        span = 30
        fdays = args.float_days if args.float_days > 0 else 30
        strings = [
            {"string_id": "SIM-S1", "n_events": n_ev, "seed_offset": 0},
            {"string_id": "SIM-S2", "n_events": n_ev, "seed_offset": 1000},
            {"string_id": "SIM-S3", "n_events": n_ev, "seed_offset": 2000},
        ]
        diff_cfg = DIFFICULTY_CONFIGS[args.difficulty]
        s1_fault_plan = get_fault_plan(args.difficulty, string_idx=0)

        print(f" GRAPE-UPS P1 全量数据生成")
        print(f"{'='*60}")
        print(f"  数据库: {db_path}")
        print(f"  串数:   {len(strings)} ({', '.join(s['string_id'] for s in strings)})")
        print(f"  事件:   每串 {n_ev} 事件")
        print(f"  电池:   {N_BATTERIES} 节/串")
        print(f"  跨度:   {span} 天 (每串覆盖完整跨度)")
        print(f"  浮充:   {fdays} 天 ({FLOAT_INTERVAL_S}s 间隔)")
        print(f"  难度:   {args.difficulty} (S1 故障 {len(s1_fault_plan)} 节, 每串独立旋转, onset×{diff_cfg['onset_multiplier']}, sev_cap={diff_cfg['severity_cap']})")
        print(f"  丢包:   {diff_cfg['packet_loss_rate']*100:.0f}% (序列间隙)")
        print(f"  上传延迟: half-normal sigma={diff_cfg['upload_latency_sigma_s']}s")
        print(
            f"  训练期: 事件 1-{p1_train_end_event(n_ev)} 无故障; "
            f"最早 onset={p1_min_fault_onset_event(n_ev)}"
        )
        print(f"  种子:   {args.seed}")
        print(f"  固定放电时长: {FIXED_DURATION_S}s, 网格点数: {EVENT_GRID_POINTS}")
        print(f"{'='*60}")
        print()

        stats = generate_simulation(
            str(db_path), n_events=n_ev, span_days=span,
            seed=args.seed, clean=True, strings=strings,
            difficulty=args.difficulty, float_days=fdays,
            packet_loss_rate=args.packet_loss,
            upload_latency_sigma_s=args.upload_latency_sigma,
        )

        print()
        print(f"{'='*60}")
        print(f" P1 全量数据生成完成 ({args.difficulty})")
        print(f"{'='*60}")
        print(f"  串数:           {stats['n_strings']}")
        print(f"  事件数:         {stats['events']}")
        print(f"  电池样本:       {stats['battery_samples']:,}")
        print(f"  电流样本:       {stats['current_samples']:,}")
        print(f"  环境样本:       {stats['environment_samples']:,}")
        print(f"  参考测量:       {stats['reference_measurements']:,}")
        print(f"  浮充样本:       {stats['float_samples']:,}")
        print(f"  Ground truth:   {stats['truth_rows']:,}")
        print(f"  故障电池:       {stats['fault_batteries']} 节")
        print()

        if args.export or args.export_dir:
            export_name = args.dataset_name or f"p1_full_{args.difficulty}"
            if args.export_dir:
                pipeline_data = Path(args.export_dir)
            else:
                pipeline_data = Path(__file__).resolve().parents[1] / "data" / export_name
            print(f" 导出 {export_name} -> {pipeline_data}")
            manifest = export_synthetic_v2(str(db_path), str(pipeline_data), seed=args.seed,
                                           dataset_name=export_name, difficulty=args.difficulty,
                                           packet_loss_rate=diff_cfg["packet_loss_rate"],
                                           upload_latency_sigma_s=diff_cfg["upload_latency_sigma_s"])
            print(f"  n_strings: {manifest['n_strings']}")
            print(f"  string_ids: {manifest.get('string_ids')}")
            print(f"  events_per_string: {manifest.get('events_per_string')}")
            print(f"  battery_event_raw_rows: {manifest['battery_event_raw_rows']:,}")
            print(f"  current_event_raw_rows: {manifest['current_event_raw_rows']:,}")
            print(f"  continuous_float_raw_rows: {manifest['continuous_float_raw_rows']:,}")
            print(f"  n_event_battery_samples: {manifest['n_event_battery_samples']:,}")
            print(f"  simulation_only: {manifest['simulation_only']}")
            print()
        return 0

    print(f" GRAPE-UPS 模拟数据生成器 v2 (P0 修正版)")
    print(f"{'='*60}")
    print(f"  数据库: {db_path}")
    print(f"  事件数: {args.events}")
    print(f"  跨度:   {args.days} 天")
    print(f"  种子:   {args.seed}")
    print(f"  电池:   {N_BATTERIES} 节")
    print(f"  故障:   {len(FAULT_PLAN_BASE)} 节 ({', '.join(f'B{k:02d}({v[0]})' for k, v in FAULT_PLAN_BASE.items())})")
    print(f"  固定放电时长: {FIXED_DURATION_S}s, 网格点数: {EVENT_GRID_POINTS}")
    print(f"{'='*60}")
    print()

    stats = generate_simulation(
        str(db_path), n_events=args.events, span_days=args.days,
        seed=args.seed, clean=args.clean,
        difficulty="p0",
    )

    print()
    print(f"{'='*60}")
    print(f" 生成完成")
    print(f"{'='*60}")
    print(f"  事件数:         {stats['events']}")
    print(f"  电池样本:       {stats['battery_samples']:,}")
    print(f"  电流样本:       {stats['current_samples']:,}")
    print(f"  环境样本:       {stats['environment_samples']:,}")
    print(f"  参考测量:       {stats['reference_measurements']:,}")
    print(f"  Ground truth:   {stats['truth_rows']:,}")
    print()

    if args.export:
        pipeline_data = Path(__file__).resolve().parents[1] / "data" / "synthetic_v2"
        print(f" 导出 synthetic_v2 -> {pipeline_data}")
        manifest = export_synthetic_v2(str(db_path), str(pipeline_data), seed=args.seed,
                                       dataset_name="synthetic_v2", difficulty="p0",
                                       packet_loss_rate=0.0, upload_latency_sigma_s=0.0)
        print(f"  battery_event_raw_rows: {manifest['battery_event_raw_rows']}")
        print(f"  current_event_raw_rows: {manifest['current_event_raw_rows']}")
        print(f"  n_event_battery_samples: {manifest['n_event_battery_samples']}")
        print(f"  event_grid_points: {manifest['event_grid_points']}")
        print(f"  simulation_only: {manifest['simulation_only']}")
        print()

    print(f"  数据库: {db_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
