#!/usr/bin/env python3
"""Generate an independent lead-acid physics suite without detector imports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
import pybamm  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "simulation/independent_four_domains.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "independent_physics_generated",
    )
    parser.add_argument("--seeds", nargs="*", type=int)
    return parser.parse_args()


def scaled_function(
    function: Callable[..., Any],
    multiplier: float,
) -> Callable[..., Any]:
    def wrapped(*args: Any) -> Any:
        return multiplier * function(*args)

    return wrapped


def apply_fault(
    parameters: pybamm.ParameterValues,
    family: str,
    multiplier: float,
) -> None:
    if family == "acid_inventory_loss":
        key = "Initial concentration in electrolyte [mol.m-3]"
        parameters.update({key: float(parameters[key]) * multiplier})
    elif family == "active_material_loss":
        keys = [
            "Negative electrode thickness [m]",
            "Positive electrode thickness [m]",
        ]
        parameters.update(
            {key: float(parameters[key]) * multiplier for key in keys}
        )
    elif family == "interfacial_kinetics_loss":
        keys = [
            "Negative electrode exchange-current density [A.m-2]",
            "Positive electrode exchange-current density [A.m-2]",
        ]
        updates = {
            key: scaled_function(parameters[key], multiplier) for key in keys
        }
        parameters.update(updates)
    elif family == "pore_blockage":
        keys = [
            "Maximum porosity of negative electrode",
            "Maximum porosity of positive electrode",
        ]
        parameters.update(
            {key: float(parameters[key]) * multiplier for key in keys}
        )
    elif family != "normal":
        raise ValueError(f"Unknown latent fault family: {family}")


def solve_profile(
    condition: dict[str, float],
    family: str,
    multiplier: float,
    event: dict[str, float],
) -> dict[str, np.ndarray]:
    model = pybamm.lead_acid.LOQS(options={"thermal": "lumped"})
    parameters = model.default_parameter_values.copy()
    parameters.update(
        {
            "Ambient temperature [K]": float(condition["ambient_k"]),
            "Initial temperature [K]": float(condition["ambient_k"]),
            "Initial State of Charge": float(condition["initial_soc"]),
        }
    )
    apply_fault(parameters, family, multiplier)
    current = float(condition["c_rate"]) * float(
        parameters["Nominal cell capacity [A.h]"]
    )
    experiment = pybamm.Experiment(
        [
            f"Rest for {event['pre_event_rest_s']} seconds",
            f"Discharge at {current} A for {event['discharge_s']} seconds",
            f"Rest for {event['post_event_rest_s']} seconds",
        ],
        period=f"{event['sample_period_s']} seconds",
    )
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameters,
        experiment=experiment,
        solver=pybamm.CasadiSolver(mode="safe"),
    )
    solution = simulation.solve()
    expected_end = (
        float(event["pre_event_rest_s"])
        + float(event["discharge_s"])
        + float(event["post_event_rest_s"])
    )
    if float(solution.t[-1]) < expected_end - 1e-6:
        raise RuntimeError(
            f"Profile terminated at {solution.t[-1]:.3f}s before "
            f"{expected_end:.3f}s: {family}, multiplier={multiplier}, "
            f"condition={condition}"
        )
    return {
        "time_s": np.asarray(solution.t, dtype=np.float32),
        "voltage_v": np.asarray(
            solution["Battery voltage [V]"].entries,
            dtype=np.float32,
        ),
        "temperature_k": np.asarray(
            solution["Volume-averaged cell temperature [K]"].entries,
            dtype=np.float32,
        ),
        "current_a": np.asarray(
            solution["Current [A]"].entries,
            dtype=np.float32,
        ),
    }


def build_library(config: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    event = config["generator"]["event"]
    families = config["latent_fault_families"]
    library: dict[str, dict[str, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    for domain, conditions in config["domains"].items():
        for condition_index, condition in enumerate(conditions):
            variants = [("normal", 1.0, 0)]
            for family, specification in families.items():
                variants.extend(
                    (family, float(multiplier), severity_index + 1)
                    for severity_index, multiplier in enumerate(
                        specification["multipliers"]
                    )
                )
            for family, multiplier, severity_index in variants:
                key = (
                    f"{domain}__c{condition_index}__{family}"
                    f"__s{severity_index}"
                )
                profile = solve_profile(
                    condition,
                    family,
                    multiplier,
                    event,
                )
                library[key] = profile
                records.append(
                    {
                        "profile_key": key,
                        "domain": domain,
                        "condition_index": condition_index,
                        "family": family,
                        "severity_index": severity_index,
                        "parameter_multiplier": multiplier,
                        **condition,
                    }
                )
    return library, records


def assemble_dataset(
    config: dict[str, Any],
    library: dict[str, dict[str, np.ndarray]],
    domain: str,
    role: str,
    seed: int,
) -> dict[str, np.ndarray]:
    sampling = config["sampling"]
    acquisition = config["acquisition_variation"]
    n_units = int(sampling["n_units"])
    n_events = int(sampling["n_events_per_domain"])
    n_fault_units = int(sampling["n_fault_units_per_event"])
    conditions = config["domains"][domain]
    families = list(config["latent_fault_families"])
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, list(config["domains"]).index(domain), 0 if role == "validation" else 1])
    )
    first = next(iter(library.values()))
    n_time = len(first["time_s"])
    n_rows = n_units * n_events
    voltage = np.empty((n_rows, n_time), dtype=np.float32)
    temperature = np.empty((n_rows, n_time), dtype=np.float32)
    current = np.empty((n_rows, n_time), dtype=np.float32)
    event_index = np.repeat(np.arange(n_events, dtype=np.int16), n_units)
    unit_index = np.tile(np.arange(n_units, dtype=np.int16), n_events)
    family_output = np.full(n_rows, "normal", dtype="<U32")
    severity_output = np.zeros(n_rows, dtype=np.int8)
    condition_output = np.empty(n_rows, dtype=np.int8)
    label = np.zeros(n_rows, dtype=np.int8)
    voltage_offsets = rng.normal(
        0.0,
        float(acquisition["unit_voltage_offset_sd_v"]),
        n_units,
    )
    temperature_offsets = rng.normal(
        0.0,
        float(acquisition["unit_temperature_offset_sd_k"]),
        n_units,
    )
    cursor = 0
    for event_index_value in range(n_events):
        condition_index = event_index_value % len(conditions)
        family = families[event_index_value % len(families)]
        severity_index = (event_index_value // len(families)) % 3 + 1
        targets = set(
            rng.choice(n_units, size=n_fault_units, replace=False).tolist()
        )
        common_shift = rng.normal(
            0.0,
            float(acquisition["event_common_voltage_shift_sd_v"]),
        )
        for unit in range(n_units):
            active_family = family if unit in targets else "normal"
            active_severity = severity_index if unit in targets else 0
            key = (
                f"{domain}__c{condition_index}__{active_family}"
                f"__s{active_severity}"
            )
            profile = library[key]
            voltage[cursor] = (
                profile["voltage_v"]
                + voltage_offsets[unit]
                + common_shift
                + rng.normal(
                    0.0,
                    float(acquisition["voltage_noise_sd_v"]),
                    n_time,
                )
            )
            temperature[cursor] = (
                profile["temperature_k"]
                + temperature_offsets[unit]
                + rng.normal(
                    0.0,
                    float(acquisition["temperature_noise_sd_k"]),
                    n_time,
                )
            )
            current[cursor] = profile["current_a"]
            family_output[cursor] = active_family
            severity_output[cursor] = active_severity
            condition_output[cursor] = condition_index
            label[cursor] = int(unit in targets)
            cursor += 1
    return {
        "time_s": first["time_s"],
        "voltage_v": voltage,
        "temperature_k": temperature,
        "current_a": current,
        "event_index": event_index,
        "unit_index": unit_index,
        "condition_index": condition_output,
        "fault_family": family_output,
        "severity_index": severity_output,
        "label_anomaly": label,
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds or config["sampling"]["development_seeds"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    library, library_records = build_library(config)
    np.savez_compressed(
        args.output_dir / "profile_library.npz",
        **{
            f"{key}__{name}": value
            for key, profile in library.items()
            for name, value in profile.items()
        },
    )
    (args.output_dir / "profile_library_manifest.json").write_text(
        json.dumps(library_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    dataset_files: list[str] = []
    for seed in seeds:
        for domain in config["domains"]:
            for role in config["sampling"]["roles"]:
                dataset = assemble_dataset(
                    config,
                    library,
                    domain,
                    role,
                    int(seed),
                )
                filename = f"{role}__{domain}__s{seed}.npz"
                np.savez_compressed(args.output_dir / filename, **dataset)
                dataset_files.append(filename)
                print(filename, flush=True)
    manifest = {
        "contract": config["contract"],
        "status": config["status"],
        "pybamm_version": pybamm.__version__,
        "seeds": seeds,
        "n_library_profiles": len(library_records),
        "dataset_files": dataset_files,
        "separation_contract": config["separation_contract"],
    }
    (args.output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
