# Data

## Primary simulated benchmark

`p1_confirm_v4_task_heads_*` contains the original five realizations per difficulty (seeds 20260811–20260815). `revision_x05_*` contains the 15 new confirmation realizations per difficulty (seeds 2026092301–2026092315). Easy, medium and hard remain separate tasks even when they share a seed number.

Each realization contains three 40-cell strings and 30 events per string. S1 provides healthy-reference training inputs, S2 provides validation, and S3 is held out. Use the released `split` fields instead of making a random row split. Event labels are simulated ground truth, not labels from the field recording.

The `events/` files contain voltage, temperature, shared current, observation times and event metadata. `assets/` maps simulated positions and devices. `reference/` contains simulated reference measurements. `continuous/` contains the ancillary float stream. The original sensor identifiers inherited by the simulator have been replaced by deterministic synthetic identifiers; numerical measurements and within-string battery ordering are retained.

Feature extraction runs automatically from the selected raw dataset. Generated features, sequences and feature definitions are cached locally in `.cache/<dataset>/` and are not included in the release. Set `GRAPE_CACHE_DIR` to use another cache location. Rows and sequences must remain aligned.

## Independent physical data

`independent_physics/` contains the older independent LOQS benchmark. `physical_domains/` contains the Full-model 16-domain observations and their physical parameters. `high_rate/` contains Full-model profiles for sampling-rate experiments. `event.npz` is an event interval; `history.npz` includes the preceding conditioning history. `requested.json` supplies event time origin, conditions and fault parameters. The arrays are time_s, voltage_v, temperature_k and current_a.

Some severe combinations terminated early. Partial histories are retained as observations, not successful full-duration examples. Consult the profile inventory before admitting a domain to a performance evaluation. The two incomplete domains must not acquire inferred detection results through padding or interpolation.

## Stress and selection data

`stress/` contains model inputs and labels for mixed backgrounds, telemetry, fault contamination and sampling. Files that originally combined inputs and scores were exported with input arrays only; prediction, score, metric and calibrated-output arrays are excluded. Synthetic labels and physical parameters remain because they define the experiment.

`revision_x06*` datasets are dedicated selection-development data. They are separate from the confirmation realizations and should not be pooled into the reported confirmation sample.

## Field observations

See `data/field/README.md`. Absolute calendar times, original sensor identities and site identifiers are removed. One shared relative clock preserves cross-channel alignment, timestamp anomalies and missing intervals. The files contain no independent capacity/health labels and do not support a field detection-accuracy claim.

## Units and missing values

Column suffixes identify units: V, A, degrees C, K, seconds, mOhm and dBm. Empty numeric CSV fields and NaN array entries represent missing values; they are not zero. Metadata distinguish generated records from the field recording. No interpolation is applied to the released raw field data.
