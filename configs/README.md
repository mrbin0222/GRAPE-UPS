# Configuration

- `example.json`: ready-to-run example data/model pair; evaluate or retrain it.
- `train.json`: shared training and method settings; choose a realization with `--dataset`.
- `simulation/`: three distinct physical-generation protocols (four-domain, sixteen-domain and high-rate).
- `stress/`: four distinct experiment designs (mixed backgrounds, fault contamination, telemetry and sampling).

Difficulty, seed and data paths are dataset metadata in `data/datasets.json`. They do not require one config file per run. Frozen fusion weights are embedded in the two runnable configs; no historical weight-config versions are shipped. Stress files document experiment designs; only the example model is distributed, so they do not imply the original stress-transfer checkpoint is included.
