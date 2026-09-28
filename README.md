# GRAPE-UPS

Research code and datasets for **GRAPE-UPS: Direction-Constrained Peer Anomaly Detection for Weak Batteries in Data-Center UPS Strings**.

The associated manuscript is currently **under review**.

GRAPE-UPS ranks batteries from direction-constrained differences within a shared load event. Two reconstruction branches provide an optional validation-selected alert score. The repository contains the proposed method, synthetic-data generators, de-identified field observations, one example set of final model weights, and evaluation tools. Comparator implementations and precomputed evaluation results are not included.

## Install

Use Python 3.10 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-tested.txt
```

For physical simulations, use a separate environment with `requirements-physics.txt`. The original physical-domain runs used PyBaMM 26.6.2.0. See `docs/data.md` for the distinction between the LOQS and Full-model experiments.

## Data and model downloads

Download the data and example model from [Release v0.1.0](https://github.com/mrbin0222/GRAPE-UPS/releases/tag/v0.1.0):

- [Data](https://github.com/mrbin0222/GRAPE-UPS/releases/download/v0.1.0/grape-ups-data.zip)
- [Example model](https://github.com/mrbin0222/GRAPE-UPS/releases/download/v0.1.0/grape-ups-models.zip)

Extract both archives into the repository root to create `data/` and `models/`. They are distributed separately from the code to keep Git history small. The repository is currently private; access is limited to authorized users.

## Run a released model

From the repository root:

```bash
python scripts/run_grape.py \
  --config configs/example.json \
  --mode evaluate --output outputs/example
```

The command loads the example GRAPE-UPS weights, reconstructs preprocessing from the released healthy training inputs, selects the alert head and threshold using validation labels, and evaluates held-out data. Outputs are created locally. No published result files are bundled.

`raw_g` is the unnormalized directed statistic. `peer_score` is its nonnegative, healthy-training-normalized component. Ranking uses `peer_score`; the alert pathway selects `peer_score` or the fused score. These scores can differ in ties and therefore in AUPRC.

Only one example model is distributed. It is matched to the easy dataset at seed 2026092301 and is not a universal pretrained model for all conditions.

## Train and generate data

```bash
python scripts/run_grape.py \
  --config configs/example.json \
  --mode train --output outputs/retrained
```

This trains only the two GRAPE-UPS branches and saves final weights. No intermediate checkpoints are retained. The standalone training entry has its own seeded random stream; removing the historical joint comparator training changes the random-number consumption, so retraining is not advertised as bitwise reconstruction of the archived weights. The example weights re-evaluate one easy confirmation realization (seed 2026092301), not all paper runs.

To train another released realization with the common settings:

```bash
python scripts/run_grape.py --config configs/train.json \
  --dataset revision_x05_medium_s2026092301 \
  --mode train --output outputs/medium
```

The dataset catalog supplies the difficulty, seed and data path; no separate config is needed for each realization. To evaluate this newly trained model, use the same config/dataset and add `--mode evaluate --model-dir outputs/medium/models`.

Generate a new primary dataset (use a new output directory):

```bash
python scripts/generate_primary.py --p1-full --difficulty easy \
  --seed 2026092301 --float-days 1 --export \
  --dataset-name example_easy --db outputs/example_easy.sqlite3 \
  --export-dir data/example_easy
```

Generate one physical-domain profile:

```bash
python scripts/generate_physical_domain.py \
  --seed 2026092401 --domain nominal --variant 0 \
  --output outputs/physical_profiles
```

Pass `--config configs/simulation/high_rate.json` for the high-rate sampling protocol. The profile generators save observations and status files; incomplete profiles must not be treated as complete events.

## Evaluate your own scores

```bash
python scripts/evaluate_scores.py outputs/example/scores.csv \
  --score selected_alert_score --output outputs/example/metrics_check.json
```

Input scores must be at the battery-event level. For other methods, supply your own implementation and aggregate its output to this same unit before evaluation. Thresholds must be chosen independently of held-out test labels. `scripts/paired_statistics.py` accepts independently replicated, paired metric values for statistical comparisons.

For a new development study, `scripts/select_configuration.py` selects the five-component fusion weights from development validation scores using mean AUPRC minus 0.15 times its standard deviation. The published fusion coefficients remain the default; this does not imply that all trained model instances are distributed. Trajectory-budget, composition and accumulation-selection routines are in `ups_ai.selection`.

## Files and documentation

- `src/ups_ai/`: feature extraction, GRAPE-UPS architecture/scoring, evaluation and observation perturbations.
- `scripts/`: portable generation, training and evaluation commands.
- `configs/`: experiment parameters and frozen GRAPE-UPS settings.
- `data/`: simulated measurements, physical profiles, stress-test inputs and de-identified field observations.
- Feature inputs are generated on first use into `.cache/`; they are not distributed.
- `models/example/`: one matched pair of final contextual/absolute weights, with preprocessing and calibration parameters.
- `docs/data.md`: dataset roles, labels and limitations.
- `docs/reproduction.md`: experiment-to-data/command guide.

## Citation and license

If you use this code or data in your research, please cite this GitHub repository:

```bibtex
@misc{luo2026grapeups,
  author={Luo, Shibo and Shen, Bin and Chen, Xiaoning and Fang, Yiquan and Wang, Huifeng},
  title={{GRAPE-UPS}: Direction-Constrained Peer Anomaly Detection for Weak Batteries in Data-Center {UPS} Strings},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/mrbin0222/GRAPE-UPS},
  note={Associated manuscript under review}
}
```

The citation will be updated to the journal article after publication.

Code, configurations, documentation and model weights are licensed under [Apache-2.0](LICENSE). Datasets are licensed under [CC BY 4.0](data/LICENSE). Third-party dependencies retain their respective licenses. Please credit the GRAPE-UPS research team and cite the associated manuscript when reusing the data; indicate any changes.
