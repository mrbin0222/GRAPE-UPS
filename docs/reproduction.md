# Using the experiments

| Experiment | Inputs | Starting point |
|---|---|---|
| New confirmation / GRAPE-UPS | `data/main_benchmark/confirmation/` | Example: `configs/example.json`; other datasets: train with `configs/train.json --dataset ...` |
| Original structural analyses | `data/main_benchmark/original/` | Train with the common config; keep this population separate from the new confirmation data |
| Independent four-domain benchmark | `data/physical/independent_four_domains`, `configs/simulation/independent_four_domains.json` | `generate_independent_pybamm_suite.py --config ... --output-dir ...` |
| Sixteen physical domains | `data/physical/sixteen_domains`, `configs/simulation/physical_domains.json` | `generate_physical_domain.py`; raw directed statistic via `models._directed_group_score` |
| Sampling / telemetry / background stress | `data/physical/high_rate`, `data/stress` and corresponding configs | Input arrays and the original telemetry perturbation function in `ups_ai.telemetry`; choose the score/reference specified in the paper |
| Label availability | `data/development/label_selection/`, `data/development/label_composition/` and original-five inputs | Rerun validation selection on the requested label subsets; never select using held-out confirmation labels |
| Online resources | Example model and one locally prepared 40-cell event | Measure preprocessing and inference on the stated hardware; training/setup are separate phases |
| Field sampling | `data/field` | Compute intervals per cell and shared-current channel using the common relative clock |

The release includes the proposed method and evaluation primitives; it does not bundle implementations or outputs for the other 18 methods. It therefore does not reproduce the complete 19-method table with one command. The specialized sensitivity experiments also require assembling their inputs according to the stated protocols; they are not represented as an already tested one-command workflow.

One example set of final weights permits re-evaluation of its matching easy realization without training. Other archived model instances are not distributed. Prepared preprocessing metadata and component-normalization parameters belong to the model, not to the withheld results. Fresh GRAPE-only training uses the same architecture and objective but can differ from archived joint-run weights because initialization/random-stream consumption and software versions differ.

The ranking score is always the normalized directed component in the main method. Where the paper explicitly uses raw g, omit component normalization and clipping. Do not compare values or thresholds across these two variants without matching the protocol.
