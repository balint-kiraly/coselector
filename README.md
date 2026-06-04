# coselector

Cooperative 3-D object detection with agent selection for V2X networks.

Built on top of [coperception](https://github.com/coperception/coperception) and the
[V2X-Sim 2.0](https://arxiv.org/abs/2202.08449) dataset.

---

## Overview

In a cloud-based V2X system, connected vehicles upload their LiDAR data to a Road-Side
Unit (RSU) which fuses the point clouds and runs 3-D object detection. When bandwidth
is limited, the RSU must decide *which vehicles* to request data from each frame.

This project implements and evaluates several agent-selection strategies:

| Strategy | Description |
|---|---|
| `identity` | All available vehicles (upper bound baseline) |
| `closest_k` | K vehicles nearest to the RSU |
| `velocity` | K fastest-moving vehicles |
| `heuristic` | Greedy angular-coverage maximisation |
| `bandwidth` | Greedy selection under a byte-budget constraint |
| `ml_model` | Learned MLP selector *(placeholder — not yet trained)* |

Each run measures detection quality (mAP@0.5, mAP@0.7) alongside a three-axis
communication cost (bandwidth MB, latency ms, energy J) so strategies can be
compared on a single normalised scale.

---

## Repository Layout

```
coselector/
├── preprocess/
│   ├── bev_precompute.py      # Raw LiDAR → sparse BEV .npy (adapted from coperception)
│   └── state_precompute.py    # GNSS + IMU → per-agent JSON state features
│
├── data_utils/
│   ├── state_index.py         # AgentMeta dataclass + StateIndex loader
│   ├── build_state_features.py # AgentMeta → (N, 14) feature tensor
│   └── cost_model.py          # Bandwidth / latency / energy cost model
│
├── selection/
│   ├── policy.py              # Selection strategies dispatcher
│   ├── bev_builder.py         # BEV fusion (planar Z-preserving transform for RSU)
│   ├── models.py              # AgentSelectorMLP (placeholder)
│   └── train_selector.py      # ML selector training (placeholder)
│
├── training/
│   ├── dataset_rsu_centric.py # Dataset for RSU-frame fine-tuning
│   └── train_rsu_centric.py   # Fine-tuning script
│
├── test_codet_selector.py     # Main evaluation script
├── configs/scene_split/       # train / val / test scene IDs
├── measurements/              # Calibrated cost constants (lidar size, precomp time)
├── Makefile
├── README.md                  # This file
└── DOCS.md                    # Detailed technical documentation
```

**coperception library** (at `/home/bkiraly/coperception`, installed as editable package,
used unmodified):

Two scripts in this project are extended copies of coperception originals:
- `preprocess/bev_precompute.py` — adapted from `tools/det/create_data_det.py`; adds
  idempotent completion tracking, `only_split` filter, and per-agent iteration control
- `test_codet_selector.py` — adapted from `tools/det/test_codet.py`; the original
  evaluation loop is preserved and extended with the full agent-selection pipeline,
  RSU-centric mode, cost measurement, and structured CSV/JSON logging

All other modules (`data_utils/`, `selection/`, `training/`) are written from scratch.

---

## Installation

Use a python 3.7 environment

### Install coperception
[Coperception installation guide](https://coperception.readthedocs.io/en/latest/getting_started/installation/)

```bash
# Install coselector dependencies
cd /home/bkiraly/coselector
pip install -r requirements.txt
```
Run options can be configured in `Makefile`

---

## Data Preparation

Run once. The script is idempotent — completed (agent, scene) pairs are tracked and
skipped on reruns.

```bash
# Preprocess all 100 scenes, all 6 agents
make create_data

# Preprocess only the val split (faster for development)
make create_data only_split=val

# Preprocess a subset of scenes
make create_data scene_begin=0 scene_end=10 only_split=val
```

This produces:
- `V2X-Sim-det/{train,val,test}/agent{0..5}/` — sparse BEV tensors
- `V2X-Sim-States/scene_NNN/frame_NNN/agent_NN.json` — GNSS + IMU + motion features

---

## Running Evaluations

All evaluation uses the selector target (`test_codet_selector`).
The RSU (agent 0) is the reference frame; GT labels come from agent 0's perspective.
Detection runs on a FaFNet model (`upperbound/no_rsu` checkpoint).

### All-agents baseline (identity)

```bash
make test_codet_selector sel_method=identity
```

This also auto-saves `measurements/inference_norms.json` so subsequent runs can
compute fully normalised combined costs without any extra flags.

### Selection strategies

```bash
# K-nearest agents
make test_codet_selector sel_method=closest_k K=3

# K fastest-moving agents
make test_codet_selector sel_method=velocity K=3

# Greedy angular coverage
make test_codet_selector sel_method=heuristic K=3

# Bandwidth-constrained (budget in MB)
make test_codet_selector sel_method=bandwidth budget_mb=2.0
```

### Quick test on a subset of scenes

```bash
# Scenes 5–9 only (fast sanity check)
make test_codet_selector sel_method=closest_k K=3 scene_begin=5 scene_end=10
```

---

## Output

Each run writes to a timestamped subdirectory under `results/`:

```
results/lowerbound_eval/with_rsu/
  summary.csv                 ← one row per run (mAP, avg cost, avg agents, …)
  runs/{timestamp}_{method}/
    frame_costs.csv           ← per-frame: bandwidth_MB, latency_ms, energy_J, cost
    scene_stats.csv           ← per-scene averages
    metadata.json             ← full provenance (checkpoint, args, norms, …)
```

`summary.csv` is append-only — all runs accumulate in one file for easy comparison.

Key columns in `summary.csv`:

| Column | Description |
|---|---|
| `mAP_05` / `mAP_07` | Detection mAP at IoU 0.5 / 0.7 |
| `avg_n_selected` | Mean vehicles selected per frame |
| `avg_bandwidth_MB` | Mean LiDAR upload per frame |
| `avg_latency_total_ms` | Mean end-to-end latency |
| `avg_combined_cost` | Normalised cost ∈ [0, 1] |

---

## Fine-tuning (optional)

The `upperbound/no_rsu` checkpoint was trained in vehicle-ego frame (agent 1 as origin).
Fine-tuning corrects this to RSU frame and makes the model robust to variable agent counts:

```bash
make train_rsu_centric
```

This warm-starts from `epoch_100.pth` and trains for up to 30 epochs with early stopping.
Checkpoints are saved to `checkpoints/rsu_centric/`.

---

## Technical Details

See [DOCS.md](DOCS.md) for:
- Module-by-module API reference
- The planar (Z-preserving) BEV transform for RSU-centric fusion
- Cost model calibration workflow
- Selection strategy implementation details
- Contributions to the coperception library
