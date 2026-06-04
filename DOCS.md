# coselector — Technical Documentation

## 1. Project Overview

**coselector** is a cloud-based V2X cooperative perception system built on top of the
[coperception](https://github.com/coperception/coperception) library.
The core research question is:

> Which subset of connected vehicles should the Road-Side Unit (RSU) request data from
> in order to maximize detection quality while minimizing communication cost?

The system extends the original coperception detection pipeline with:

- An **RSU-centric evaluation mode** where the RSU is the fixed reference frame
- An **agent-selection layer** that filters the vehicle set per frame before fusion
- A **three-axis cost model** (bandwidth, latency, energy) to compare selection strategies
- A **fine-tuning pipeline** for training a model robust to variable-density merged BEVs

Dataset: **V2X-Sim 2.0** (NuScenes format). 100 urban intersection scenes, 6 agents per
scene (max) — agent 0 is a stationary RSU, agents 1–5 are vehicles.

---

## 2. System Architecture

```
Raw V2X-Sim-2                       Preprocessed Data
─────────────                       ─────────────────────────────────────────────
NuScenes format   ──bev_precompute──▶  V2X-Sim-det/{train,val,test}/agent{0..5}/
(LIDAR, GNSS,                           sparse BEV .npy files
 IMU, annotations)
                  ──state_precompute──▶  V2X-Sim-States/scene_NNN/frame_NNN/
                                          agent_NN.json  (GNSS + IMU + motion)

                              ▼
                    test_codet_selector.py
                    ┌─────────────────────────────────────────────────────┐
                    │                                                     │
                    │  DataLoader (V2XSimDet)                             │
                    │       ↓ per frame                                   │
                    │  StateIndex.get_agents_meta()  ───▶  AgentMeta[]   │
                    │       ↓                              (14 features)  │
                    │  build_state_features()         ───▶  (N, 14) feat │
                    │       ↓                                             │
                    │  select_agents_from_metadata()  ─── policy.py      │
                    │       ↓  selected indices                           │
                    │  assemble_detection_inputs()    ─── bev_builder.py │
                    │       planar BEV fusion into RSU frame              │
                    │       ↓                                             │
                    │  FaFNet(num_agent=1).forward()  ─── coperception   │
                    │       ↓                                             │
                    │  eval_map (mAP@0.5, mAP@0.7)                       │
                    │  compute_cost()  ─── cost_model.py                 │
                    │       ↓                                             │
                    │  CSV logging: frame_costs.csv                      │
                    │               scene_stats.csv                      │
                    │               summary.csv                          │
                    │               metadata.json                        │
                    └─────────────────────────────────────────────────────┘
```

---

## 3. Relationship to the coperception Library

The `coperception/` package is used **unmodified** as an installed dependency.
No files inside the coperception directory were changed.

The project uses these coperception components directly:

| Component | Used for |
|---|---|
| `V2XSimDet` dataset | Loading preprocessed sparse BEV tensors for all agents |
| `FaFNet` model | 3-D object detection on the fused BEV |
| `FaFModule` / `CoDetModule` | Loss computation and prediction postprocessing |
| `eval_map` | mAP@0.5 and mAP@0.7 evaluation |
| `late_fusion` | Box-level post-detection fusion (non-selection mode) |
| `voxelize_occupy` | Voxelization in `bev_builder.py` |

### Scripts adapted from coperception originals

Two scripts in this project are extended copies of coperception scripts.
The originals were not modified — the extensions live only inside `coselector/`.

#### `preprocess/bev_precompute.py`

Based on `coperception/tools/det/create_data_det.py`. Additions:

| Addition | Purpose |
|---|---|
| `bev_completed.json` completion tracker | Idempotent reruns: each (agent, scene) pair is marked done atomically so interrupted runs can resume |
| `only_split` argument | Skip scenes not belonging to a requested train/val/test split |
| `--from_agent / --to_agent` arguments | Allow preprocessing a subset of agents (e.g. only val agents) |

The core LiDAR voxelization and BEV-generation logic is unchanged from the original.

#### `test_codet_selector.py`

Based on `coperception/tools/det/test_codet.py`. All additions are new code layered on top
of the original evaluation loop — the original loop structure and mAP computation are preserved:

| Addition | Purpose |
|---|---|
| `--selection` flag + agent selection pipeline | Per-frame vehicle filtering before BEV fusion |
| RSU-centric evaluation (`--rsu 1`) | GT from agent 0; planar BEV fusion into RSU frame |
| `--scene_begin / --scene_end` + `Subset` | Scene-range filtering without loading out-of-range frames |
| Manager cache → plain `dict` | Eliminates IPC overhead from `multiprocessing.Manager` proxy lookups |
| `StateIndex.from_fs(scene_ids=…)` | Only loads JSON state features for evaluated scenes |
| `_timed_forward()` with `cuda.synchronize()` | GPU-accurate inference timing |
| Run-level `codecarbon` energy tracker | One tracker for the whole run; per-frame energy distributed proportionally to inference time |
| Inference norm auto-save | Identity runs write `measurements/inference_norms.json` for reuse in subsequent runs |
| Structured CSV/JSON output | `frame_costs.csv`, `scene_stats.csv`, `summary.csv`, `metadata.json` per run |

---

## 4. Module Reference

### 4.1 `preprocess/bev_precompute.py`

Converts raw NuScenes-format LiDAR sweeps into sparse BEV voxel arrays.

**Output structure:**
```
V2X-Sim-det/
  {train,val,test}/
    agent0/
      {scene}_{frame}/
        0.npy   ← sparse dict with keys: voxel_indices_0, gt_max_iou,
                                          reg_target_sparse, label_sparse,
                                          trans_matrices, …
    agent1/ … agent5/
```

Key functions:
- `create_data()` — outer loop over scenes; calls `convert_to_dense_bev` then `convert_to_sparse_bev`
- `convert_to_dense_bev()` — voxelizes point clouds, builds GT bounding-box annotations
- `convert_to_sparse_bev()` — compresses the dense BEV into index arrays for disk storage

### 4.2 `preprocess/state_precompute.py`

Reads GNSS + IMU sensor data from the raw NuScenes dataset and writes one JSON file
per agent per frame.

**Output structure:**
```
V2X-Sim-States/
  scene_000/
    frame_000/
      agent_00.json  ← { x, y, z, yaw, vx, vy, speed, yaw_rate, ax, ay, az, gx, gy, gz, … }
      agent_01.json
      …
    frame_001/
```

Motion (`vx`, `vy`, `yaw_rate`) is estimated by finite-differencing consecutive GNSS samples.

### 4.3 `data_utils/state_index.py`

**`AgentMeta`** — dataclass holding all per-agent state fields for one (scene, frame):
`x, y, z, yaw, vx, vy, speed, yaw_rate, ax, ay, az, gx, gy, gz` plus lidar spec constants.

**`StateIndex`** — two-mode loader:

| Mode | When used |
|---|---|
| `StateIndex(dataroot=…)` + `build_index()` | Builds the index on-the-fly from raw NuScenes (used during `state_precompute.py`) |
| `StateIndex.from_fs(root, scene_ids=…)` | Loads pre-written JSON tree (used during evaluation); accepts a `scene_ids` filter to avoid reading thousands of files for out-of-range scenes |

### 4.4 `data_utils/build_state_features.py`

**`build_state_features(state_index, scene_id, frame_id)`**

Returns an `(N_agents, 14)` float tensor of `[x, y, z, yaw, vx, vy, speed, yaw_rate, ax, ay, az, gx, gy, gz]`
plus an `AgentMeta` list in the same order. Used as input to selection policies.

### 4.5 `data_utils/cost_model.py`

Three-axis communication cost model for evaluating selection strategies.

**Axes:**

| Axis | Formula |
|---|---|
| Bandwidth | `N_selected × lidar_size_MB` |
| Latency | `upload_ms + N × precomp_ms + inference_ms` (upload is parallel → one agent's time) |
| Energy | `N × precomp_energy_J + inference_energy_J` |

**Combined cost (normalised):**
```
combined = (α_bw × BW/BW_max + α_lat × Lat/Lat_max + α_energy × E/E_max)
           / (α_bw + α_lat + α_energy)
```
All three terms are capped to [0, 1] so α weights are directly interpretable.

**Calibration workflow:**
1. Run `make test_codet_selector sel_method=identity` (all-agents baseline).
2. The script auto-writes `measurements/inference_norms.json` with `avg_inference_ms`.
3. All subsequent runs load the norm automatically — no `--max_inference_ms` needed.

**Measured constants** (from `tools/measure_lidar_size.py` and `tools/measure_precomp_cost.py`):

| Constant | Value | Source |
|---|---|---|
| `lidar_size_MB` | 0.4934 MB | `measurements/lidar_size.json` |
| `precomp_per_agent_ms` | 1.982 ms | `measurements/precomp_cost.json` |
| `precomp_energy_J` | 0.035252 J | `measurements/precomp_cost.json` |

### 4.6 `selection/policy.py`

Agent selection dispatcher and individual strategy implementations.

**`SelectionMethod` enum:**

| Value | Description | Status |
|---|---|---|
| `identity` | Keep all available agents | ✅ Implemented |
| `closest_k` | K nearest agents (Euclidean distance in RSU frame) | ✅ Implemented |
| `velocity` | K fastest-moving agents | ✅ Implemented |
| `heuristic` | Greedy max-angular-coverage around RSU | ✅ Implemented |
| `bandwidth` | Greedy under a byte-budget constraint | ✅ Implemented |
| `ml_model` | Learned MLP selector | ⚠️ Placeholder |

> **Note:** `ml_model` falls back to `identity` when no model checkpoint is provided.
> `selection/models.py` (AgentSelectorMLP) and `selection/train_selector.py` are
> placeholder stubs — the learned selector is not yet trained.

**Strategy details:**

- **`closest_k`**: Reads 2-D RSU-frame positions from `T_{j→RSU}` translation column;
  sorts by Euclidean distance; returns the K closest.

- **`velocity`**: Reads `state_features[:, 6]` (speed field); returns K highest-speed agents.
  Rationale: fast-moving vehicles observe the most scene change per frame.

- **`heuristic`**: Greedy angular-coverage maximisation. Picks the closest agent first,
  then iteratively picks the remaining agent whose bearing from the RSU is farthest
  (max-min angular gap) from already-selected bearings.

- **`bandwidth`**: Greedy inclusion by ascending BEV byte size until budget exhausted.
  Falls back to distance proxy when BEV list is unavailable.

### 4.7 `selection/bev_builder.py`

**`assemble_detection_inputs()`** — fuses selected agents' BEVs into a single tensor
ready for `FaFNet(num_agent=1)`.

The fusion path depends on what data is available:

```
Raw point cloud available  → _voxelize_points()        (full 3-D transform + voxelize)
Teacher BEV available      → _warp_and_merge_bevs()    (affine BEV warp)
Pre-voxelized BEV only     → _bev_to_pointcloud()      (BEV → pseudo-points)
                             + _planar_voxelize_points() (Z-preserving transform)
```

**The planar (Z-preserving) transform** is the key contribution for RSU-centric mode:

The RSU is mounted ~5.5 m above road level. A full rigid 3-D transform T_{j→RSU} applies
a −5.5 m Z-shift that pushes all vehicle voxels (Z ∈ [−3, +2] m) below the grid floor
(Z ∈ [−8.5, −3.5] m), producing an empty BEV.

Fix: apply T for X and Y only; restore the original vehicle-frame Z.

```python
z_orig = pts[:, 2].clone()          # save vehicle-frame Z
pts_ref = (T @ homo.t()).t()        # apply full rigid transform
pts_ref[:, 2] = z_orig              # restore Z
```

This is valid because:
1. V2X-Sim vehicles and RSU are roughly level → Z does not cross-contaminate transformed X/Y.
2. GT annotations are 2-D (no Z label) → absolute Z reference does not affect supervision.
3. Same transform is used at train and test time → distributions match.

After planar fusion, a `rot90(k=3)` is applied to match the storage convention used by
`V2XSimDet` (which applies `rot90(k=3)` to every BEV before saving).

### 4.8 `training/dataset_rsu_centric.py`

RSU-centric dataset for fine-tuning. Per training step:

1. Loads RSU-frame GT labels from `agent0/` (supervision target).
2. Loads each vehicle's own BEV (in its local frame) from `agentJ/`.
3. Transforms each vehicle BEV to RSU frame using the planar transform.
4. Randomly selects k vehicles, k ~ U[`min_agents`, `max_agents`].
5. Merges the k BEVs into a single RSU-frame tensor.

This trains the model to detect robustly regardless of how many agents are selected,
which is exactly what the selection strategies require at inference time.

### 4.9 `training/train_rsu_centric.py`

Fine-tuning script. Warm-starts from `upperbound/no_rsu` (5-vehicle merged, no RSU lidar)
and fine-tunes with:
- Mixed-precision AMP (`--amp`) — ~1.5× speed, −40% VRAM
- Early stopping with configurable patience
- Per-epoch checkpoint saving (safe for resume)
- 1e-4 learning rate (10× lower than scratch to avoid catastrophic forgetting)

The correct warm-start is `upperbound/no_rsu` (not `with_rsu`) because the RSU lidar
will never be present at inference. The frame geometry correction (agent1-frame → RSU-frame)
converges in approximately 10 epochs.

### 4.10 `test_codet_selector.py`

Main evaluation script (adapted from coperception's `tools/det/test_codet.py`).

Key additions over the original:

| Feature | Description |
|---|---|
| `--selection` flag | Activates agent selection mode; sets `num_agent=1` for FaFNet |
| `--rsu 1` | RSU is reference frame; GT evaluated from agent-0 perspective |
| `--scene_begin/--scene_end` | Restrict evaluation to a scene range |
| Dataset Subset optimization | Wraps `V2XSimDet` in `torch.utils.data.Subset` to avoid loading out-of-range frames |
| Manager cache replacement | Replaces `multiprocessing.Manager` proxy caches with plain `dict` (eliminates IPC overhead per frame) |
| `StateIndex.from_fs(..., scene_ids=…)` | Only loads JSON for the evaluated scenes |
| Run-level energy tracker | Single `codecarbon.EmissionsTracker` for the whole run; per-frame energy distributed proportionally to inference time (avoids 3 s/frame blocking stop) |
| `_timed_forward()` | GPU-synchronized inference timing via `torch.cuda.synchronize()` |
| Inference norm auto-save | Identity runs write `measurements/inference_norms.json` for reuse |
| CSV outputs | `frame_costs.csv`, `scene_stats.csv`, `summary.csv`, `metadata.json` per run |

**RSU-centric evaluation mode** (`--rsu 1 --selection`):
- The RSU GT (`gt_max_iou` from agent 0) is saved before agent filtering.
- Selected vehicle BEVs are fused into RSU frame using `assemble_detection_inputs`.
- RSU GT labels are prepended to the supervision list so the detector's reference
  frame labels come from agent 0.
- Detection results are evaluated against RSU GT, not the selected vehicles' GT.

---

## 5. Data Flow

### 5.1 Preprocessing (one-time)

```
make create_data [scene_begin=X scene_end=Y] [only_split=val]
```

1. `bev_precompute.py` reads LiDAR sweeps from `V2X-Sim-2/` and writes sparse `.npy`
   files to `V2X-Sim-det/{train,val,test}/agentN/`. One file per (scene, frame, agent).
2. `state_precompute.py` reads GNSS + IMU from the same dataset and writes JSON files
   to `V2X-Sim-States/scene_NNN/frame_NNN/agent_NN.json`.

Both scripts are idempotent: completed work is tracked in `bev_completed.json` /
`state_completed.json` and skipped on reruns.

### 5.2 Evaluation (per-run)

```
make test_codet_selector sel_method=<method> [K=3] [scene_begin=X scene_end=Y]
```

Per frame:
1. `V2XSimDet` DataLoader yields sparse BEV tensors for all 6 agents.
2. `StateIndex` returns `AgentMeta[]` for (scene_id, frame_id).
3. `build_state_features()` converts metadata → (N, 14) tensor.
4. `select_agents_from_metadata()` returns selected agent indices.
5. Per-agent BEVs and labels are filtered to the selected set.
6. `assemble_detection_inputs()` fuses selected BEVs into a single RSU-frame tensor.
7. `FaFNet(num_agent=1)` runs detection on the fused BEV.
8. `cal_local_mAP()` accumulates detection results for RSU-frame GT.
9. `compute_cost()` records bandwidth, latency, energy for this frame.

After all frames:
10. `eval_map()` computes mAP@0.5 and mAP@0.7.
11. Run-level energy is distributed proportionally across frames.
12. Results are written to structured CSV/JSON files.

### 5.3 Output Layout

```
results/
  lowerbound_eval/with_rsu/
    log_test.txt
    summary.csv              ← one row per run; append mode across runs
    runs/
      {timestamp}_{flag}_{method}_{split}/
        frame_costs.csv      ← per-frame bandwidth, latency, energy, combined_cost
        scene_stats.csv      ← per-scene averages
        metadata.json        ← full provenance (checkpoint, args, norm key, …)
```

---

## 6. Checkpoint Structure

```
checkpoints/
  upperbound/
    no_rsu/epoch_100.pth    ← agents 1-5 all merged (no RSU lidar); agent1 frame
    with_rsu/epoch_100.pth  ← agents 0-5 all merged (RSU lidar included); RSU frame
  lowerbound/
    no_rsu/epoch_100.pth    ← each agent detects in its own frame; no fusion
    with_rsu/epoch_100.pth  ← same, but agent0 (RSU) is included
  rsu_centric/              ← output of train_rsu_centric (fine-tuned)
    epoch_N.pth
```

**For `test_codet_selector`:**  `upperbound/no_rsu` is used.

Rationale: this checkpoint was trained on dense merged vehicle BEVs (closest density match
to what the RSU receives from all 5 vehicles). The RSU lidar is excluded because the RSU
is a passive relay in Option B — it only receives and fuses, never senses.

---

## 7. Selection Strategy Summary

| Strategy | Key parameter | Complexity | Thesis role |
|---|---|---|---|
| `identity` | — | O(N) | All-agents baseline |
| `closest_k` | K | O(N log N) | Distance heuristic |
| `velocity` | K | O(N log N) | Motion heuristic |
| `heuristic` | K | O(K·N) | Angular coverage |
| `bandwidth` | `budget_mb` | O(N log N) | Bandwidth-constrained |
| `ml_model` | checkpoint | O(N) | Learned (placeholder) |

---

## 8. Cost Model Calibration

The combined cost requires a normalisation denominator for the latency and energy axes.
This denominator is the all-agents (identity) baseline cost.

**Automatic workflow:**

```bash
# Step 1: run identity baseline (all agents, all val scenes)
make test_codet_selector sel_method=identity scene_begin=0 scene_end=100

# Step 2: measurements/inference_norms.json is auto-updated with avg_inference_ms.
#         All subsequent runs load it automatically.

# Step 3: compare strategies — combined_cost is now fully normalised
make test_codet_selector sel_method=closest_k K=3
make test_codet_selector sel_method=heuristic K=3
make test_codet_selector sel_method=bandwidth budget_mb=2.0
```

**Manual override** (if needed):
```bash
make test_codet_selector max_inference_ms=42.5
```

---

## 9. Dependencies

- Python ≥ 3.9
- PyTorch ≥ 1.13
- `coperception` (local editable install at `/home/bkiraly/coperception`)
- `nuscenes-devkit`
- `numpy`, `scipy`, `pyquaternion`
- `codecarbon` (optional; energy measurement falls back to 0 if absent)
- `streamlit` (optional; only for the GUI tool, not part of the evaluation pipeline)

See `requirements.txt` for the full pinned list.
