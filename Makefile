.PHONY: create_data test test_identity_rsu test_rsu_only test_late_fusion_rsu train_rsu_centric \
        train_selector train_selector_smoke clean_selector

# ── Paths ─────────────────────────────────────────────────────────────────────
# V2X-Sim-2 raw dataset root
original_data_path     := /mnt/10TB/balintkiraly/data/data/V2X-Sim-2
# Preprocessed sparse BEV data (output of create_data)
create_bev_data_save_path  := /mnt/10TB/balintkiraly/created_data/V2X-Sim-det
# Preprocessed state-feature JSON tree (output of create_data)
create_state_data_save_path := /mnt/10TB/balintkiraly/created_data/V2X-Sim-States
# Pretrained checkpoints
checkpoint_path        := /mnt/10TB/balintkiraly/checkpoints
# Evaluation result logs
results_path               := /mnt/10TB/balintkiraly/results
# Fine-tuned RSU-centric checkpoint output
rsu_ckpt_save          := /mnt/10TB/balintkiraly/checkpoints/rsu_centric

# ── Preprocessing scope ────────────────────────────────────────────────────────
# Scene range processed by create_data (0-indexed, end exclusive; max 100).
scene_begin  := 0
scene_end    := 100    # max 100
# Agent range (0 = RSU, 1-5 = vehicles; V2X-Sim has 6 agents total)
from_agent   := 0
to_agent     := 6
# Dataset version: v2.0 (full) or v2.0-mini
dataset_version := v2.0
# Optional split filter for create_data: train | val | test | (empty = all)
only_split   :=

# ── Evaluation parameters ──────────────────────────────────────────────────────
# Val data path (Coperception uses the val split)
testing_data := $(create_bev_data_save_path)/val
# Checkpoint epoch to load
n_epoch      := 100
# Selection strategy: identity | closest_k | velocity | heuristic | bandwidth | ml_model
sel_method   := identity
# K agents to select (closest_k / velocity / heuristic)
K            := 3
# Bandwidth budget in MB (bandwidth strategy)
budget_mb    := 2.0
# 1 = enable visualisation, 0 = disable
visualization := 0

# ── Training parameters ────────────────────────────────────────────────────────
batch_size   := 4
min_agents   := 1
max_agents   := 5
train_patience := 7

# ── Selector training parameters ──────────────────────────────────────────────
selector_save_dir          := $(results_path)/selector
selector_rl_epochs         := 10
selector_curriculum_epochs := 0
selector_lambda_cost       := 0.4
selector_lr                := 1e-3
selector_val_every         := 2
selector_frames_per_epoch  := 30    # smoke test only
selector_frames_per_epoch_full := 2000  # full run: ~17 min/epoch × 10 = ~3 h
# Directory for pre-training frame caches (q_lower/q_upper/features).
# The script names each file selector_cache_s{begin}-{end}.pt automatically.
# Delete the relevant file to force a rebuild (e.g. after swapping the detector ckpt).
selector_cache_dir         := /mnt/10TB/balintkiraly/created_data/selector_cache
# Path to a trained selector checkpoint; used by test_codet_selector sel_method=ml_model
sel_model_path := $(selector_save_dir)/selector_best.pth
sel_threshold  := 0.5


# ── Targets ───────────────────────────────────────────────────────────────────

# Preprocess raw V2X-Sim dataset into sparse BEV tensors and state-feature JSON.
# Idempotent: completed scenes are tracked in bev_completed.json /
# state_completed.json and skipped on reruns.
create_data:
	CUDA_VISIBLE_DEVICES="" python -u preprocess/bev_precompute.py \
		--root $(original_data_path) \
		--scene_begin $(scene_begin) \
		--scene_end $(scene_end) \
		--from_agent $(from_agent) \
		--to_agent $(to_agent) \
		--save_path $(create_bev_data_save_path) \
		--dataset_version $(dataset_version) \
		$(if $(only_split),--only_split $(only_split),)

	CUDA_VISIBLE_DEVICES="" python -u -m preprocess.state_precompute \
		--root $(original_data_path) \
		--scene_begin $(scene_begin) \
		--scene_end $(scene_end) \
		--save_path $(create_state_data_save_path) \
		$(if $(only_split),--only_split $(only_split),)


# RSU-centric cooperative detection evaluation with agent selection.
#
# The RSU (agent 0) is the reference frame and GT evaluator.
# Vehicles (agents 1-5) send their BEVs; the RSU fuses the selected subset
# and runs detection using the upperbound/no_rsu FaFNet checkpoint.
#
# Usage:
#   make test_codet_selector                             # identity (all agents)
#   make test_codet_selector sel_method=closest_k K=3
#   make test_codet_selector sel_method=bandwidth budget_mb=2.0
#   make test_codet_selector scene_begin=5 scene_end=10  # subset of scenes
test_codet_selector:
	python test_codet_selector.py \
	--data_prep $(testing_data) \
	--state_path $(create_state_data_save_path) \
	--com lowerbound \
	--resume $(checkpoint_path)/upperbound/no_rsu/epoch_$(n_epoch).pth \
	--logpath $(results_path) \
	--apply_late_fusion 0 \
	--visualization $(visualization) \
	--rsu 1 \
	--num_agent 6 \
	--selection \
	--sel_method $(sel_method) \
	--K $(K) \
	--budget_mb $(budget_mb) \
	--sel_model_path $(sel_model_path) \
	--sel_threshold $(sel_threshold) \
	--scene_begin $(scene_begin) \
	--scene_end $(scene_end) \


# Late-fusion RSU baseline.
# Vehicles 1-5 each detect independently on their own BEV in their own frame.
# Box predictions are transformed into RSU frame and NMS-fused at the RSU.
# RSU's own lidar is suppressed (--no_rsu_detect 1): it is a passive relay only.
# Only the RSU's merged detection result is evaluated against RSU GT (agent 0 AOI).
# eval_start_idx and eval loop bound are set automatically in code for this mode.
# Checkpoint: lowerbound/with_rsu (matches num_agent=6 model architecture).
test_late_fusion_rsu:
	python test_codet_selector.py \
	--data_prep $(testing_data) \
	--state_path $(create_state_data_save_path) \
	--com lowerbound \
	--resume $(checkpoint_path)/lowerbound/with_rsu/epoch_$(n_epoch).pth \
	--logpath $(results_path) \
	--apply_late_fusion 1 \
	--no_rsu_detect 1 \
	--visualization $(visualization) \
	--rsu 1 \
	--num_agent 6 \
	--scene_begin $(scene_begin) \
	--scene_end $(scene_end)


# RSU lidar-only baseline.
# The RSU detects using only its own point cloud — no vehicle data received.
# num_agent=1 with rsu=1 loads only agent0 and evaluates it directly (no vehicle skip).
# Checkpoint: lowerbound/with_rsu (trained on individual-agent BEVs, RSU included).
test_rsu_only:
	python test_codet_selector.py \
	--data_prep $(testing_data) \
	--state_path $(create_state_data_save_path) \
	--com lowerbound \
	--resume $(checkpoint_path)/lowerbound/with_rsu/epoch_$(n_epoch).pth \
	--logpath $(results_path) \
	--apply_late_fusion 0 \
	--visualization $(visualization) \
	--rsu 1 \
	--num_agent 1 \
	--scene_begin $(scene_begin) \
	--scene_end $(scene_end)


# General test target (non-RSU / vehicle-ego mode, original coperception style).
# Useful for comparing against the original coperception baselines.
# Override com = [lowerbound / upperbound / v2v / disco / when2com / max / mean / sum / agent]
com               := upperbound
rsu               := 0
num_agent         := 6
apply_late_fusion := 1

test:
	python test_codet_selector.py \
	--data_prep $(testing_data) \
	--state_path $(create_state_data_save_path) \
	--com $(com) \
	--resume $(checkpoint_path)/$(com)/no_rsu/epoch_$(n_epoch).pth \
	--logpath $(results_path) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu $(rsu) \
	--num_agent $(num_agent) \
	--selection \
	--sel_method $(sel_method) \
	--K $(K) \
	--budget_mb $(budget_mb) \
	--scene_begin $(scene_begin) \
	--scene_end $(scene_end) \
	$(if $(filter-out 0,$(max_inference_ms)),--max_inference_ms $(max_inference_ms),)


# Train the learned agent selector with REINFORCE.
#
# Smoke run (2 epochs × 30 frames — confirms data loads, cache builds, reward moves):
#   make train_selector_smoke
#
# Full overnight run:
#   make train_selector
#
# Override any hyper:
#   make train_selector selector_rl_epochs=30 selector_lambda_cost=0.2
train_selector_smoke:
	python -m selection.train_selector \
	--ckpt       $(checkpoint_path)/upperbound/no_rsu/epoch_$(n_epoch).pth \
	--data_det   $(create_bev_data_save_path)/train \
	--data_val   $(create_bev_data_save_path)/test \
	--data_state $(create_state_data_save_path) \
	--save_dir   $(selector_save_dir)_smoke \
	--cache_dir  $(selector_cache_dir) \
	--train_scene_begin 0  --train_scene_end 10 \
	--val_scene_begin   0  --val_scene_end   10 \
	--rl_epochs 2 \
	--curriculum_epochs 0 \
	--lambda_cost $(selector_lambda_cost) \
	--lr $(selector_lr) \
	--frames_per_epoch $(selector_frames_per_epoch) \
	--val_every 1 \
	--tensorboard

train_selector:
	python -m selection.train_selector \
	--ckpt       $(checkpoint_path)/upperbound/no_rsu/epoch_$(n_epoch).pth \
	--data_det   $(create_bev_data_save_path)/train \
	--data_val   $(create_bev_data_save_path)/test \
	--data_state $(create_state_data_save_path) \
	--save_dir   $(selector_save_dir) \
	--cache_dir  $(selector_cache_dir) \
	--train_scene_begin 0  --train_scene_end 100 \
	--val_scene_begin   0  --val_scene_end   100 \
	--rl_epochs $(selector_rl_epochs) \
	--curriculum_epochs $(selector_curriculum_epochs) \
	--lambda_cost $(selector_lambda_cost) \
	--lr $(selector_lr) \
	--frames_per_epoch $(selector_frames_per_epoch_full) \
	--val_every $(selector_val_every) \
	--tensorboard


# Remove selector training artifacts for a clean restart.
# Deletes checkpoints, metrics CSV, and TensorBoard events from both the
# full and smoke save directories.  The frame cache (selector_cache.pt) is
# intentionally preserved — it takes ~1 h to build and is reusable.
#
# Usage:
#   make clean_selector          # wipe full-run artifacts
#   make clean_selector smoke=1  # also wipe smoke-run artifacts
clean_selector:
	rm -fv $(selector_save_dir)/selector_last.pth \
	       $(selector_save_dir)/selector_best.pth \
	       $(selector_save_dir)/train_metrics.csv \
	       $(selector_save_dir)/events.out.tfevents.*
	$(if $(smoke),rm -fv $(selector_save_dir)_smoke/selector_last.pth \
	                     $(selector_save_dir)_smoke/selector_best.pth \
	                     $(selector_save_dir)_smoke/train_metrics.csv \
	                     $(selector_save_dir)_smoke/events.out.tfevents.*,)


# Fine-tune FaFNet for RSU-centric cooperative detection.
# Warm-starts from upperbound/no_rsu (vehicle-merged BEVs, no RSU lidar)
# and trains with random vehicle subsets in RSU frame.
train_rsu_centric:
	python training/train_rsu_centric.py \
	--data_train $(create_bev_data_save_path)/train \
	--data_val   $(create_bev_data_save_path)/val \
	--resume     $(checkpoint_path)/upperbound/no_rsu/epoch_$(n_epoch).pth \
	--logpath    $(rsu_ckpt_save) \
	--nepoch     30 \
	--batch_size $(batch_size) \
	--nworker    4 \
	--lr         1e-4 \
	--min_agents $(min_agents) \
	--max_agents $(max_agents) \
	--patience   $(train_patience) \
	--amp
