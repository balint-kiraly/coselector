.PHONY: create_data test test_identity_rsu train_rsu_centric

# ── Paths ─────────────────────────────────────────────────────────────────────
# V2X-Sim-2 raw dataset root
original_data_path     := /mnt/10TB/balintkiraly/data/data/V2X-Sim-2
# Preprocessed sparse BEV data (output of create_data)
create_bev_data_save_path  := /mnt/10TB/balintkiraly/created_data/V2X-Sim-det
# Preprocessed state-feature JSON tree (output of create_data)
create_state_data_save_path := /mnt/10TB/balintkiraly/created_data/V2X-Sim-States
# Pretrained FaFNet checkpoints
checkpoint_path        := /mnt/10TB/balintkiraly/checkpoints
# Evaluation result logs
log_path               := /mnt/10TB/balintkiraly/results
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


# ── Targets ───────────────────────────────────────────────────────────────────

# Preprocess raw V2X-Sim dataset into sparse BEV tensors and state-feature JSON.
# Idempotent: completed scenes are tracked in bev_completed.json /
# state_completed.json and skipped on reruns.
create_data:
	python preprocess/bev_precompute.py \
		--root $(original_data_path) \
		--scene_begin $(scene_begin) \
		--scene_end $(scene_end) \
		--from_agent $(from_agent) \
		--to_agent $(to_agent) \
		--save_path $(create_bev_data_save_path) \
		--dataset_version $(dataset_version) \
		$(if $(only_split),--only_split $(only_split),)

	python -m preprocess.state_precompute \
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
	--logpath $(log_path) \
	--apply_late_fusion 0 \
	--visualization $(visualization) \
	--rsu 1 \
	--num_agent 6 \
	--selection \
	--sel_method $(sel_method) \
	--K $(K) \
	--budget_mb $(budget_mb) \
	--scene_begin $(scene_begin) \
	--scene_end $(scene_end) \


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
	--logpath $(log_path) \
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
