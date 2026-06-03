# Path to the original V2X-Sim dataset
original_data_path := /mnt/10TB/balintkiraly/data/V2X-Sim-2
# Where to save the created bev data
create_bev_data_save_path := /mnt/10TB/balintkiraly/created_data/V2X-Sim-det
# Where to save the created state data
create_state_data_save_path := /mnt/10TB/balintkiraly/created_data/V2X-Sim-States
# Index of the beginning scene
scene_begin := 0
 # Index of the ending scene + 1
scene_end := 1 # max 100
# Index of the start agent
from_agent := 0
# Index of the end agent + 1
to_agent := 6 # max 6
# [v2.0 / v2.0-mini]
dataset_version := v2.0
# Optional split filter: set to train/val/test to skip other splits (leave empty for all)
only_split :=

# Path to the val data (matches coperception evaluation split)
testing_data := $(create_bev_data_save_path)/val
# [lowerbound / upperbound / v2v / disco / when2com / max / mean / sum / agent]
com := upperbound
batch_size := 4
# Where to load/save the checkpoints
checkpoint_path := checkpoints
# Where to store the logs
log_path := logs
# Train for how many epochs
n_epoch := 100
# 1: apply late fusion. 0: no late fusion
apply_late_fusion := 1
# 1: do visualization. 0: no visualization
visualization := 0

create_data:
	python preprocess/bev_precompute.py \
		--root $(original_data_path) \
		--scene_begin $(scene_begin) \
		--scene_end $(scene_end) \
		--save_path $(create_bev_data_save_path) \
		$(if $(only_split),--only_split $(only_split),)

	python -m preprocess.state_precompute \
		--root $(original_data_path) \
		--scene_begin $(scene_begin) \
		--scene_end $(scene_end) \
		--save_path $(create_state_data_save_path) \
		$(if $(only_split),--only_split $(only_split),)

inspect_sensor:
	python tools/inspect_sensor_data.py \
        --sensor "gnss" \
        --agent_id 5 \
        --scene_id 5 \
        --frame_id 7

inspect_bev:
	python tools/inspect_bev_sample.py \
    --data $(create_bev_data_save_path) \
    --split "train" \
    --agent 1 \
    --scene 0

plot_agents:
	python tools/plot_agent_counts.py \
		--state_root $(create_state_data_save_path) \
		--save_path $(create_state_data_save_path)/agent_counts.png

# Selection method: identity | closest_k | heuristic | ml_model
sel_method := identity
# RSU: 0 = no RSU (agents 1-5), 1 = include RSU (agent 0)
rsu := 0
# Number of agents (including RSU slot 0); keep at 6 for V2X-Sim
num_agent := 6

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
	--sel_method $(sel_method)

test_identity:
	$(MAKE) test sel_method=identity

# RSU-centric with planar BEV fusion (Z-preserving transform).
# Vehicles send their own BEVs; RSU transforms (X,Y) into its frame and keeps
# vehicle-frame Z so voxels survive the ~5.5 m height offset.  GT labels are 2D.
# Uses upperbound/no_rsu checkpoint: trained on dense merged vehicle BEVs
# (closest density match; no RSU lidar — correct for Option B).
lb_checkpoint_path := /home/bkiraly/coperception/tools/det/checkpoints
test_identity_rsu:
	python test_codet_selector.py \
	--data_prep $(testing_data) \
	--state_path $(create_state_data_save_path) \
	--com lowerbound \
	--resume $(lb_checkpoint_path)/upperbound/no_rsu/epoch_$(n_epoch).pth \
	--logpath $(log_path) \
	--apply_late_fusion 0 \
	--visualization $(visualization) \
	--rsu 1 \
	--num_agent $(num_agent) \
	--selection \
	--sel_method $(sel_method)

# rsu_suffix: no_rsu or with_rsu — auto-selected based on rsu flag
rsu_suffix := $(if $(filter 1,$(rsu)),with_rsu,no_rsu)

test_all_agents:
	python test_codet_selector.py \
	--data_prep $(testing_data) \
	--state_path $(create_state_data_save_path) \
	--com $(com) \
	--resume $(lb_checkpoint_path)/$(com)/$(rsu_suffix)/epoch_$(n_epoch).pth \
	--logpath $(log_path) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu $(rsu) \
	--num_agent $(num_agent)

# ── RSU-centric fine-tuning ──────────────────────────────────────────────────
# Warm-starts from upperbound/with_rsu (all-agents oracle) and fine-tunes on
# random vehicle subsets merged in RSU frame. Produces a model robust to any k
# selected agents.  Safe defaults: 30 epochs, early-stop patience=7, AMP on.
rsu_ckpt_save := /home/bkiraly/coselector/checkpoints/rsu_centric
min_agents := 1
max_agents := 5
train_patience := 7

# Warm-start from upperbound/no_rsu (vehicles 1-5 merged, no RSU lidar).
# Option B: RSU is a passive relay — RSU lidar is never included.
# The only difference vs the pretrained model is the reference frame
# (agent1 frame → RSU frame), which fine-tuning corrects in ~10 epochs.
train_rsu_centric:
	python training/train_rsu_centric.py \
	--data_train $(create_bev_data_save_path)/train \
	--data_val   $(create_bev_data_save_path)/val \
	--resume     $(lb_checkpoint_path)/upperbound/no_rsu/epoch_$(n_epoch).pth \
	--logpath    $(rsu_ckpt_save) \
	--nepoch     30 \
	--batch_size $(batch_size) \
	--nworker    4 \
	--lr         1e-4 \
	--min_agents $(min_agents) \
	--max_agents $(max_agents) \
	--patience   $(train_patience) \
	--amp

# Quick 2-epoch smoke test (no warm start, tiny lr, just verifies the pipeline)
train_rsu_centric_smoke:
	python training/train_rsu_centric.py \
	--data_train $(create_bev_data_save_path)/train \
	--data_val   $(create_bev_data_save_path)/val \
	--logpath    /tmp/rsu_centric_smoke \
	--nepoch     2 \
	--batch_size 2 \
	--nworker    2 \
	--lr         1e-4 \
	--min_agents 2 \
	--max_agents 3

train_selector:
	python -m selection.train_selector \
	--data_det $(create_bev_data_save_path)/val \
	--data_state $(create_state_data_save_path) \
	--agent_start $(from_agent) \
	--agent_end $(to_agent) \
	--scene_start $(scene_begin) \
	--scene_end $(scene_end) \
	--ckpt $(checkpoint_path)/$(com)/no_rsu/epoch_$(n_epoch).pth \
	--save_path $(checkpoint_path)/selector_models/agent_selector.pth