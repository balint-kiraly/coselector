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

# Path to the test/val data
testing_data := $(create_bev_data_save_path)/test
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
		--save_path $(create_bev_data_save_path)

	python -m preprocess.state_precompute \
		--root $(original_data_path) \
		--scene_begin $(scene_begin) \
		--scene_end $(scene_end) \
		--save_path $(create_state_data_save_path)

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
	--scene_begin 5 \
	--scene_end 20 \
	--selection \
	--sel_method $(sel_method)

test_identity:
	$(MAKE) test sel_method=identity

test_all_agents:
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
	--scene_begin 5 \
	--scene_end 20

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