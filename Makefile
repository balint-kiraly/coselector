# Path to the original V2X-Sim dataset
original_data_path := ../data/V2X-Sim-2
# Where to save the created bev data
create_bev_data_save_path := ./created_data/V2X-Sim-det
# Where to save the created state data
create_state_data_save_path := ./created_data/V2X-Sim-States
# Index of the beginning scene
scene_begin := 0
 # Index of the ending scene + 1
scene_end := 100 # max 100
# Index of the start agent
from_agent := 0
# Index of the end agent + 1
to_agent := 6 # max 6
# [v2.0 / v2.0-mini]
dataset_version := v2.0

# Path to the test/val data
testing_data := $(create_bev_data_save_path)/V2X-Sim-det/test
# [lowerbound / upperbound / v2v / disco / when2com / max / mean / sum / agent]
com := upperbound
batch_size := 4
# Where to load/save the checkpoints
checkpoint_path := checkpoints
# Where to store the logs
log_path := logs
# Train for how many epochs
n_epoch := 20
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

test:
	python test_codet_selector.py \
	--data $(original_data_path) \
	--data_prep $(testing_data)
	--com $(com) \
	--resume $(checkpoint_path)/$(com)/with_rsu/epoch_$(n_epoch).pth \
	#--tracking \
	--logpath $(log_path) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu 1
	--scene_begin 0 \
	--scene_end 20 \
	--sel_method "ml_model" \
	--sel_model_path ${checkpoint_path}/selector_models/agent_selector.pth

# --data ../data/V2X-Sim-2 --data_prep ./created_data/V2X-Sim-det/test	--com upperbound --resume checkpoints/upperbound/with_rsu/epoch_100.pth	--logpath logs --apply_late_fusion 1 --visualization 0 --rsu 1 --scene_begin 0	--scene_end 20 --sel_method "ml_model" --sel_model_path checkpoints/selector_models/agent_selector.pth