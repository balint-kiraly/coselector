# Path to the original V2X-Sim dataset
original_data_path := ../data/V2X-Sim-2
# Where to save the created data
create_data_save_path := ./created_data/V2X-Sim-det
# Index of the beginning scene
scene_begin := 10
 # Index of the ending scene + 1
scene_end := 20 # max 100
# Index of the start agent
from_agent := 0
# Index of the end agent + 1
to_agent := 6 # max 6
# [v2.0 / v2.0-mini]
dataset_version := v2.0

# Path to the test/val data
testing_data := $(create_data_save_path)/test
# [lowerbound / upperbound / v2v / disco / when2com / max / mean / sum / agent]
com := upperbound
batch_size := 4
# Where to load/save the checkpoints
checkpoint_path := checkpoints
# Where to store the logs
log_path := logs
# Train for how many epochs
nepoch := 100
# 1: apply late fusion. 0: no late fusion
apply_late_fusion := 1
# 1: do visualization. 0: no visualization
visualization := 0

create_data:
	python preprocess/bev_precompute.py \
	--root $(original_data_path) \
	--scene_begin $(scene_begin) \
	--scene_end $(scene_end) \
	--savepath $(create_data_save_path) \
	--from_agent $(from_agent) \
	--to_agent $(to_agent) \
	--dataset_version $(dataset_version)

inspect_bev:
	python tools/inspect_bev_sample.py \
    --data $(create_data_save_path) \
    --split "train" \
    --agent 1 \
    --scene 0

test:
	python test_codet_selector.py \
	--data $(testing_data) \
	--com $(com) \
	--resume $(checkpoint_path)/$(com)/with_rsu/epoch_$(nepoch).pth \
	--tracking \
	--logpath $(log_path) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu 1