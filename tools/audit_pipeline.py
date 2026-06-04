"""
Audit the selection pipeline vs non-selection path.
Checks:
 1. BEV content differences (teacher vs individual; fused vs raw)
 2. What assemble_detection_inputs actually produces
 3. Trans-matrix indexing in _warp_and_merge_bevs
 4. GT structure and area-of-interest semantics
 5. Model forward with 1 vs 5 agents
"""
import os, sys
sys.path.insert(0, '/home/bkiraly/coselector')

import torch
import numpy as np
import torch.nn as nn
from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import FaFModule
from coperception.utils.loss import SoftmaxFocalClassificationLoss, WeightedSmoothL1LocalizationLoss
from coperception.models.det import FaFNet
from torch.utils.data import DataLoader
from torch import optim
from selection.bev_builder import assemble_detection_inputs

DATA_TEST = '/mnt/10TB/balintkiraly/created_data/V2X-Sim-det/test'
CKPT = '/mnt/10TB/balintkiraly/checkpoints/upperbound/no_rsu/epoch_100.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- load one sample ----
config = Config('train', binary=True, only_det=True)
config_global = ConfigGlobal('train', binary=True, only_det=True)
config.flag = 'upperbound'; config.split = 'test'

ds = V2XSimDet(
    dataset_roots=[f'{DATA_TEST}/agent{i}' for i in range(1, 6)],
    config=config, config_global=config_global,
    split='val', val=True, bound='upperbound', kd_flag=False, rsu=False,
)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
sample = next(iter(loader))

fields = list(zip(*sample))
(pvp_list, pvp_teacher_list, label_list, reg_target_list, reg_loss_mask_list,
 anchors_list, vis_maps_list, gt_max_iou,
 filenames, target_agent_id_list, num_agent_list, trans_matrices_list) = fields

print("=" * 70)
print("1. BEV CONTENT ANALYSIS")
print("=" * 70)
print(f"pvp_list[0] type={type(pvp_list[0])}, is empty list: {pvp_list[0] == []}")
print(f"pvp_teacher_list has {len(pvp_teacher_list)} agents")
for i, bev in enumerate(pvp_teacher_list):
    t = torch.as_tensor(bev, dtype=torch.float32)
    nonzero = (t > 0).sum().item()
    total = t.numel()
    print(f"  agent{i} teacher BEV shape={t.shape}  nonzero={nonzero}/{total} ({100*nonzero/total:.2f}%)")

# Lowerbound data: check if pvp_list would be populated
print("\n--- Loading same scenes with bound='lowerbound' ---")
ds_lb = V2XSimDet(
    dataset_roots=[f'{DATA_TEST}/agent{i}' for i in range(1, 6)],
    config=config, config_global=config_global,
    split='val', val=True, bound='lowerbound', kd_flag=False, rsu=False,
)
loader_lb = DataLoader(ds_lb, batch_size=1, shuffle=False, num_workers=0)
sample_lb = next(iter(loader_lb))
fields_lb = list(zip(*sample_lb))
pvp_lb = fields_lb[0]
pvp_teacher_lb = fields_lb[1]
print(f"pvp_list[0] type={type(pvp_lb[0])}, shape={pvp_lb[0].shape if hasattr(pvp_lb[0],'shape') else 'not tensor'}")
if hasattr(pvp_lb[0], 'shape'):
    for i, bev in enumerate(pvp_lb):
        t = torch.as_tensor(bev, dtype=torch.float32)
        nonzero = (t > 0).sum().item()
        total = t.numel()
        print(f"  agent{i} lowerbound BEV shape={t.shape}  nonzero={nonzero}/{total} ({100*nonzero/total:.2f}%)")

print("\n" + "=" * 70)
print("2. TRANS-MATRIX STRUCTURE")
print("=" * 70)
print(f"trans_matrices_list has {len(trans_matrices_list)} agents")
for i, tm in enumerate(trans_matrices_list):
    t = torch.as_tensor(tm)
    print(f"  agent{i}: shape={t.shape}")
    # Show the matrices
    for j in range(t.shape[1]):
        mat = t[0, j]
        is_identity = torch.allclose(mat, torch.eye(4, dtype=mat.dtype), atol=1e-3)
        print(f"    slot[{j}]: {'IDENTITY' if is_identity else 'transform'} "
              f"t=({mat[0,3]:.2f}, {mat[1,3]:.2f}, {mat[2,3]:.2f})")

print("\n" + "=" * 70)
print("3. WHAT assemble_detection_inputs ACTUALLY PRODUCES")
print("=" * 70)

# Simulate selection = all 4 agents (those with non-empty teachers)
selected_indices = list(range(len(pvp_teacher_list)))
def _sel(lst):
    return [lst[i] for i in selected_indices]

pvp_sel = _sel(pvp_list)
pvp_teacher_sel = _sel(pvp_teacher_list)
label_sel = _sel(label_list)
reg_sel = _sel(reg_target_list)
regmask_sel = _sel(reg_loss_mask_list)
anchors_sel = _sel(anchors_list)
vis_sel = _sel(vis_maps_list)
agent_id_sel = _sel(target_agent_id_list)
num_sel = _sel(num_agent_list)
trans_sel = _sel(trans_matrices_list)

data_assembled = assemble_detection_inputs(
    config,
    pvp_sel, pvp_teacher_sel,
    label_sel, reg_sel, regmask_sel, anchors_sel, vis_sel,
    agent_id_sel, num_sel, trans_sel,
    device,
)
fused = data_assembled["bev_seq"]
print(f"Assembled bev_seq shape: {fused.shape}")
print(f"Assembled bev_seq dtype: {fused.dtype}")
nonzero_fused = (fused > 0).sum().item()
total_fused = fused.numel()
print(f"Assembled bev_seq nonzero: {nonzero_fused}/{total_fused} ({100*nonzero_fused/total_fused:.2f}%)")

# Compare with single teacher BEV (agent 1, already in ref frame)
teacher0 = torch.as_tensor(pvp_teacher_list[0], dtype=torch.float32)
print(f"\nAgent1 teacher BEV shape: {teacher0.shape}")
nonzero_t0 = (teacher0 > 0).sum().item()
print(f"Agent1 teacher BEV nonzero: {nonzero_t0}/{teacher0.numel()} ({100*nonzero_t0/teacher0.numel():.2f}%)")

# Reference: official non-selection path
official_stacked = torch.cat(tuple(pvp_teacher_list), 0)
print(f"\nOfficial stacked BEV (cat dim0): {official_stacked.shape}")

print("\n" + "=" * 70)
print("4. FORWARD PASS COMPARISON")
print("=" * 70)

# Load model as 5-agent (official)
model5 = FaFNet(config, layer=3, kd_flag=False, num_agent=5)
model5 = nn.DataParallel(model5).to(device)
model5.eval()
ckpt = torch.load(CKPT, map_location=device)
optimizer5 = optim.Adam(model5.parameters(), lr=0.001)
criterion = {"cls": SoftmaxFocalClassificationLoss(), "loc": WeightedSmoothL1LocalizationLoss()}
faf5 = FaFModule(model5, model5, config, optimizer5, criterion, 0)
faf5.model.load_state_dict(ckpt["model_state_dict"])
faf5.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
faf5.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
faf5.model.eval()
print(f"5-agent model loaded from epoch {ckpt['epoch']}")

# Load model as 1-agent (selection)
model1 = FaFNet(config, layer=3, kd_flag=False, num_agent=1)
model1 = nn.DataParallel(model1).to(device)
model1.eval()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
faf1 = FaFModule(model1, model1, config, optimizer1, criterion, 0)
try:
    faf1.model.load_state_dict(ckpt["model_state_dict"])
    print("1-agent model: checkpoint loaded successfully (weights ARE compatible)")
except RuntimeError as e:
    print(f"1-agent model: checkpoint load FAILED: {e}")

# Try 1-agent forward with assembled BEV
print("\n--- 1-agent forward with assembled BEV ---")
try:
    with torch.no_grad():
        loss1, cls1, loc1, result1 = faf1.model(data_assembled, 1, num_agent=1)
    print(f"  result[0]: {len(result1[0])} detections")
except Exception as e:
    print(f"  FAILED: {e}")

# Official 5-agent forward
trans_stack = torch.stack(tuple(trans_matrices_list), 1).to(device)
num_agents_t = torch.stack(tuple(num_agent_list), 1).to(device) - 1  # no RSU
data5 = {
    "bev_seq": official_stacked.to(device),
    "labels": torch.cat(tuple(label_list), 0).to(device),
    "reg_targets": torch.cat(tuple(reg_target_list), 0).to(device),
    "anchors": torch.cat(tuple(anchors_list), 0).to(device),
    "vis_maps": torch.cat(tuple(vis_maps_list), 0).to(device),
    "reg_loss_mask": torch.cat(tuple(reg_loss_mask_list), 0).to(device).bool(),
    "target_agent_ids": torch.stack(tuple(target_agent_id_list), 1).to(device),
    "num_agent": num_agents_t,
    "trans_matrices": trans_stack,
}
print("\n--- 5-agent forward (official path) ---")
with torch.no_grad():
    loss5, cls5, loc5, result5 = faf5.model(data5, 1, num_agent=5)
print(f"  result[0] (#dets for agent1): {len(result5[0])}")
if result5[0] and result5[0][0]:
    boxes = result5[0][0][0]
    if isinstance(boxes, dict):
        print(f"    pred boxes shape: {boxes.get('pred', torch.zeros(1)).shape if isinstance(boxes.get('pred'), torch.Tensor) else 'N/A'}")

print("\n" + "=" * 70)
print("5. GT SEMANTICS: area of interest")
print("=" * 70)
print("Per-agent GT box counts (scene5, frame0):")
for i, gt in enumerate(gt_max_iou):
    boxes = gt[0]["gt_box"]
    if isinstance(boxes, torch.Tensor):
        n = boxes.shape[1]
    elif isinstance(boxes, list) and len(boxes) == 0:
        n = 0
    else:
        n = 0
    print(f"  agent{i+1} (filename={filenames[i][0][0].split('/')[-3]}): {n} GT boxes")

print("\nConclusion:")
print("  GT is per-agent LOCAL area (not global). agent1's GT = objects in agent1's detection range.")
print("  For cloud server evaluation (area centred on agent1): agent1's GT IS the right target.")
print("  Non-selection path accumulates ALL agents' GT (5x more objects), inflating totals.")
print("  Fair comparison: agent1's individual mAP from non-selection run vs selection mAP.")
