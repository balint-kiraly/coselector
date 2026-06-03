"""Inspect raw dataloader output shapes, BEV merging, and metric pipeline."""
import os, sys
sys.path.insert(0, '/home/bkiraly/coselector')

import torch
import numpy as np
from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import FaFModule
from coperception.utils.loss import SoftmaxFocalClassificationLoss, WeightedSmoothL1LocalizationLoss
from coperception.models.det import FaFNet
from torch.utils.data import DataLoader
from torch import nn, optim

DATA_TEST = '/mnt/10TB/balintkiraly/created_data/V2X-Sim-det/test'
CKPT = '/home/bkiraly/coselector/checkpoints/upperbound/no_rsu/epoch_100.pth'

config = Config('train', binary=True, only_det=True)
config_global = ConfigGlobal('train', binary=True, only_det=True)
config.flag = 'upperbound'
config.split = 'test'

agent_idx_range = range(1, 6)
ds = V2XSimDet(
    dataset_roots=[f'{DATA_TEST}/agent{i}' for i in agent_idx_range],
    config=config, config_global=config_global,
    split='val', val=True, bound='upperbound', kd_flag=False, rsu=False,
)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
sample = next(iter(loader))

print("=" * 60)
print("DATASET BATCH STRUCTURE")
print("=" * 60)
print(f"len(sample) = {len(sample)}  <- number of agents")

# V2XSimDet with multiple dataset_roots returns one item per agent per batch
# sample[agent_idx] = (pvp, pvp_teacher, label, reg_target, reg_loss_mask,
#                       anchors, vis_maps, gt_max_iou, filenames,
#                       target_agent_id, num_agent, trans_matrices)
FIELD_NAMES = [
    'pvp_list', 'pvp_teacher_list', 'label_one_hot_list', 'reg_target_list',
    'reg_loss_mask_list', 'anchors_map_list', 'vis_maps_list', 'gt_max_iou',
    'filenames', 'target_agent_id_list', 'num_agent_list', 'trans_matrices_list'
]

for ag_i, ag_item in enumerate(sample):
    print(f"\n--- Agent index {ag_i} ---")
    for fi, (name, field) in enumerate(zip(FIELD_NAMES, ag_item)):
        if isinstance(field, torch.Tensor):
            print(f"  {name}: Tensor {tuple(field.shape)} {field.dtype}")
        elif isinstance(field, np.ndarray):
            print(f"  {name}: ndarray {field.shape} {field.dtype}")
        elif isinstance(field, list):
            if len(field) == 0:
                print(f"  {name}: empty list")
            elif isinstance(field[0], dict):
                print(f"  {name}: list[dict] len={len(field)}")
                for k, v in field[0].items():
                    if hasattr(v, 'shape'):
                        print(f"    [{k}]: {getattr(v,'shape','?')}")
                    else:
                        print(f"    [{k}]: {type(v).__name__} = {v}")
            elif isinstance(field[0], (list, tuple)):
                print(f"  {name}: list of lists, len={len(field)}, inner[0] len={len(field[0])}")
                print(f"    inner[0][0] = {repr(str(field[0][0])[:100])}")
            else:
                print(f"  {name}: list len={len(field)}, el[0] type={type(field[0]).__name__}")
        elif isinstance(field, str):
            print(f"  {name}: str = {repr(field[:100])}")
        else:
            print(f"  {name}: {type(field).__name__} = {field}")

print("\n" + "=" * 60)
print("HOW zip(*sample) WORKS")
print("=" * 60)
# zip(*sample) groups by FIELD across agents, not by agent
fields = list(zip(*sample))
print(f"len(fields) = {len(fields)}  <- number of fields")
print(f"len(fields[0]) = {len(fields[0])}  <- number of agents (5)")
print("  pvp_list[agent0] type:", type(fields[0][0]).__name__)
if isinstance(fields[0][0], torch.Tensor):
    print("  pvp_list[agent0] shape:", fields[0][0].shape)

print("\n" + "=" * 60)
print("OFFICIAL UPPERBOUND PATH (how coperception does it)")
print("=" * 60)
# This is what the original coperception test_codet.py does:
(pvp_list, pvp_teacher_list, label_list, reg_target_list, reg_loss_mask_list,
 anchors_list, vis_maps_list, gt_max_iou, filenames,
 target_agent_id_list, num_agent_list, trans_matrices_list) = fields

print("pvp_teacher_list: tuple of", len(pvp_teacher_list), "tensors")
for i, t in enumerate(pvp_teacher_list):
    print(f"  agent{i}: {t.shape}")

# Official upperbound: torch.cat(pvp_teacher_list, dim=0)
stacked = torch.cat(tuple(pvp_teacher_list), 0)
print("\ntorch.cat(pvp_teacher_list, 0) shape:", stacked.shape)

trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
print("torch.stack(trans_matrices_list, 1) shape:", trans_matrices.shape)

num_agents_t = torch.stack(tuple(num_agent_list), 1)
print("num_agent stacked:", num_agents_t, "-> after -=1 (no RSU):", num_agents_t - 1)

print("\nFilename[0][0][0]:", filenames[0][0][0])
print("target_agent_id[0]:", target_agent_id_list[0])

print("\n" + "=" * 60)
print("GT STRUCTURE")
print("=" * 60)
for i, gt in enumerate(gt_max_iou):
    print(f"gt_max_iou[{i}] type={type(gt).__name__}, len={len(gt)}")
    if isinstance(gt, list) and len(gt) > 0:
        print(f"  gt[0] type={type(gt[0]).__name__}")
        if isinstance(gt[0], dict):
            for k, v in gt[0].items():
                if hasattr(v, 'shape'): print(f"    {k}: {v.shape}")
                else: print(f"    {k}: {type(v).__name__} = {v}")
