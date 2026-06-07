import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime

import seaborn as sns
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.utils.mean_ap import eval_map
from coperception.models.det import *
from coperception.utils.detection_util import late_fusion
from coperception.utils.data_util import apply_pose_noise

from data_utils.build_state_features import build_state_features
from data_utils.cost_model import (
    CostConfig,
    MEASURED_LIDAR_SIZE_MB,
    compute_cost,
    make_balanced_config, bev_size_from_tensor,
)
from data_utils.state_index import StateIndex
from selection.bev_builder import assemble_detection_inputs
from selection.models import AgentSelectorMLP
from selection.policy import select_agents_from_metadata, SelectionMethod


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


try:
    from codecarbon import EmissionsTracker as _EmissionsTracker
    _HAS_CODECARBON = True
except ImportError:
    _HAS_CODECARBON = False


def _tracker_energy_J(tracker) -> float:
    """Extract joules from an EmissionsTracker (handles codecarbon 1.x and 2.x)."""
    if hasattr(tracker, "final_emissions_data") and tracker.final_emissions_data is not None:
        energy_kwh = tracker.final_emissions_data.energy_consumed
    elif hasattr(tracker, "_total_energy"):
        energy_kwh = tracker._total_energy.kwh
    else:
        return 0.0
    return energy_kwh * 3_600_000


def _start_run_tracker():
    """Start ONE codecarbon tracker for the whole eval run.

    IMPORTANT: never create a tracker per frame — EmissionsTracker.stop()
    blocks for ~3 s (final measurement flush), which was adding 3 s/frame.
    A single run-level tracker is started once and stopped once; per-frame
    energy is derived afterwards by distributing the measured total across
    frames in proportion to inference time.
    """
    if not _HAS_CODECARBON:
        return None
    try:
        tracker = _EmissionsTracker(save_to_file=False, measure_power_secs=1)
        tracker.start()
        return tracker
    except Exception as e:
        print(f"WARNING: could not start codecarbon tracker ({e}); energy=0")
        return None


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed_forward(device, forward_fn):
    """Run ``forward_fn`` and return ``(result, inference_time_ms)`` with GPU sync.

    Energy is NOT measured here (see _start_run_tracker): a per-call codecarbon
    tracker costs ~3 s due to the blocking stop(). Time only — cheap and exact.
    """
    _sync_device(device)
    t0 = time.perf_counter()
    result = forward_fn()
    _sync_device(device)
    inference_time_ms = (time.perf_counter() - t0) * 1000.0
    return result, inference_time_ms


@torch.no_grad()
def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    need_log = args.log
    num_workers = args.nworker
    apply_late_fusion = args.apply_late_fusion
    pose_noise = args.pose_noise
    compress_level = args.compress_level
    only_v2i = args.only_v2i

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    config.inference = args.inference
    if args.com == "upperbound":
        flag = "upperbound"
    elif args.com == "when2com":
        flag = "when2com"
        if args.inference == "argmax_test":
            flag = "who2com"
        if args.warp_flag:
            flag = flag + "_warp"
    elif args.com in {"v2v", "disco", "sum", "mean", "max", "cat", "agent"}:
        flag = args.com
    elif args.com == "lowerbound":
        flag = "lowerbound"
        if args.box_com:
            flag += "_box_com"
    else:
        raise ValueError(f"com: {args.com} is not supported")

    print("flag", flag)
    config.flag = flag
    config.split = "test"

    num_agent = args.num_agent
    # agent0 is the RSU
    agent_idx_range = range(num_agent) if args.rsu else range(1, num_agent)
    validation_dataset = V2XSimDet(
        dataset_roots=[f"{args.data_prep}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound="lowerbound" if args.selection else (
            "upperbound" if args.com == "upperbound" else "lowerbound"
        ),
        kd_flag=args.kd_flag,
        rsu=args.rsu,
    )

    # ── Speedup 1: replace the multiprocessing.Manager() cache with plain dicts.
    # V2XSimDet sets cache_size=0 for val (nothing is ever stored), but its
    # `pick_single_agent` still does `if idx in self.cache[agent_id]` — a Manager
    # proxy lookup that costs an IPC round-trip per agent per frame. Plain dicts
    # make that check a free local lookup.
    validation_dataset.cache = [{} for _ in range(validation_dataset.num_agent)]

    # ── Speedup 2: filter to the requested scene range at the DATASET level.
    # Previously every frame of all 10 val scenes was fully loaded from disk
    # (sparse BEV → dense voxel grids for all agents) and only then discarded by
    # the in-loop `scene_id out of range` check. Wrapping in a Subset means the
    # dataloader never even touches out-of-range frames.
    scene_of_idx = validation_dataset.seq_scenes[0]  # scene id per sample index
    keep_indices = [
        i for i, s in enumerate(scene_of_idx)
        if args.scene_begin <= s < args.scene_end
    ]
    if not keep_indices:
        raise RuntimeError(
            f"No frames in scene range [{args.scene_begin}, {args.scene_end}). "
            f"Scenes present: {sorted(set(scene_of_idx))}"
        )
    n_total_frames = len(scene_of_idx)
    eval_dataset = torch.utils.data.Subset(validation_dataset, keep_indices)
    print(
        f"Scene filter: keeping {len(keep_indices)}/{n_total_frames} frames "
        f"in scene range [{args.scene_begin}, {args.scene_end})"
    )

    validation_data_loader = DataLoader(
        eval_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    print("Frames to evaluate:", len(eval_dataset))

    # ===== load state index (only the scenes we will evaluate) =====
    state_index = StateIndex.from_fs(
        args.state_path,
        scene_ids=range(args.scene_begin, args.scene_end),
    )

    if not args.rsu:
        num_agent -= 1

    if args.selection:
        model = FaFNet(
            config, layer=args.layer, kd_flag=0, num_agent=1
        )
    elif flag == "upperbound" or flag.startswith("lowerbound"):
        model = FaFNet(
            config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent
        )
    elif flag.startswith("when2com") or flag.startswith("who2com"):
        # model = PixelwiseWeightedFusionSoftmax(config, layer=args.layer)
        model = When2com(
            config,
            layer=args.layer,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "disco":
        model = DiscoNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "sum":
        model = SumFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "mean":
        model = MeanFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "max":
        model = MaxFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "cat":
        model = CatFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "agent":
        model = AgentWiseWeightedFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "v2v":
        model = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )

    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)

    model_save_path = args.resume[: args.resume.rfind("/")]

    if args.inference == "argmax_test":
        model_save_path = model_save_path.replace("when2com", "who2com")

    os.makedirs(model_save_path, exist_ok=True)
    log_file_name = os.path.join(model_save_path, "log.txt")
    saver = open(log_file_name, "a")
    saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
    saver.flush()

    # Logging the details for this experiment
    saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
    saver.write(args.__repr__() + "\n\n")
    saver.flush()

    if not os.path.isfile(args.resume):
        rsu_hint = "with_rsu" if args.rsu else "no_rsu"
        raise FileNotFoundError(
            f"Checkpoint not found: {args.resume}. "
            f"With --rsu {args.rsu}, use a checkpoint trained for {rsu_hint}."
        )

    checkpoint = torch.load(
        args.resume, map_location="cpu"
    )  # We have low GPU utilization for testing
    start_epoch = checkpoint["epoch"] + 1
    fafmodule.model.load_state_dict(checkpoint["model_state_dict"])
    fafmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    #  ===== eval =====
    fafmodule.model.eval()
    save_fig_path = [
        check_folder(os.path.join(model_save_path, f"vis{i}")) for i in agent_idx_range
    ]
    tracking_path = [
        check_folder(os.path.join(model_save_path, f"tracking{i}"))
        for i in agent_idx_range
    ]

    # for local and global mAP evaluation
    det_results_local = [[] for i in agent_idx_range]
    annotations_local = [[] for i in agent_idx_range]

    # per-scene detection accumulation (for per-scene AP/recall in scene_stats.csv)
    scene_det_results: dict = {}   # scene_id -> list
    scene_annotations: dict = {}   # scene_id -> list

    tracking_file = [set()] * num_agent

    sel_model = None
    if args.selection and args.sel_method == "ml_model":
        assert args.sel_model_path is not None, \
            "--sel_model_path is required for sel_method=ml_model"
        sel_model = AgentSelectorMLP(input_dim=12).to(device)
        ckpt_data = torch.load(args.sel_model_path, map_location=device)
        if "model" in ckpt_data:
            sel_model.load_state_dict(ckpt_data["model"])
        else:
            sel_model.load_state_dict(ckpt_data)
        sel_model.eval()
        print(f"Loaded agent selection model from {args.sel_model_path}")

    if args.eval_start_idx >= 0:
        eval_start_idx = args.eval_start_idx          # explicit override
    elif args.rsu and num_agent == 1:
        eval_start_idx = 0                            # RSU-only: evaluate the RSU itself
    else:
        eval_start_idx = 1 if args.rsu else 0         # default: skip RSU slot in multi-agent mode
    frames_processed = 0
    measurements_dir = args.measurements_dir if args.measurements_dir else None
    try:
        cost_config = make_balanced_config(
            link_rate_Mbps=args.link_rate_mbps,
            measurements_dir=measurements_dir,
        )
        if measurements_dir:
            print(f"Loaded cost constants from {measurements_dir}")
    except FileNotFoundError as e:
        print(f"WARNING: {e}\nFalling back to default lidar_size_MB={MEASURED_LIDAR_SIZE_MB}")
        cost_config = make_balanced_config(link_rate_Mbps=args.link_rate_mbps)
    # ── Normalisation denominators ─────────────────────────────────────────
    # Keyed per model so different det models don't share the same denominator.
    # Priority:
    #   1. --max_inference_ms CLI (explicit override, any model)
    #   2. measurements/inference_norms.json  key = "{com}_{no_rsu|with_rsu}"
    #   3. combined_cost is bandwidth-only (with a warning)
    #
    # The identity run auto-writes to inference_norms.json so you never need
    # to pass --max_inference_ms manually for models you've already baselined.
    rsu_suffix = "with_rsu" if args.rsu else "no_rsu"
    norm_key = f"{args.com}_{rsu_suffix}"
    inference_norms_path = os.path.join(
        args.measurements_dir or "measurements", "inference_norms.json"
    )

    def _load_inference_norms():
        if os.path.isfile(inference_norms_path):
            with open(inference_norms_path) as _f:
                return json.load(_f)
        return {}

    # Late-fusion mode: vehicles detect on-board, send only boxes to RSU.
    # apply_late_fusion=1 without --selection means we're running the late-fusion
    # lowerbound baseline.  Switch the cost model so bandwidth reflects box payload
    # and latency reflects one parallel precomp+inference+box-upload chain.
    is_late_fusion = (apply_late_fusion == 1 and not args.selection)
    cost_config.late_fusion = is_late_fusion
    if is_late_fusion:
        print("Cost model: late-fusion mode (box payload only, parallel per-agent latency)")

    if args.max_inference_ms and args.max_inference_ms > 0:
        cost_config.compute_normalization_constants(
            ref_inference_ms=args.max_inference_ms,
            ref_inference_energy_J=args.max_inference_energy_J if args.max_inference_energy_J > 0 else None,
        )
        print(f"Cost norms from CLI: {norm_key} → {args.max_inference_ms:.1f} ms")
    else:
        _norms = _load_inference_norms()
        if norm_key in _norms:
            _entry = _norms[norm_key]
            cost_config.compute_normalization_constants(
                ref_inference_ms=_entry["avg_inference_ms"],
                ref_inference_energy_J=_entry.get("avg_inference_energy_J", 0.0) or None,
            )
            print(f"Cost norms from inference_norms.json: {norm_key} → {_entry['avg_inference_ms']:.1f} ms")
        elif cost_config.norms_calibrated:
            # late_fusion flag was set after from_measurements_dir already ran _compute_norms,
            # so recompute with the correct mode now.
            cost_config._compute_norms()
            print(f"Cost norms from measurements: inference_norm_ms={cost_config.inference_norm_ms:.1f} ms")
        else:
            print(
                f"WARNING: no norm found for '{norm_key}' in {inference_norms_path}. "
                "Run: make test sel_method=identity  to populate it automatically. "
                "Until then, combined_cost is bandwidth-only."
            )
    frame_cost_rows: list = []
    n_available = 0
    # per-scene accumulators: scene_id -> list of n_selected values
    scene_selected: dict = {}
    scene_available: dict = {}

    # ONE energy tracker for the whole run (per-frame trackers cost ~3 s each).
    run_tracker = _start_run_tracker()
    run_wall_t0 = time.time()

    for cnt, sample in enumerate(validation_data_loader):
        t = time.time()
        (
            padded_voxel_point_list,
            padded_voxel_points_teacher_list,
            label_one_hot_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_map_list,
            vis_maps_list,
            gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*sample)

        filename0 = filenames[0][0][0]

        parts = filename0.split(os.sep)
        scene_frame = parts[-2]  # e.g. "12_5"
        scene_id, frame_id = map(int, scene_frame.split("_"))

        if scene_id < args.scene_begin or scene_id >= args.scene_end:
            print(f"Skipping scene {scene_id} as it is outside the range [{args.scene_begin}, {args.scene_end})")
            continue

        if args.selection:
            # --------------------------------------------------
            # Agent selection: build state features, run policy
            # --------------------------------------------------

            state_feats, metas = build_state_features(
                state_index=state_index,
                scene_id=scene_id,
                frame_id=frame_id,
                return_meta=True,
            )
            state_feats = state_feats.to(device)
            n_available = len(metas)
            agent_id_to_idx = {
                agent_id: idx for idx, agent_id in enumerate(agent_idx_range)
            }

            # Extract transform table from the RSU's data (always trans_matrices_list[0]
            # because agent_idx_range[0] = 0 when --rsu 1).
            # Row j = T_{agent_j → RSU_frame}; translation [j, 0:2, 3] = vehicle position.
            # IMPORTANT: this is only valid when --rsu 1 (RSU is agent_idx_range[0]).
            if not args.rsu:
                raise RuntimeError(
                    "--selection requires --rsu 1: distance-based policies need "
                    "T_{j→RSU} from agent 0's trans_matrices."
                )
            _ref_table = torch.as_tensor(trans_matrices_list[0])  # (1, N_agents, 4, 4)

            # sel_model is loaded once before the loop (see above).

            selected_state_indices = select_agents_from_metadata(
                state_feats,
                method=SelectionMethod(args.sel_method),
                model=sel_model,
                K=args.K,
                threshold=args.sel_threshold,
                budget_bytes=int(args.budget_mb * 1024 * 1024),
                rsu_trans=_ref_table[0],   # (N_agents, 4, 4)
                metas=metas,
            )
            selected_indices = [
                agent_id_to_idx[metas[i].agent_id]
                for i in selected_state_indices
                if metas[i].agent_id in agent_id_to_idx
            ]
            # Record the actual agent IDs (not internal indices) for the frame log
            selected_agent_ids_str = ",".join(
                str(metas[i].agent_id)
                for i in selected_state_indices
                if metas[i].agent_id in agent_id_to_idx
            )
            if not selected_indices:
                # No agents selected (e.g. bandwidth budget too tight for even one agent).
                # Record a zero-cost frame — no bandwidth spent, no inference, no detections.
                print(
                    f"scene {scene_id} frame {frame_id}: "
                    "0 agents selected — recording zero-cost frame, skipping inference."
                )
                zero_cost = compute_cost(0, 0.0, 0.0, cost_config)
                frame_cost_rows.append({
                    "scene_id": scene_id, "frame_id": frame_id,
                    "flag": flag,
                    "sel_method": args.sel_method if args.selection else "all_agents",
                    "n_available": n_available, "n_selected": 0,
                    "selected_agent_ids": "",
                    "bandwidth_MB": 0.0, "upload_ms": 0.0, "precomp_ms": 0.0,
                    "inference_ms": 0.0, "latency_total_ms": 0.0,
                    "precomp_energy_J": 0.0, "inference_energy_J": 0.0,
                    "total_energy_J": 0.0, "combined_cost": 0.0,
                })
                scene_selected.setdefault(scene_id, []).append(0)
                scene_available.setdefault(scene_id, []).append(n_available)
                frames_processed += 1
                continue

            selected_num_agents = len(selected_indices)
            n_for_cost = selected_num_agents

            # --- Phase 2: RSU-centric mode ---
            # With --rsu 1, index 0 in the loaded data is the RSU (agent 0).
            # Save its GT boxes and supervision tensors BEFORE _sel() filters
            # the lists down to selected vehicles. Evaluation is against RSU GT;
            # supervision tensors are prepended to assemble_detection_inputs so
            # the reference frame labels/anchors come from the RSU.
            if args.rsu:
                rsu_gt_max_iou = gt_max_iou[0]
                rsu_label_one_hot = label_one_hot_list[0]
                rsu_reg_target = reg_target_list[0]
                rsu_reg_loss_mask = reg_loss_mask_list[0]
                rsu_anchors_map = anchors_map_list[0]
                rsu_vis_maps = vis_maps_list[0]
            else:
                rsu_gt_max_iou = None

            trans_sel = [_ref_table[0, j] for j in selected_indices]  # list of (4,4)

            def _sel(lst):
                return [lst[i] for i in selected_indices]

            padded_voxel_point_list = _sel(padded_voxel_point_list)
            padded_voxel_points_teacher_list = _sel(padded_voxel_points_teacher_list)
            label_one_hot_list = _sel(label_one_hot_list)
            reg_target_list = _sel(reg_target_list)
            reg_loss_mask_list = _sel(reg_loss_mask_list)
            anchors_map_list = _sel(anchors_map_list)
            vis_maps_list = _sel(vis_maps_list)
            target_agent_id_list = _sel(target_agent_id_list)
            num_agent_list = _sel(num_agent_list)
            # trans_matrices_list is NOT passed through _sel — trans_sel above replaces it.
            gt_max_iou = _sel(list(gt_max_iou))

            # Bandwidth cost: size of an individual BEV upload (pvp_list, lowerbound).
            # Each selected vehicle transmits its own unseen individual BEV, not the
            # pre-fused oracle teacher BEV.
            _valid_pvp = [
                torch.as_tensor(t)
                for t in padded_voxel_point_list
                if t is not None and not isinstance(t, list)
                and torch.as_tensor(t).numel() > 0
            ]
            bev_size_bytes = (
                bev_size_from_tensor(torch.stack(_valid_pvp, dim=0))
                if _valid_pvp
                else 0
            )

            # In RSU-centric mode, prepend RSU's supervision tensors so that
            # assemble_detection_inputs picks the RSU frame's labels/anchors as
            # the reference (first-non-empty) for loss computation.
            if args.rsu:
                assemble_labels      = [rsu_label_one_hot]   + list(label_one_hot_list)
                assemble_reg_target  = [rsu_reg_target]      + list(reg_target_list)
                assemble_reg_mask    = [rsu_reg_loss_mask]   + list(reg_loss_mask_list)
                assemble_anchors_map = [rsu_anchors_map]     + list(anchors_map_list)
                assemble_vis_maps    = [rsu_vis_maps]        + list(vis_maps_list)
            else:
                assemble_labels      = label_one_hot_list
                assemble_reg_target  = reg_target_list
                assemble_reg_mask    = reg_loss_mask_list
                assemble_anchors_map = anchors_map_list
                assemble_vis_maps    = vis_maps_list

            # Fuse selected vehicles' individual BEVs into the reference frame.
            # trans_sel carries a (4,4) matrix per vehicle: vehicle_j → reference frame.
            data = assemble_detection_inputs(
                config,
                padded_voxel_point_list,
                padded_voxel_points_teacher_list,
                assemble_labels,
                assemble_reg_target,
                assemble_reg_mask,
                assemble_anchors_map,
                assemble_vis_maps,
                target_agent_id_list,
                num_agent_list,
                trans_sel,
                device,
            )

            trans_matrices = data["trans_matrices"]
            num_all_agents = data["num_agent"]
            padded_voxel_points = data["bev_seq"]
            label_one_hot = data["labels"]
            reg_target = data["reg_targets"]
            anchors_map = data["anchors"]
            filename0 = filenames[0]
        else:
            filename0 = filenames[0]
            trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
            target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
            num_all_agents = torch.stack(tuple(num_agent_list), 1)

            # add pose noise
            if pose_noise > 0:
                apply_pose_noise(pose_noise, trans_matrices)

            if not args.rsu:
                num_all_agents -= 1

            if flag == "upperbound":
                padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
            else:
                padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)

            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            reg_target = torch.cat(tuple(reg_target_list), 0)
            reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
            anchors_map = torch.cat(tuple(anchors_map_list), 0)
            vis_maps = torch.cat(tuple(vis_maps_list), 0)

            data = {
                "bev_seq": padded_voxel_points.to(device),
                "labels": label_one_hot.to(device),
                "reg_targets": reg_target.to(device),
                "anchors": anchors_map.to(device),
                "vis_maps": vis_maps.to(device),
                "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
                "target_agent_ids": target_agent_ids.to(device),
                "num_agent": num_all_agents.to(device),
                "trans_matrices": trans_matrices.to(device),
            }
            n_for_cost = len(agent_idx_range)
            n_available = n_for_cost

            # Save RSU GT before any filtering — used by the late-fusion eval branch.
            if args.rsu:
                rsu_gt_max_iou = gt_max_iou[0]
            else:
                rsu_gt_max_iou = None

            _teacher_tensors = [
                torch.as_tensor(t)
                for t in padded_voxel_points_teacher_list
                if t is not None and torch.as_tensor(t).numel() > 0
            ]
            if _teacher_tensors:
                bev_size_bytes = bev_size_from_tensor(torch.stack(_teacher_tensors, dim=0))
            else:
                # lowerbound: no teacher BEVs; measure individual BEVs instead
                _lb_tensors = [
                    torch.as_tensor(t)
                    for t in padded_voxel_point_list
                    if t is not None and torch.as_tensor(t).numel() > 0
                ]
                bev_size_bytes = bev_size_from_tensor(torch.stack(_lb_tensors, dim=0)) if _lb_tensors else 0

        if args.selection:
            predict_out, inference_time_ms = _timed_forward(
                device,
                lambda: fafmodule.predict_all(data, 1, num_agent=1),
            )
            loss, cls_loss, loc_loss, result = predict_out
        elif flag == "lowerbound_box_com":
            predict_out, inference_time_ms = _timed_forward(
                device,
                lambda: fafmodule.predict_all_with_box_com(
                    data, data["trans_matrices"]
                ),
            )
            loss, cls_loss, loc_loss, result = predict_out
        elif flag == "disco":
            predict_out, inference_time_ms = _timed_forward(
                device,
                lambda: fafmodule.predict_all(data, 1, num_agent=num_agent),
            )
            loss, cls_loss, loc_loss, result, save_agent_weight_list = predict_out
        else:
            predict_out, inference_time_ms = _timed_forward(
                device,
                lambda: fafmodule.predict_all(data, 1, num_agent=num_agent),
            )
            loss, cls_loss, loc_loss, result = predict_out

        # Energy is filled in after the loop (distributed from the run-level
        # tracker total). Use 0 here — combined_cost depends on energy only when
        # alpha_energy > 0, and the post-loop pass recomputes it then.
        comm_cost = compute_cost(n_for_cost, inference_time_ms, 0.0, cost_config)
        if args.selection:
            _frame_agent_ids = selected_agent_ids_str
        else:
            # All-agents baseline: record all agents in the current idx range
            _frame_agent_ids = ",".join(str(a) for a in agent_idx_range)
        frame_cost_rows.append({
            "scene_id": scene_id,
            "frame_id": frame_id,
            "flag": flag,
            "sel_method": args.sel_method if args.selection else "all_agents",
            "n_available": n_available,
            "n_selected": n_for_cost,
            "selected_agent_ids": _frame_agent_ids,
            "bandwidth_MB": comm_cost.bandwidth_MB,
            "upload_ms": comm_cost.upload_ms,
            "precomp_ms": comm_cost.precomp_ms,
            "inference_ms": comm_cost.inference_ms,
            "latency_total_ms": comm_cost.latency_total_ms,
            "precomp_energy_J": comm_cost.precomp_energy_J,
            "inference_energy_J": comm_cost.inference_energy_J,
            "total_energy_J": comm_cost.total_energy_J,
            "combined_cost": comm_cost.combined_cost,
        })
        # accumulate per-scene stats
        scene_selected.setdefault(scene_id, []).append(n_for_cost)
        scene_available.setdefault(scene_id, []).append(n_available)

        box_color_map = ["red", "yellow", "blue", "purple", "black", "orange"]

        if args.selection:
            eval_start_idx = 0
            num_agent = 1
            # late_fusion_n_agents: how many result slots late_fusion() should scan.
            # For selection mode there is only 1 agent, so 1.
            late_fusion_n_agents = 1
        elif is_late_fusion and args.rsu:
            # Late fusion at RSU: FaFNet predicted on all N agents' individual BEVs.
            # late_fusion() merges agents 1..N-1 boxes into agent 0 (RSU frame).
            # We then evaluate only agent 0 against RSU GT.
            # IMPORTANT: num_agent must stay at its original value so late_fusion()
            # can scan all result slots when merging.  eval_start/end control the
            # evaluation loop independently.
            eval_start_idx = 0
            eval_end_idx = 1          # only evaluate RSU (k=0)
            late_fusion_n_agents = num_agent   # full count for the merge
        elif args.eval_start_idx >= 0:
            eval_start_idx = args.eval_start_idx
            eval_end_idx = num_agent
            late_fusion_n_agents = num_agent
        elif args.rsu and num_agent == 1:
            eval_start_idx = 0
            eval_end_idx = num_agent
            late_fusion_n_agents = num_agent
        else:
            eval_start_idx = 1 if args.rsu else 0
            eval_end_idx = num_agent
            late_fusion_n_agents = num_agent

        if args.selection:
            eval_end_idx = num_agent   # num_agent already set to 1 above

        # local qualitative evaluation
        for k in range(eval_start_idx, eval_end_idx):
            box_colors = None
            # Late fusion is not applicable in selection mode: BEVs have already
            # been fused at the data level before the detector runs.
            run_late_fusion = apply_late_fusion == 1 and not args.selection
            if run_late_fusion and len(result[k]) != 0:
                pred_restore = result[k][0][0][0]["pred"]
                score_restore = result[k][0][0][0]["score"]
                selected_idx_restore = result[k][0][0][0]["selected_idx"]

            data_agents = {
                "bev_seq": torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1),
                "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
                "anchors": torch.unsqueeze(anchors_map[k, :, :, :, :], 0),
            }
            # Use RSU GT when: (a) selection mode with RSU, (b) late-fusion at RSU.
            # In both cases all vehicle detections are evaluated in the RSU's AOI.
            # IMPORTANT: gt_max_iou has already been filtered by _sel() so index 0
            # is the first *selected vehicle*, not the RSU.  Use rsu_gt_max_iou
            # which was saved before _sel() was applied.
            if (args.rsu and args.selection) or (is_late_fusion and args.rsu and k == 0):
                temp = rsu_gt_max_iou  # RSU GT saved before _sel() filtered the list
            else:
                temp = gt_max_iou[k]

            if len(temp[0]["gt_box"]) == 0:
                data_agents["gt_max_iou"] = []
            else:
                data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]

            # late fusion
            if run_late_fusion and len(result[k]) != 0:
                # When the RSU is a passive relay (no_rsu_detect), suppress its
                # own lidar predictions before fusing so only vehicle boxes
                # contribute.  pred_restore/score_restore already saved above.
                if args.no_rsu_detect and k == 0:
                    result[k][0][0][0]["pred"] = np.empty((0, 1, 4, 2))
                    result[k][0][0][0]["score"] = np.array([])
                    result[k][0][0][0]["selected_idx"] = np.array([])
                box_colors = late_fusion(
                    k, late_fusion_n_agents, result, trans_matrices, box_color_map
                )

            # ── RSU AOI filter ───────────────────────────────────────────────
            # Whenever we evaluate against RSU GT, clip predictions to the RSU's
            # area_extents before scoring.  In late-fusion mode, vehicles detect
            # in their own frame (±range around the vehicle) and boxes transformed
            # into RSU frame can land far outside the RSU's ±32 m grid — those
            # have no matching GT and count as false positives, depressing mAP.
            # In BEV-upload (selection) mode, the model grid already enforces the
            # bounds, so this is a no-op in practice but keeps the logic consistent.
            #
            # NOTE: late_fusion's NMS only filters "pred" and box_colors; "score"
            # and "selected_idx" are left at their pre-NMS length.  Only filter
            # arrays that are post-NMS and aligned with "pred".
            using_rsu_gt = (args.rsu and args.selection) or (is_late_fusion and args.rsu and k == 0)
            if using_rsu_gt and len(result[k]) > 0 and len(result[k][0][0][0]["pred"]) > 0:
                _boxes = result[k][0][0][0]["pred"]      # (M, 1, 4, 2)
                _centers = _boxes[:, 0, :, :].mean(axis=1)   # (M, 2)
                _x_min, _x_max = float(config.area_extents[0][0]), float(config.area_extents[0][1])
                _y_min, _y_max = float(config.area_extents[1][0]), float(config.area_extents[1][1])
                _in_aoi = (
                    (_centers[:, 0] >= _x_min) & (_centers[:, 0] <= _x_max) &
                    (_centers[:, 1] >= _y_min) & (_centers[:, 1] <= _y_max)
                )
                result[k][0][0][0]["pred"] = _boxes[_in_aoi]
                if box_colors is not None:
                    box_colors = box_colors[_in_aoi]
            # ── end RSU AOI filter ───────────────────────────────────────────

            result_temp = result[k]

            temp = {
                "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
                "result": [] if len(result_temp) == 0 else result_temp[0][0],
                "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
                "anchors_map": data_agents["anchors"].cpu().numpy()[0],
                "gt_max_iou": data_agents["gt_max_iou"],
            }
            det_results_local[k], annotations_local[k] = cal_local_mAP(
                config, temp, det_results_local[k], annotations_local[k]
            )
            # also accumulate per-scene for scene-level AP/recall
            _sd = scene_det_results.setdefault(scene_id, [])
            _sa = scene_annotations.setdefault(scene_id, [])
            scene_det_results[scene_id], scene_annotations[scene_id] = cal_local_mAP(
                config, temp, _sd, _sa
            )

            filename = str(filename0[0][0])
            cut = filename[filename.rfind("agent") + 7 :]
            seq_name = cut[: cut.rfind("_")]
            idx = cut[cut.rfind("_") + 1 : cut.rfind("/")]
            seq_save = os.path.join(save_fig_path[k], seq_name)
            check_folder(seq_save)
            idx_save = str(idx) + ".png"
            temp_ = deepcopy(temp)
            if args.visualization:
                visualization(
                    config,
                    temp,
                    box_colors,
                    box_color_map,
                    apply_late_fusion,
                    os.path.join(seq_save, idx_save),
                )

            # plot the cell-wise edge
            if flag == "disco" and args.visualization and k < len(save_agent_weight_list):
                one_agent_edge = save_agent_weight_list[k]
                for kk in range(len(one_agent_edge)):
                    idx_edge_save = (
                        str(idx) + "_edge_" + str(kk) + "_to_" + str(k) + ".png"
                    )
                    savename_edge = os.path.join(seq_save, idx_edge_save)
                    sns.set()
                    sns.heatmap(one_agent_edge[kk].cpu())
                    plt.savefig(savename_edge, dpi=500)
                    plt.clf()
                    plt.close(0)

            # == tracking ==
            if args.tracking:
                scene, frame = filename.split("/")[-2].split("_")
                det_file = os.path.join(tracking_path[k], f"det_{scene}.txt")
                if scene not in tracking_file[k]:
                    det_file = open(det_file, "w")
                    tracking_file[k].add(scene)
                else:
                    det_file = open(det_file, "a")
                det_corners = get_det_corners(config, temp_)
                for ic, c in enumerate(det_corners):
                    det_file.write(
                        ",".join(
                            [
                                str(
                                    int(frame) + 1
                                ),  # frame idx is 1-based for tracking
                                "-1",
                                "{:.2f}".format(c[0]),
                                "{:.2f}".format(c[1]),
                                "{:.2f}".format(c[2]),
                                "{:.2f}".format(c[3]),
                                str(result_temp[0][0][0]["score"][ic]),
                                "-1",
                                "-1",
                                "-1",
                            ]
                        )
                        + "\n"
                    )
                    det_file.flush()

                det_file.close()

            # restore data before late-fusion
            if run_late_fusion and len(result[k]) != 0:
                result[k][0][0][0]["pred"] = pred_restore
                result[k][0][0][0]["score"] = score_restore
                result[k][0][0][0]["selected_idx"] = selected_idx_restore

        print("Validation scene {}, at frame {}".format(seq_name, idx))
        print("Communication cost: {}".format(comm_cost))
        print("Takes {} s\n".format(str(time.time() - t)))
        frames_processed += 1

    # ── Stop run tracker and distribute measured energy across frames ─────────
    # codecarbon measures the whole run (data loading + assemble + inference).
    # We attribute per-frame inference energy proportionally to inference time,
    # so the per-frame numbers sum exactly to the measured run total.
    run_wall_s = time.time() - run_wall_t0
    total_run_energy_J = 0.0
    if run_tracker is not None:
        try:
            run_tracker.stop()   # ~3 s, but ONCE for the whole run
            total_run_energy_J = _tracker_energy_J(run_tracker)
        except Exception as e:
            print(f"WARNING: codecarbon stop failed ({e}); energy=0")

    total_inf_ms = sum(r["inference_ms"] for r in frame_cost_rows) or 1.0
    for r in frame_cost_rows:
        share = r["inference_ms"] / total_inf_ms
        inf_energy_batch = total_run_energy_J * share   # total for all agents in this frame
        # Late fusion: each vehicle runs inference independently.
        # compute_cost expects per-agent inference energy so it can compute N × inference_J.
        # BEV-upload: one centralized inference — pass the batch total directly.
        if is_late_fusion:
            inf_energy_for_cost = inf_energy_batch / max(r["n_selected"], 1)
        else:
            inf_energy_for_cost = inf_energy_batch
        cc = compute_cost(r["n_selected"], r["inference_ms"], inf_energy_for_cost, cost_config)
        r["inference_energy_J"] = cc.inference_energy_J
        r["total_energy_J"] = cc.total_energy_J
        r["combined_cost"] = cc.combined_cost
    print(
        f"Run energy: {total_run_energy_J:.2f} J over {run_wall_s:.1f} s "
        f"({frames_processed} frames)"
    )

    if frames_processed == 0:
        raise RuntimeError(
            "No frames were evaluated. Check --scene_begin/--scene_end against "
            "scenes present in --data_prep (test split commonly uses scenes "
            "such as 5, 8, and 19)."
        )

    logger_root = args.logpath if args.logpath != "" else "/mnt/10TB/balintkiraly/results"
    logger_root = os.path.join(
        logger_root, f"{flag}_eval", "with_rsu" if args.rsu else "no_rsu"
    )
    os.makedirs(logger_root, exist_ok=True)
    log_file_path = os.path.join(logger_root, "log_test.txt")
    log_file = open(log_file_path, "w")

    def print_and_write_log(log_str):
        print(log_str)
        log_file.write(log_str + "\n")

    mean_ap_local = []
    # local mAP evaluation
    det_results_all_local = []
    annotations_all_local = []
    for k in range(eval_start_idx, num_agent):
        if type(det_results_local[k]) != list or len(det_results_local[k]) == 0:
            continue

        print_and_write_log("Local mAP@0.5 from agent {}".format(k))
        mean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap)
        print_and_write_log("Local mAP@0.7 from agent {}".format(k))

        mean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.7,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap)

        det_results_all_local += det_results_local[k]
        annotations_all_local += annotations_local[k]

    mean_ap_local_average, _eval_res_05 = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.5,
        dataset=None,
        logger=None,
    )
    mean_ap_local.append(mean_ap_local_average)

    mean_ap_local_average, _eval_res_07 = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.7,
        dataset=None,
        logger=None,
    )
    mean_ap_local.append(mean_ap_local_average)

    # Extract max recall achieved (last point on P-R curve) for each threshold.
    # eval_results[0] is the single vehicle class; recall is a 1-D array.
    def _max_recall(eval_res):
        try:
            r = eval_res[0]["recall"]
            return float(r[-1]) if len(r) > 0 else 0.0
        except Exception:
            return 0.0

    recall_05 = _max_recall(_eval_res_05)
    recall_07 = _max_recall(_eval_res_07)

    print_and_write_log(
        "Quantitative evaluation results of model from {}, at epoch {}".format(
            args.resume, start_epoch - 1
        )
    )

    # mean_ap_local has 2 entries per evaluated agent plus 2 for the overall average.
    # Derive the count from the list length rather than num_agent so late-fusion RSU
    # mode (1 agent evaluated) and standard multi-agent mode both work correctly.
    n_evaluated_agents = (len(mean_ap_local) - 2) // 2
    for k in range(n_evaluated_agents):
        print_and_write_log(
            "agent{} mAP@0.5 is {} and mAP@0.7 is {}".format(
                k + 1, mean_ap_local[k * 2], mean_ap_local[(k * 2) + 1]
            )
        )

    print_and_write_log(
        "average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(
            mean_ap_local[-2], mean_ap_local[-1]
        )
    )

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    # Top-level results directory (summary.csv lives here and aggregates runs).
    csv_dir = args.csv_path if args.csv_path else logger_root

    # Infer split name from data path  (…/V2X-Sim-det/val  →  "val")
    _data_parts = os.path.normpath(args.data_prep).split(os.sep)
    split_name = _data_parts[-1] if _data_parts[-1] in ("train", "val", "test") else "unknown"

    sel_method_label = args.sel_method if args.selection else "all_agents"

    # ── Run identity: timestamp + descriptive run_id ────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenes_evaluated = sorted(scene_selected.keys())
    scene_range = f"{args.scene_begin}-{args.scene_end}"
    run_id = f"{timestamp}_{flag}_{sel_method_label}"
    if args.selection and sel_method_label == "bandwidth":
        run_id += f"_b{args.budget_mb}"
    elif args.selection and sel_method_label in ("closest_k", "velocity", "heuristic"):
        run_id += f"_K{args.K}"

    # Per-run subdirectory: never clobbers previous runs' frame-level CSVs.
    run_dir = os.path.join(csv_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print_and_write_log(f"Run outputs -> {run_dir}")

    # ── Per-frame cost CSV (one fresh file per run) ─────────────────────────
    frame_csv_path = os.path.join(run_dir, "frame_costs.csv")
    if frame_cost_rows:
        frame_fieldnames = list(frame_cost_rows[0].keys())
        with open(frame_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=frame_fieldnames)
            writer.writeheader()
            writer.writerows(frame_cost_rows)
        print_and_write_log(f"Per-frame cost CSV saved to {frame_csv_path}")

    # ── Per-scene CSV (one fresh file per run) ──────────────────────────────
    scene_csv_path = os.path.join(run_dir, "scene_stats.csv")

    def _scene_ap_recall(scene_id):
        """Compute mAP@0.5, mAP@0.7, recall@0.5, recall@0.7 for one scene."""
        dets = scene_det_results.get(scene_id, [])
        anns = scene_annotations.get(scene_id, [])
        if not dets or not anns:
            return 0.0, 0.0, 0.0, 0.0
        try:
            ap05, er05 = eval_map(dets, anns, scale_ranges=None, iou_thr=0.5, dataset=None, logger=None)
            ap07, er07 = eval_map(dets, anns, scale_ranges=None, iou_thr=0.7, dataset=None, logger=None)
            def _last_recall(er):
                try:
                    r = er[0]["recall"]
                    return float(r[-1]) if len(r) > 0 else 0.0
                except Exception:
                    return 0.0
            return float(ap05), float(ap07), _last_recall(er05), _last_recall(er07)
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    scene_rows = []
    for sid in scenes_evaluated:
        sel_vals = scene_selected[sid]
        avail_vals = scene_available[sid]
        s_ap05, s_ap07, s_rec05, s_rec07 = _scene_ap_recall(sid)
        scene_rows.append({
            "timestamp": timestamp,
            "scene_id": sid,
            "split": split_name,
            "com": args.com,
            "sel_method": sel_method_label,
            "K": args.K,
            "budget_mb": args.budget_mb,
            "n_frames": len(sel_vals),
            "avg_n_available": sum(avail_vals) / max(1, len(avail_vals)),
            "avg_n_selected": sum(sel_vals) / max(1, len(sel_vals)),
            "mAP_05": s_ap05,
            "mAP_07": s_ap07,
            "recall_05": s_rec05,
            "recall_07": s_rec07,
        })
    if scene_rows:
        with open(scene_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(scene_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scene_rows)
        print_and_write_log(f"Per-scene stats CSV saved to {scene_csv_path}")

    # ── Summary CSV (top-level, appends one row per run for comparison) ─────
    summary_csv_path = os.path.join(csv_dir, "summary.csv")
    def _avg(key):
        return sum(r[key] for r in frame_cost_rows) / max(1, len(frame_cost_rows))

    all_n_selected = [r["n_selected"] for r in frame_cost_rows]
    summary_row = {
        "timestamp": timestamp,
        "run_id": run_id,
        "com": args.com,
        "flag": flag,
        "split": split_name,
        "selection": int(args.selection),
        "sel_method": sel_method_label,
        "scene_range": scene_range,
        "scenes_evaluated": ";".join(str(s) for s in scenes_evaluated),
        "K": args.K,
        "budget_mb": args.budget_mb,
        "link_rate_mbps": args.link_rate_mbps,
        "rsu": args.rsu,
        "total_scenes": len(scene_selected),
        "n_frames": frames_processed,
        "run_wall_s": round(run_wall_s, 1),
        "mAP_05": mean_ap_local[-2],
        "mAP_07": mean_ap_local[-1],
        "recall_05": recall_05,
        "recall_07": recall_07,
        "avg_n_selected": sum(all_n_selected) / max(1, len(all_n_selected)),
        "avg_bandwidth_MB": _avg("bandwidth_MB"),
        "avg_upload_ms": _avg("upload_ms"),
        "avg_precomp_ms": _avg("precomp_ms"),
        "avg_inference_ms": _avg("inference_ms"),
        "avg_latency_total_ms": _avg("latency_total_ms"),
        "avg_precomp_energy_J": _avg("precomp_energy_J"),
        "avg_inference_energy_J": _avg("inference_energy_J"),
        "avg_total_energy_J": _avg("total_energy_J"),
        "total_run_energy_J": round(total_run_energy_J, 3),
        "avg_combined_cost": _avg("combined_cost"),
    }
    write_header = not os.path.exists(summary_csv_path)
    with open(summary_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    print_and_write_log(f"Summary CSV appended to {summary_csv_path}")

    # ── Per-run summary JSON (self-contained, lives in run_dir) ─────────────
    run_summary_path = os.path.join(run_dir, "summary.json")
    with open(run_summary_path, "w") as f:
        json.dump(summary_row, f, indent=2)
    print_and_write_log(f"Per-run summary saved to {run_summary_path}")

    # ── Auto-write inference norm for this model (identity runs only) ────────
    # Any run with sel_method=identity is by definition the all-agents baseline
    # for this model. Write its avg_inference_ms into inference_norms.json so
    # all subsequent runs with the same model never need --max_inference_ms.
    if sel_method_label == "identity":
        _norms = _load_inference_norms()
        _norms[norm_key] = {
            "avg_inference_ms": summary_row["avg_inference_ms"],
            "avg_inference_energy_J": summary_row["avg_inference_energy_J"],
            "recorded_at": timestamp,
            "run_id": run_id,
        }
        os.makedirs(os.path.dirname(os.path.abspath(inference_norms_path)), exist_ok=True)
        with open(inference_norms_path, "w") as f:
            json.dump(_norms, f, indent=2)
        print_and_write_log(
            f"Inference norm saved: {norm_key} → "
            f"{summary_row['avg_inference_ms']:.1f} ms  ({inference_norms_path})"
        )

    # ── Run metadata JSON (full provenance for the thesis) ──────────────────
    metadata = {
        **summary_row,
        "resume_checkpoint": args.resume,
        "data_prep": args.data_prep,
        "apply_late_fusion": args.apply_late_fusion,
        "num_agent": args.num_agent,
        "max_inference_ms": args.max_inference_ms,
        "cost_norms_calibrated": cost_config.norms_calibrated,
        "norm_key": norm_key,
        "late_fusion": is_late_fusion,
        "detection_size_MB": cost_config.detection_size_MB,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default=None,
        type=str,
        help="The path to the original dataset",
    )
    parser.add_argument(
        "--data_prep",
        default=None,
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument(
        "--state_path",
        default=None,
        type=str,
        help="The path to the state feature files for agent selection",
    )
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=4, type=int, help="Number of DataLoader workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument(
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--warp_flag", default=0, type=int, help="Whether to use pose info for When2com"
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--visualization", type=int, default=0, help="Visualize validation result"
    )
    parser.add_argument(
        "--com",
        default="",
        type=str,
        help="lowerbound/upperbound/disco/when2com/v2v/sum/mean/max/cat/agent",
    )
    parser.add_argument("--inference", type=str)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--box_com", action="store_true")
    parser.add_argument("--rsu", default=0, type=int, help="0: no RSU, 1: RSU")
    # scene_batch => batch size in each scene
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--apply_late_fusion",
        default=0,
        type=int,
        help="1: apply late fusion. 0: no late fusion",
    )
    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )
    parser.add_argument(
        "--no_rsu_detect",
        default=0,
        type=int,
        help=(
            "1: suppress RSU lidar predictions before late fusion so only vehicle "
            "boxes contribute (RSU acts as passive relay). "
            "Requires --apply_late_fusion 1 --eval_start_idx 0 --rsu 1."
        ),
    )
    parser.add_argument(
        "--eval_start_idx",
        default=-1,
        type=int,
        help=(
            "Override the agent index at which evaluation starts. "
            "-1 (default) = auto: 0 when rsu+num_agent==1 (RSU-only), "
            "1 when rsu>0 (skip RSU slot), 0 otherwise. "
            "Set to 0 to evaluate from RSU frame (e.g. late-fusion RSU baseline)."
        ),
    )
    parser.add_argument(
        "--scene_begin",
        default=0,
        type=int,
        help="The begin index of scenes to be evaluated",
    )
    parser.add_argument(
        "--scene_end",
        default=100,
        type=int,
        help="The end index of scenes to be evaluated",
    )
    parser.add_argument(
        "--selection",
        action="store_true",
        help="Whether to perform agent selection",
    )
    parser.add_argument(
        "--sel_method",
        default="identity",
        type=str,
        help="Agent selection method: identity/closest_k/velocity/heuristic/bandwidth/ml_model",
    )
    parser.add_argument(
        "--K",
        default=3,
        type=int,
        help="Number of agents to select (used by closest_k, velocity, heuristic)",
    )
    parser.add_argument(
        "--budget_mb",
        default=2.0,
        type=float,
        help="Bandwidth budget in MB (used by bandwidth_aware strategy)",
    )
    parser.add_argument(
        "--sel_model_path",
        default=None,
        type=str,
        help="Path to the trained agent selection model",
    )
    parser.add_argument(
        "--sel_threshold",
        default=0.5,
        type=float,
        help="Sigmoid threshold for ml_model selection (default 0.5). "
             "Sweep 0.3/0.5/0.7 to trace the quality/cost Pareto curve.",
    )
    parser.add_argument(
        "--csv_path",
        default="",
        type=str,
        help="Directory to write CSV outputs (frame_costs.csv, summary.csv). "
             "Defaults to the same directory as log_test.txt.",
    )
    parser.add_argument(
        "--link_rate_mbps",
        default=10.0,
        type=float,
        help="Simulated uplink bandwidth in Mbps for upload latency model (default: 10.0)",
    )
    parser.add_argument(
        "--measurements_dir",
        default="measurements",
        type=str,
        help="Directory containing lidar_size.json and precomp_cost.json from measurement tools. "
             "If the files are missing, falls back to built-in defaults.",
    )
    parser.add_argument(
        "--max_inference_ms",
        default=0.0,
        type=float,
        help=(
            "All-agents baseline inference latency in ms, used as the normalisation "
            "denominator for combined_cost. Obtain from avg_inference_ms in "
            "summary.csv after running: make test sel_method=identity. "
            "If 0 (default), latency/energy terms are omitted from combined_cost."
        ),
    )
    parser.add_argument(
        "--max_inference_energy_J",
        default=0.0,
        type=float,
        help="All-agents inference energy in J for combined_cost normalisation (optional).",
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)
