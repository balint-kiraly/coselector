"""
Fine-tune a FaFNet detection model on RSU-centred, variable-density merged BEVs.

Design: RSU is a passive relay (Option B)
-----------------------------------------
The RSU has NO self-sensing role.  It only receives BEVs from k selected vehicles,
transforms them to RSU frame, merges, and runs detection.  The RSU's own LiDAR is
NOT included — neither at training time nor at inference.  GT labels are still
evaluated in RSU frame (from agent0's stored annotations).

What this trains
----------------
Warm-starts from `upperbound/no_rsu` (vehicles 1-5 merged, no RSU lidar, agent1 frame)
and fine-tunes so the model handles:
  (a) RSU-frame geometry instead of agent1-frame geometry
  (b) Variable density: k ~ U[min_agents, max_agents] vehicles per step instead
      of always 5 — making the model robust regardless of how many agents the RSU
      selects.

Why `upperbound/no_rsu` and NOT `upperbound/with_rsu`
------------------------------------------------------
`upperbound/with_rsu` was trained with RSU lidar contributing ~48% of the merged BEV
density for agent0 samples.  Since we will NEVER have RSU lidar at inference (Option B),
warm-starting from that checkpoint would require the model to "forget" RSU-lidar features.
`upperbound/no_rsu` was trained on vehicle-only merged BEVs — correct density, only
wrong frame (agent1's frame instead of RSU frame) which fine-tuning corrects in ~10 epochs.

Safe boundaries
---------------
* Warm start from an existing checkpoint (fast convergence, ~20 epochs vs 100).
* Mixed precision (torch.cuda.amp) — cuts VRAM by ~40%, speeds up ~1.5×.
* Early stopping: stop if validation loss doesn't improve for `patience` epochs.
* Save a checkpoint every epoch (resume from any point with --resume).
* Max epochs capped by --nepoch (default 30).
* Learning rate 1e-4 (10× lower than scratch) to avoid catastrophic forgetting.

Typical command
---------------
  python training/train_rsu_centric.py \\
      --data_train /mnt/10TB/balintkiraly/created_data/V2X-Sim-det/train \\
      --data_val   /mnt/10TB/balintkiraly/created_data/V2X-Sim-det/val \\
      --resume     /home/bkiraly/coperception/tools/det/checkpoints/upperbound/no_rsu/epoch_100.pth \\
      --logpath    /home/bkiraly/coselector/checkpoints/rsu_centric \\
      --nepoch     30  --batch_size 4  --patience 7 \\
      --min_agents 1   --max_agents 5  --amp
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Coperception imports
from coperception.configs import Config
from coperception.models.det import FaFNet
from coperception.utils import AverageMeter
from coperception.utils.CoDetModule import FaFModule
from coperception.utils.loss import SoftmaxFocalClassificationLoss, WeightedSmoothL1LocalizationLoss

# Local
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.dataset_rsu_centric import V2XSimDetRsuCentric


# ── helpers ────────────────────────────────────────────────────────────────────

def check_folder(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def collate_fn(batch):
    """
    Custom collator for V2XSimDetRsuCentric.

    Each sample is:
        (padded_bev,       # (1, H, W, C) float32
         label_one_hot,    # (H, W, A, cat) float32
         reg_target,       # (H, W, A, 1, box_code) float32
         reg_loss_mask,    # (H, W, A, 1) bool
         anchors_map,      # (H, W, A, ...) float32
         vis_maps)         # scalar zero  (unused)

    Returns each field stacked along dim-0 (batch dimension).
    """
    bevs, labels, regs, masks, anchors, _ = zip(*batch)
    return (
        torch.from_numpy(np.stack(bevs,    axis=0)),
        torch.from_numpy(np.stack(labels,  axis=0)),
        torch.from_numpy(np.stack(regs,    axis=0)),
        torch.from_numpy(np.stack(masks,   axis=0)),
        torch.from_numpy(np.stack(anchors, axis=0)),
        torch.zeros(len(bevs)),   # vis_maps placeholder
    )


# ── training / validation step (mirrors FaFModule.step) ──────────────────────

def run_epoch(loader, faf_module, device, scaler, train: bool, num_agent: int):
    loss_meter  = AverageMeter("loss",     ":.4f")
    cls_meter   = AverageMeter("cls_loss", ":.4f")
    loc_meter   = AverageMeter("loc_loss", ":.4f")

    faf_module.model.train(train)

    with torch.set_grad_enabled(train):
        pbar = tqdm(loader, desc="train" if train else " val ", leave=False)
        for batch in pbar:
            bev, labels, reg_target, reg_loss_mask, anchors, vis_maps = batch

            # FaFModule expects the BEV in channel-last (H, W, C) format
            # The dataset returns (B, 1, H, W, C) – squeeze the sweep dimension
            bev = bev.squeeze(1)   # (B, H, W, C)

            data = {
                "bev_seq":      bev.to(device),
                "labels":       labels.to(device),
                "reg_targets":  reg_target.to(device),
                "reg_loss_mask":reg_loss_mask.to(device).bool(),
                "anchors":      anchors.to(device),
                "vis_maps":     vis_maps.to(device),
                # FaFNet (lowerbound) ignores these; required by FaFModule.step
                "trans_matrices": torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                    bev.shape[0], 1, 1, 4, 4).to(device),
                "target_agent_ids": torch.zeros(bev.shape[0], 1, dtype=torch.long).to(device),
                "num_agent":        torch.ones(bev.shape[0], 1, dtype=torch.long).to(device),
            }

            ctx = autocast() if (train and scaler is not None) else nullcontext()
            with ctx:
                loss, cls_loss, loc_loss = faf_module.step(
                    data, bev.shape[0], num_agent=num_agent
                )

            if train and scaler is not None:
                # step() already called .backward(); scale the existing gradient
                scaler.step(faf_module.optimizer)
                scaler.update()
                faf_module.optimizer.zero_grad()
            elif train:
                # step() called .backward(); manual optimiser step
                faf_module.optimizer.step()
                faf_module.optimizer.zero_grad()

            loss_meter.update(loss)
            cls_meter.update(cls_loss)
            loc_meter.update(loc_loss)
            pbar.set_postfix(loss=f"{loss:.4f}")

            if np.isnan(loss):
                print("NaN loss detected — aborting.")
                sys.exit(1)

    return loss_meter.avg, cls_meter.avg, loc_meter.avg


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu      = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs: {n_gpu}")

    # ── datasets ──────────────────────────────────────────────────────────────
    common_kw = dict(
        vehicle_agents = list(range(1, args.num_agent)),
        min_agents     = args.min_agents,
        max_agents     = args.max_agents,
        include_rsu_bev= args.include_rsu_bev,
    )
    train_ds = V2XSimDetRsuCentric(args.data_train, cache_size=5000, **common_kw)
    val_ds   = V2XSimDetRsuCentric(args.data_val,   cache_size=1000, **common_kw)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.nworker, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.nworker, collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── model ─────────────────────────────────────────────────────────────────
    config = Config("train", binary=True, only_det=True)
    config.flag = "lowerbound"

    # num_agent=1: the merged BEV is treated as a single-agent input
    model = FaFNet(config, layer=args.layer, kd_flag=0, num_agent=1)
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer  = optim.Adam(model.parameters(), lr=args.lr)
    criterion  = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }
    faf_module = FaFModule(model, model, config, optimizer, criterion, kd_flag=0)

    # ── resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0

    save_path = check_folder(args.logpath)
    log_file  = open(os.path.join(save_path, "log.txt"), "a")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # Load model weights; allow partial load (different num_agent head)
        state = ckpt["model_state_dict"]
        missing, unexpected = faf_module.model.load_state_dict(state, strict=False)
        if missing:
            print(f"Missing keys (expected for warm-start): {len(missing)} keys")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        # Don't restore optimizer/scheduler — fresh fine-tune schedule
        print(f"Warm-started from {args.resume} (epoch {ckpt.get('epoch', '?')})")
        log_file.write(f"Warm-started from {args.resume}\n")
    else:
        print("Training from scratch.")

    # ── mixed precision scaler ────────────────────────────────────────────────
    scaler = GradScaler() if (args.amp and torch.cuda.is_available()) else None
    if scaler:
        print("Mixed precision (AMP) enabled.")

    log_file.write(f"Args: {args}\n")
    log_file.flush()

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.nepoch + 1):
        t0 = time.time()
        lr = faf_module.optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.nepoch}  lr={lr:.2e}")

        train_loss, train_cls, train_loc = run_epoch(
            train_loader, faf_module, device, scaler, train=True, num_agent=1
        )
        val_loss, val_cls, val_loc = run_epoch(
            val_loader,   faf_module, device, scaler, train=False, num_agent=1
        )

        faf_module.scheduler.step()

        elapsed = time.time() - t0
        print(f"  train loss={train_loss:.4f} (cls={train_cls:.4f}, loc={train_loc:.4f})")
        print(f"  val   loss={val_loss:.4f}   (cls={val_cls:.4f}, loc={val_loc:.4f})")
        print(f"  time={elapsed:.1f}s")

        log_file.write(
            f"epoch={epoch}\ttrain={train_loss:.6f}\tval={val_loss:.6f}\t"
            f"cls={train_cls:.6f}\tloc={train_loc:.6f}\ttime={elapsed:.1f}s\n"
        )
        log_file.flush()

        # ── save checkpoint ───────────────────────────────────────────────────
        ckpt_path = os.path.join(save_path, f"epoch_{epoch}.pth")
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     faf_module.model.state_dict(),
            "optimizer_state_dict": faf_module.optimizer.state_dict(),
            "scheduler_state_dict": faf_module.scheduler.state_dict(),
            "train_loss":           train_loss,
            "val_loss":             val_loss,
        }, ckpt_path)

        # ── early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss - 1e-5:
            best_val_loss    = val_loss
            patience_counter = 0
            # Also save as "best.pth"
            torch.save(torch.load(ckpt_path), os.path.join(save_path, "best.pth"))
            print(f"  ✓ New best val loss: {best_val_loss:.6f} → saved best.pth")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    log_file.write(f"Training finished. Best val loss: {best_val_loss:.6f}\n")
    log_file.close()
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved in: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FaFNet on RSU-centric merged BEVs")

    # ── data ─────────────────────────────────────────────────────────────────
    parser.add_argument("--data_train", required=True,
                        help="Path to train split, e.g. .../V2X-Sim-det/train")
    parser.add_argument("--data_val",   required=True,
                        help="Path to val split,   e.g. .../V2X-Sim-det/val")
    parser.add_argument("--num_agent",  default=6, type=int,
                        help="Total agents in dataset (0=RSU, 1..N=vehicles)")

    # ── agent selection ───────────────────────────────────────────────────────
    parser.add_argument("--min_agents", default=1, type=int,
                        help="Minimum vehicles merged per training sample")
    parser.add_argument("--max_agents", default=5, type=int,
                        help="Maximum vehicles merged per training sample")
    parser.add_argument("--include_rsu_bev", action="store_true",
                        help="Always add RSU's own BEV to the merge")

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument("--nepoch",     default=30,   type=int)
    parser.add_argument("--batch_size", default=4,    type=int)
    parser.add_argument("--nworker",    default=4,    type=int)
    parser.add_argument("--lr",         default=1e-4, type=float,
                        help="Learning rate (use ~1e-4 for fine-tuning)")
    parser.add_argument("--patience",   default=7,    type=int,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--amp",        action="store_true",
                        help="Enable automatic mixed precision (AMP / FP16)")
    parser.add_argument("--layer",      default=3, type=int,
                        help="FaFNet communication layer (must match pretrained ckpt)")

    # ── checkpointing ─────────────────────────────────────────────────────────
    parser.add_argument("--resume",  default="",
                        help="Path to a .pth checkpoint to warm-start from")
    parser.add_argument("--logpath", default="checkpoints/rsu_centric",
                        help="Directory for saving checkpoints and logs")

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)
