import argparse
import os
import glob
import pathlib

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="create_data_save_path root (e.g. V2X-Sim-det)")
    parser.add_argument("--split", default="train", help="train/val/test")
    parser.add_argument("--agent", type=int, default=0, help="agent index, e.g. 0..5")
    parser.add_argument("--scene", type=int, default=0, help="scene id, e.g. 0..19 for your case")
    args = parser.parse_args()

    # 1) Go to this agent's split directory
    split_dir = os.path.join(args.data, args.split, f"agent{args.agent}")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    # 2) Find all frames for the given scene, like '0_0', '0_1', ...
    pattern = os.path.join(split_dir, f"{args.scene}_*")
    sample_dirs = sorted(glob.glob(pattern))
    if not sample_dirs:
        raise FileNotFoundError(f"No samples matching {pattern}")

    print(f"Found {len(sample_dirs)} frames for scene {args.scene} in agent{args.agent}")
    sample_dir = sample_dirs[0]
    print(f"Inspecting sample directory: {sample_dir}")

    path = os.path.join(sample_dir, "0.npy")

    # 3) List files and try to torch.load each; print keys/shapes
    arr = np.load(path, allow_pickle=True)
    print(f"Loaded: {path}")
    print(f"type: {type(arr)}, dtype: {getattr(arr, 'dtype', None)}")

    if isinstance(arr, np.ndarray) and arr.ndim == 0:
        obj = arr.item()
        print(f"type(obj) = {type(obj)}")

        if isinstance(obj, dict):
            print("dict keys:", list(obj.keys()))
            for k, v in obj.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', None)}")
                else:
                    text = str(v)
                    if len(text) > 200:
                        text = text[:197] + "..."
                    print(f"  {k}: type={type(v).__name__}, value={text}")
        else:
            # Not a dict; just print a truncated repr
            text = str(obj)
            if len(text) > 500:
                text = text[:497] + "..."
            print("obj repr:", text)
    else:
        print("Not a 0-d array; nothing more to inspect.")


if __name__ == "__main__":
    main()
