import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensor",
        type=str,
        help="Type of sensor data ('gnss'/'imu')",
    )
    parser.add_argument(
        "--agent_id",
        type=int,
        default=1,
        help="Agent ID to inspect",
    )
    parser.add_argument(
        "--scene_id",
        type=int,
        default=1,
        help="Scene ID to inspect",
    )
    parser.add_argument(
        "--frame_id",
        type=int,
        default=6,
        help="Frame ID to inspect",
    )
    args = parser.parse_args()

    path = os.path.join("../data/V2X-Sim-2", args.sensor, f"{args.sensor.upper()}_TOP_id_{args.agent_id}", f"scene_{args.scene_id}_{args.frame_id:06d}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading: {path}")
    arr = np.load(path, allow_pickle=True)

    print(f"type: {type(arr)}, dtype: {getattr(arr, 'dtype', None)}, ndim: {getattr(arr, 'ndim', None)}")

    # Case 1: object stored in a 0-d numpy array
    if isinstance(arr, np.ndarray) and arr.ndim == 0:
        obj = arr.item()
        print(f"Top-level object type: {type(obj)}")

        if isinstance(obj, dict):
            print("dict keys:", list(obj.keys()))
            for k, v in obj.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', None)}")
                else:
                    s = str(v)
                    if len(s) > 200:
                        s = s[:197] + "..."
                    print(f"  {k}: type={type(v).__name__}, value={s}")

        else:
            s = str(obj)
            if len(s) > 500:
                s = s[:497] + "..."
            print("object repr:", s)

    # Case 2: array-like
    else:
        print("Array shape:", arr.shape)
        print("Array dtype:", arr.dtype)
        sample_preview = arr[:5] if arr.size > 5 else arr
        print("Sample values:", sample_preview)


if __name__ == "__main__":
    main()
