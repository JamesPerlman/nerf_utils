from argparse import ArgumentParser
import os
import sys
from pathlib import Path

NGP_DIR = os.environ['NGP_DIR']

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--transforms", type=str, help="NeRF-style transforms.json used to load a scene.")
    parser.add_argument("--snapshot", type=str, help="Snapshot to load.")
    args = parser.parse_args()

    if args.transforms:
        scene_path = Path(args.transforms)
        os.system(f"python \"{NGP_DIR}/scripts/run.py\" --gui --mode nerf --scene \"{str(scene_path.as_posix())}\"")
    elif args.snapshot:
        snapshot_path = Path(args.snapshot)
        os.system(f"python \"{NGP_DIR}/scripts/run.py\" --gui --mode nerf --load_snapshot \"{str(snapshot_path.as_posix())}\"")