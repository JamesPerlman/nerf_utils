import os
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print(f"Usage: nerf preview <project/directory/or/transforms.json>")
    exit()

NGP_DIR = os.environ['NGP_DIR']
scene_path = Path(os.getcwd()) / sys.argv[1]
os.system(f"python \"{NGP_DIR}/scripts/run.py\" --gui --mode nerf --scene \"{str(scene_path.as_posix())}\"")
