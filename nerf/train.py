import json
import os

from argparse import ArgumentParser
from pathlib import Path
import time
from tqdm import tqdm

# pyngp
import load_ngp
import pyngp as ngp # noqa

# local imports
from multi_gpu import split_across_gpus
from project import get_project_snapshot_name, get_project_snapshot_path

NGP_DIR = Path(os.environ['NGP_DIR'])
NETWORK_CONFIG_PATH = NGP_DIR / "configs" / "nerf" / "base.json"
NGP_SCALE = 0.33

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--transforms", type=str, required=True, help="NeRF-style transforms.json used to load a scene.")
    parser.add_argument("--steps", nargs="+", type=int, required=True, help="Number of steps to train (can be a list).")
    parser.add_argument("--load", type=str, required=False, help="Initial snapshot to load.")
    parser.add_argument("--save", type=str, required=False, help="Either a path to the desired output file, or a folder (if training multiple levels).  Will be auto-generated if not given.")
    parser.add_argument("--batch_size", type=int, default=2<<19, help="NeRF training batch size.")
    
    parser.add_argument("--gpus", type=str, default="all", help="Which GPUs to use.  Can be a list of indices, or 'all'.")
    parser.add_argument("--batch", type=str, default=None, help="For multi-GPU rendering. It is not recommended to use this feature directly.")

    args = parser.parse_args()

    return args

def get_snapshot_path(args: dict, transforms_path: Path, n_steps: int) -> Path:
    # no args.save, use generated path
    if args.save is None:
        return get_project_snapshot_path(transforms_path, n_steps)

    save_path = Path(args.save)
    snapshot_name = get_project_snapshot_name(n_steps)
    
    if save_path.is_file():
        # if save path already exists, then use a generated path inside the same project with the transforms.json
        if save_path.exists():
            project_path = Path(args.transforms).parent
            return get_project_snapshot_path(project_path, n_steps)
        else:
            # doesn't exist, we can use the args.save path
            return Path(args.save)
    
    # otherwise, save path must be a directory
    return save_path / snapshot_name
    
    

def is_step_complete(args: dict, transforms_path: Path, n_steps: int) -> bool:
    snapshot_path = get_snapshot_path(args, transforms_path, n_steps)
    return snapshot_path.exists()

def parse_transforms(transforms_path: Path) -> dict:
    with open(transforms_path, "r") as f:
        return json.load(f)

# prepare training snapshots
def train(args: dict, transforms_path: Path):
    print(f"Training steps for {transforms_path.name}")
    # transforms = parse_transforms(transforms_path)

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_training_data(str(transforms_path.absolute()))
    testbed.shall_train = True
    testbed.reload_network_from_file(str(NETWORK_CONFIG_PATH.absolute()))
    testbed.nerf.training.near_distance = 0.2 # min(frame["near"] for frame in transforms["frames"]) * NGP_SCALE
    testbed.nerf.training.optimize_exposure = True
    testbed.training_batch_size = args.batch_size
    testbed.nerf.training.optimize_extrinsics = True
    # testbed.nerf.training.optimize_extra_dims = True
    testbed.nerf.training.optimize_focal_length = True
    testbed.nerf.training.optimize_distortion = True

    # eliminate duplicates
    training_steps = [*set(args.steps)]
    print(f"Training up to these levels: {training_steps}")
    print(f"Using near value {testbed.nerf.training.near_distance}")

    # sort ascending
    training_steps = sorted(training_steps)

    # make sure no item is <= 0
    if min(training_steps) <= 0:
        print("n_steps must all be greater than zero.")
        return
    
    # if some steps have already been trained, we can skip those
    finished_steps = [step for step in training_steps if is_step_complete(args, transforms_path, step)]
    training_steps = [step for step in training_steps if not is_step_complete(args, transforms_path, step)]

    if len(training_steps) == 0:
        print("All training is already complete!")
        return

    min_step = min(training_steps)
    max_step = max(training_steps)
    
    # load a checkpoint, if the args dictate such
    is_testbed_ready = False
    if args.load != None:
        snapshot_path = Path(args.load)
        print(f"Loading snapshot from {snapshot_path.absolute()}...")
        testbed.load_snapshot(str(snapshot_path.absolute()))

        if testbed.training_step < min_step:
            testbed.reset()
            print(f"Given snapshot is overtrained. (loaded step={testbed.training_step}, min_step={min_step})")
        else:
            is_testbed_ready = True
    
    # otherwise, try loading a checkpoint from min_step
    if not is_testbed_ready:
        if len(finished_steps) > 0:
            last_finished_step = finished_steps[-1]
            snapshot_path = get_snapshot_path(args, transforms_path, last_finished_step)
            testbed.load_snapshot(str(snapshot_path.absolute()))
            
            # TODO: double-check that this snapshot is valid and that testbed.training_step == min_step
            # need to also determine what to do if there's a mismatch - we may need to delete the snapshot
        is_testbed_ready = True

    # start up training loops
    tqdm_last_update = 0
    previous_step = testbed.training_step
    with tqdm(desc="Training", total=max(args.steps), unit="step") as t:
        while testbed.frame():
            current_step = testbed.training_step
            
            # update progress bar

            if current_step < previous_step or previous_step == 0:
                previous_step = 0
                t.reset()
            
            now = time.monotonic()
            if now - tqdm_last_update > 0.1:
                t.update(current_step - previous_step)
                t.set_postfix(loss=testbed.loss)
                previous_step = current_step
                tqdm_last_update = now
            
            # save checkpoint
            if current_step in training_steps:
                snapshot_path = get_snapshot_path(args, transforms_path, current_step)
                
                if snapshot_path.exists():
                    continue

                print(f"Saving snapshot for step {current_step}: {snapshot_path}")
                
                if not snapshot_path.parent.exists():
                    snapshot_path.parent.mkdir(exist_ok=True, parents=True)
                
                testbed.save_snapshot(str(snapshot_path.absolute()), False)

        
            # break when training is done
            if testbed.training_step >= max_step:
                break
            


if __name__ == "__main__":
    args = parse_args()

    transforms_path = Path(args.transforms)
    if transforms_path.is_dir():
        json_paths = []
        for path in transforms_path.iterdir():
            # filter out non-json files
            if path.suffix != ".json":
                continue
            
            # make sure the file is a valid transforms.json
            if "frames" not in parse_transforms(path):
                continue
                
            json_paths.append(path)
        
        # train on each transforms.json
        def train_fn(proc_idx, n_procs):
            for i in range(len(json_paths)):
                if i % n_procs == proc_idx:
                    print(f"TRAINING {i} {n_procs} {proc_idx}")
                    train(args, json_paths[i])
                    
        
        def done_fn():
            print("All training is complete!")

        split_across_gpus(args, len(json_paths), train_fn, done_fn)
        


    else:
        train(args, transforms_path)
