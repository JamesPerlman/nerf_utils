import os

from argparse import ArgumentParser
from pathlib import Path

# pyngp
import load_ngp
import pyngp as ngp # noqa

# local imports
from project import get_project_snapshot_name, get_project_snapshot_path

NGP_DIR = Path(os.environ['NGP_DIR'])
NETWORK_CONFIG_PATH = NGP_DIR / "configs" / "nerf" / "base.json"

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--transforms", type=str, required=True, help="NeRF-style transforms.json used to load a scene.")
    parser.add_argument("--steps", nargs="+", type=int, required=True, help="Number of steps to train (can be a list).")
    parser.add_argument("--load", type=str, required=False, help="Initial snapshot to load.")
    parser.add_argument("--save", type=str, required=False, help="Either a path to the desired output file, or a folder (if training multiple levels).  Will be auto-generated if not given.")

    args = parser.parse_args()

    return args

def get_snapshot_path(args: dict, n_steps: int) -> Path:
    # no args.save, use generated path
    if args.save is None:
        project_path = Path(args.transforms).parent
        return get_project_snapshot_path(project_path, n_steps)

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
    
    

def is_step_complete(args: dict, n_steps: int) -> bool:
    snapshot_path = get_snapshot_path(args, n_steps)
    return snapshot_path.exists()

# prepare training snapshots
def train(args: dict):
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.load_training_data(str(Path(args.transforms).absolute()))
    testbed.shall_train = True
    testbed.reload_network_from_file(str(NETWORK_CONFIG_PATH.absolute()))

    # eliminate duplicates
    training_steps = [*set(args.steps)]
    print(training_steps)

    # sort ascending
    training_steps = sorted(training_steps)

    # make sure no item is <= 0
    if min(training_steps) <= 0:
        print("n_steps must all be greater than zero.")
        return
    
    # if some steps have already been trained, we can skip those
    finished_steps = [step for step in training_steps if is_step_complete(args, step)]
    training_steps = [step for step in training_steps if not is_step_complete(args, step)]

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
            snapshot_path = get_snapshot_path(args, last_finished_step)
            testbed.load_snapshot(str(snapshot_path.absolute()))
            
            # TODO: double-check that this snapshot is valid and that testbed.training_step == min_step
            # need to also determine what to do if there's a mismatch - we may need to delete the snapshot
        is_testbed_ready = True

    # start up training loops
    while testbed.frame():
        current_step = testbed.training_step
        
        # save checkpoint
        if current_step in training_steps:
            snapshot_path = get_snapshot_path(args, current_step)
            
            if snapshot_path.exists():
                continue

            print(f"Saving snapshot for step {current_step}: {snapshot_path}")
            
            if not snapshot_path.parent.exists():
                snapshot_path.parent.mkdir()
            
            testbed.save_snapshot(str(snapshot_path.absolute()), False)

        # break when training is done
        if testbed.training_step >= max_step:
            break


if __name__ == "__main__":
    args = parse_args()

    train(args)
