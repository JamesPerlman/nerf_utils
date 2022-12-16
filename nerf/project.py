from pathlib import Path

# TODO: create Project class

def get_project_snapshot_name(transforms_path: Path, n_steps: int) -> str:
    tname = transforms_path.stem.split('.')[0]
    return f"{tname}-step{n_steps}.msgpack"

def get_project_snapshot_path(transforms_path: Path, n_steps) -> Path:
    return transforms_path.parent / "ngp" / get_project_snapshot_name(transforms_path, n_steps)
    