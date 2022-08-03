from pathlib import Path


def get_project_snapshot_name(n_steps: int) -> str:
    return f"snapshot-{n_steps}.msgpack"

def get_project_snapshot_path(project_root: Path, n_steps) -> Path:
    return project_root / "training" / get_project_snapshot_name(n_steps)
    