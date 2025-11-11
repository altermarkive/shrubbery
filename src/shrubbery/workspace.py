import os
from pathlib import Path


def get_workspace_path(subdirectory: str | None = None) -> Path:
    workspace_path = Path(os.environ.get('SHRUBBERY_WORKSPACE', './workspace'))
    if subdirectory is not None:
        workspace_path = workspace_path / subdirectory
    workspace_path.mkdir(parents=True, exist_ok=True)
    return workspace_path
