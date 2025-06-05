#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Optional


def get_workspace_path(subdirectory: Optional[str] = None) -> Path:
    workspace_path = Path(os.environ.get('SHRUBBERY_WORKSPACE', './workspace'))
    if subdirectory is not None:
        workspace_path = workspace_path / subdirectory
    workspace_path.mkdir(parents=True, exist_ok=True)
    return workspace_path
