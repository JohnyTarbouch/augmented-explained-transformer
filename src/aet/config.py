'''
Augmented-Explained-Transformer Configuration Loader
This module provides functionality to load YAML configuration files
from the configs/ directory.'''
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:

    '''
    Load a YAML configuration file from configs/ dir.
    '''    

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
