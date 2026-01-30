'''
Utility functions for resolving model paths and directories.
'''
from __future__ import annotations

from pathlib import Path

_MODEL_DIR_MARKERS = (
    "config.json",
    "adapter_config.json",
    "pytorch_model.bin",
    "model.safetensors",
)


def with_run_id(base_dir: str | Path, run_id: str | None) -> Path:
    # Append run_id to base_dir for each seed
    base = Path(base_dir)
    return base / run_id if run_id else base


def _looks_like_model_dir(path: Path) -> bool:
    # Check for presence of model marker files
    if not path.exists() or not path.is_dir():
        return False
    return any((path / marker).exists() for marker in _MODEL_DIR_MARKERS)


def _candidate_model_dirs(
    base: str | Path | None,
    run_id: str | None,
    seed: int | None,
) -> list[Path]:
    '''
    Generate candidate model directory paths based on base, run_id, and seed.
    1. base
    2. base_run_id
    3. base_run_id_s{seed}
    4. base_s{seed}
    5. (if none of the above exist, return empty list)
    '''
    if not base:
        return []
    base_path = Path(base)
    candidates = [base_path]
    if run_id:
        candidates.append(Path(f"{base_path}_{run_id}"))
        if seed is not None and not str(run_id).endswith(f"_s{seed}"):
            candidates.append(Path(f"{base_path}_{run_id}_s{seed}"))
    elif seed is not None:
        candidates.append(Path(f"{base_path}_s{seed}"))
    return candidates


def resolve_model_id(
    *,
    model_path: str | Path | None,
    training_output_dir: str | Path | None,
    model_name: str | None,
    run_id: str | None,
    seed: int | None,
) -> str:
    '''
    Resolve the model identifier or path based on provided parameters.
    Priority:
    1. If model_path is provided and looks like a model dir, return it.
    2. Search candidate dirs based on training_output_dir, run_id, seed.
    3. Fallback to model_name or default to "distilbert-base-uncased".
    '''
    if model_path:
        for candidate in _candidate_model_dirs(model_path, run_id, seed):
            if _looks_like_model_dir(candidate):
                return str(candidate)
        return str(model_path)

    for candidate in _candidate_model_dirs(training_output_dir, run_id, seed):
        if _looks_like_model_dir(candidate):
            return str(candidate)

    return model_name or "distilbert-base-uncased"
