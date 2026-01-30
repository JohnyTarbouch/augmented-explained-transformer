'''
This module provides functionality to resolve the computation device
'''
from __future__ import annotations

from typing import Literal

import torch


def resolve_device(device: str | None) -> Literal["cuda", "cpu"]:
    if device is None:
        device = "auto"
    device = device.lower()
    # Use device, GPU
    if device in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return "cuda"
    if device == "cpu":
        return "cpu"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    raise ValueError(f"Unknown device setting: {device}")
