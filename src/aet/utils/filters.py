from __future__ import annotations


def should_skip_flip(flip: bool, require_label_preservation: bool) -> bool:
    return require_label_preservation and flip
