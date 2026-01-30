from __future__ import annotations


def should_skip_flip(flip: bool, require_label_preservation: bool) -> bool:
    # Determine whether to skip flip augmentation (flip meaning label change)
    return require_label_preservation and flip
