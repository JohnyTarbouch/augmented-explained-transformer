from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from datasets import load_dataset

from aet.data.augment import augment_text


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sentence", "label", "source"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_split(
    split_ds,
    *,
    replace_prob: float,
    seed: int,
    max_samples: int | None,
    augment_fraction: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows_orig: list[dict[str, object]] = []
    rows_aug: list[dict[str, object]] = []

    total = len(split_ds)
    limit = total if max_samples is None else min(max_samples, total)
    rng = random.Random(seed)
    indices = list(range(limit))
    rng.shuffle(indices)
    aug_count = int(limit * augment_fraction)
    aug_indices = set(indices[:aug_count])

    for idx in range(limit):
        row = split_ds[idx]
        text = row["sentence"]
        label = row["label"] if "label" in row else None
        rows_orig.append({"sentence": text, "label": label, "source": "original"})
        if idx in aug_indices:
            aug_text = augment_text(text, replace_prob=replace_prob, seed=seed + idx)
            rows_aug.append({"sentence": aug_text, "label": label, "source": "augmented"})

    return rows_orig, rows_aug


def main() -> None:
    parser = argparse.ArgumentParser(description="Create augmented SST-2 CSVs")
    parser.add_argument("--cache-dir", default="data/raw/hf_cache")
    parser.add_argument("--out-dir", default="data/interim/sst2_augmented")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="train",
    )
    parser.add_argument("--replace-prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--augment-fraction",
        type=float,
        default=1.0,
        help="Fraction of rows to augment (e.g., 0.1 for 10%).",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Write combined original+augmented CSV.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.augment_fraction <= 1.0:
        raise ValueError("--augment-fraction must be between 0.0 and 1.0")

    dataset = load_dataset("glue", "sst2", cache_dir=args.cache_dir)
    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    out_dir = Path(args.out_dir)
    for split in splits:
        split_ds = dataset[split]
        rows_orig, rows_aug = build_split(
            split_ds,
            replace_prob=args.replace_prob,
            seed=args.seed,
            max_samples=args.max_samples,
            augment_fraction=args.augment_fraction,
        )

        write_csv(out_dir / f"{split}_original.csv", rows_orig)
        write_csv(out_dir / f"{split}_augmented.csv", rows_aug)
        if args.combined:
            write_csv(out_dir / f"{split}_combined.csv", rows_orig + rows_aug)


if __name__ == "__main__":
    main()
