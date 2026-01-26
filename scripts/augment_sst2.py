from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")

import argparse
import csv
import os
import random
import time
from pathlib import Path

from datasets import load_dataset

from aet.data.augment import augment_text
from aet.data.backtranslation import back_translate_batch, back_translate_text
from aet.utils.device import resolve_device


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


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
    method: str,
    backtranslation_cfg: dict[str, object] | None = None,
    progress_every: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    rows_orig: list[dict[str, object]] = []
    rows_aug: list[dict[str, object]] = []

    total = len(split_ds)
    limit = total if max_samples is None else min(max_samples, total)
    rng = random.Random(seed)
    indices = list(range(limit))
    rng.shuffle(indices)
    aug_count = int(limit * augment_fraction)
    aug_indices = indices[:aug_count]
    aug_index_set = set(aug_indices)

    stats = {
        "requested": 0,
        "augmented": 0,
        "filtered": 0,
        "failed": 0,
    }

    aug_rows: dict[int, dict[str, object]] = {}
    if method == "backtranslation" and aug_indices:
        aug_texts: list[str] = []
        aug_meta: list[tuple[int, str, object]] = []
        for idx in aug_indices:
            row = split_ds[idx]
            text = row["sentence"]
            label = row["label"] if "label" in row else None
            aug_texts.append(text)
            aug_meta.append((idx, text, label))

        stats["requested"] = len(aug_texts)
        cfg = backtranslation_cfg or {}
        min_ratio = float(cfg.get("min_length_ratio", 0.7))
        max_ratio = float(cfg.get("max_length_ratio", 1.3))
        cfg_batch = {
            "src_lang": cfg.get("src_lang", "en"),
            "pivot_lang": cfg.get("pivot_lang", "de"),
            "device": cfg.get("device", "cpu"),
            "batch_size": cfg.get("batch_size", 8),
            "max_length": cfg.get("max_length", 128),
        }
        try:
            print(f"Back-translation: translating {len(aug_texts)} samples...")
            translated = back_translate_batch(
                aug_texts,
                **cfg_batch,
                progress_every=progress_every,
            )
        except Exception as exc:
            print(f"Back-translation failed, falling back to single-sample mode: {exc}")
            translated = []
            for text in aug_texts:
                result = back_translate_text(text, **cfg)
                translated.append(result.text)

        for (idx, text, label), aug_text in zip(aug_meta, translated):
            src_len = len(text.split())
            tgt_len = len(aug_text.split())
            ratio = (tgt_len / float(src_len)) if src_len else 0.0
            if min_ratio > 0 and max_ratio > 0 and not (min_ratio <= ratio <= max_ratio):
                stats["filtered"] += 1
                continue
            if not aug_text or aug_text == text:
                stats["failed"] += 1
                continue
            aug_rows[idx] = {"sentence": aug_text, "label": label, "source": "augmented"}
            stats["augmented"] += 1

    for idx in range(limit):
        row = split_ds[idx]
        text = row["sentence"]
        label = row["label"] if "label" in row else None
        rows_orig.append({"sentence": text, "label": label, "source": "original"})
        if idx in aug_index_set:
            if method == "backtranslation":
                if idx in aug_rows:
                    rows_aug.append(aug_rows[idx])
                continue

            stats["requested"] += 1
            aug_text = augment_text(
                text,
                replace_prob=replace_prob,
                seed=seed + idx,
                method=method,
                backtranslation_cfg=backtranslation_cfg,
            )
            if aug_text == text:
                stats["failed"] += 1
                continue
            rows_aug.append({"sentence": aug_text, "label": label, "source": "augmented"})
            stats["augmented"] += 1

    return rows_orig, rows_aug, stats


def main() -> None:
    _load_dotenv(Path(".env"))
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
        "--method",
        choices=["wordnet", "backtranslation"],
        default="wordnet",
        help="Augmentation method to apply.",
    )
    parser.add_argument("--bt-src-lang", default="en")
    parser.add_argument("--bt-pivot-lang", default="de")
    parser.add_argument("--bt-device", default="auto")
    parser.add_argument("--bt-batch-size", type=int, default=8)
    parser.add_argument("--bt-max-length", type=int, default=128)
    parser.add_argument("--bt-progress-every", type=int, default=0)
    parser.add_argument("--min-length-ratio", type=float, default=0.7)
    parser.add_argument("--max-length-ratio", type=float, default=1.3)
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
        back_cfg = {
            "src_lang": args.bt_src_lang,
            "pivot_lang": args.bt_pivot_lang,
            "device": resolve_device(args.bt_device),
            "batch_size": args.bt_batch_size,
            "max_length": args.bt_max_length,
            "min_length_ratio": args.min_length_ratio,
            "max_length_ratio": args.max_length_ratio,
        }
        progress_every = args.bt_progress_every
        if progress_every <= 0 and args.max_samples is not None and args.max_samples <= 200:
            progress_every = 1
        rows_orig, rows_aug, stats = build_split(
            split_ds,
            replace_prob=args.replace_prob,
            seed=args.seed,
            max_samples=args.max_samples,
            augment_fraction=args.augment_fraction,
            method=args.method,
            backtranslation_cfg=back_cfg if args.method == "backtranslation" else None,
            progress_every=progress_every,
        )

        write_csv(out_dir / f"{split}_original.csv", rows_orig)
        write_csv(out_dir / f"{split}_augmented.csv", rows_aug)
        if args.combined:
            write_csv(out_dir / f"{split}_combined.csv", rows_orig + rows_aug)

        print(
            f"[{split}] requested={stats['requested']} augmented={stats['augmented']} "
            f"filtered={stats['filtered']} failed={stats['failed']}"
        )


if __name__ == "__main__":
    main()
