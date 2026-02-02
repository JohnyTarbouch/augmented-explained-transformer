"""
Analyse and understand SST-2 dataset statistics 
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter
from pathlib import Path

try:
    from aet.data.datasets import load_sst2
except ImportError:
    # Allow running this script from outside an installed package context by
    # adding the repo's src/ directory to sys.path.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from aet.data.datasets import load_sst2


# SST-2 convention: 0 = negative, 1 = positive.
LABEL_NAMES = {0: "negative", 1: "positive"}


def ascii_histogram(values: list[int], bins: int = 10, width: int = 40) -> str:
    """Build a simple ASCII histogram for quick console inspection"""
    # Useful when matplotlib isn't installed or when you just want quick CLI output.
    if not values:
        return ""
    v_min = min(values)
    v_max = max(values)

    # Degenerate case: all values identical.
    if v_min == v_max:
        return f"{v_min:>4} | {'#' * width} ({len(values)})"

    # Build equally spaced bin edges from [min, max].
    step = (v_max - v_min) / float(bins)
    edges = [v_min + i * step for i in range(bins + 1)]

    # Count how many values fall into each bin.
    counts = [0] * bins
    for v in values:
        idx = int((v - v_min) / step)
        if idx >= bins:
            idx = bins - 1  # include max value in final bin
        counts[idx] += 1

    # Normalize bars relative to the largest bin count.
    max_count = max(counts) or 1
    lines = []
    for i in range(bins):
        left = int(edges[i])
        right = int(edges[i + 1])
        bar_len = int(counts[i] / max_count * width)
        lines.append(f"{left:>3}-{right:<3} | {'#' * bar_len} ({counts[i]})")
    return "\n".join(lines)


def summarize_split(name: str, split, bins: int, sample_count: int, seed: int) -> None:
    """Print summary stats and sample sentences for one split"""
    # Extract sentences and compute simple word-count lengths.
    sentences = split["sentence"]
    lengths = [len(text.split()) for text in sentences]

    # SST-2 test split has no labels in some versions; handle gracefully.
    labels = split["label"] if "label" in split.column_names else None

    print(f"\n== {name} ==")
    print(f"rows: {len(sentences)}")

    # Length statistics + ASCII histogram (quick view of distribution shape).
    if lengths:
        print(
            "lengths (words) -> "
            f"min: {min(lengths)}, max: {max(lengths)}, "
            f"mean: {statistics.mean(lengths):.2f}, "
            f"median: {statistics.median(lengths):.2f}"
        )
        print("length histogram:")
        print(ascii_histogram(lengths, bins=bins))

    # Label distribution (counts and percentages).
    if labels is not None:
        counts = Counter(labels)
        total = sum(counts.values()) or 1
        parts = []
        for label in sorted(counts):
            name_label = LABEL_NAMES.get(label, str(label))
            frac = counts[label] / total * 100.0
            parts.append(f"{name_label}: {counts[label]} ({frac:.1f}%)")
        print("label distribution: " + ", ".join(parts))
    else:
        print("label distribution: n/a (no labels in this split)")

    # Optional: print a few random examples for qualitative inspection.
    if sample_count > 0:
        random.seed(seed)
        indices = random.sample(range(len(sentences)), k=min(sample_count, len(sentences)))
        print("samples:")
        for idx in indices:
            # Keep each sample on one line and cap length for readability.
            text = sentences[idx].replace("\n", " ").strip()
            text = text[:200] + ("..." if len(text) > 200 else "")
            if labels is None:
                print(f"- {text}")
            else:
                label_name = LABEL_NAMES.get(labels[idx], str(labels[idx]))
                print(f"- [{label_name}] {text}")


def save_length_plot(all_lengths: list[int], out_path: Path, bins: int) -> bool:
    """Save a histogram plot if matplotlib is available"""
    # Optional visualization for reports.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not all_lengths:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hist(all_lengths, bins=bins, color="#2b6cb0", edgecolor="#1a202c")
    plt.title("SST-2 Sentence Lengths")
    plt.xlabel("Words per sentence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def main() -> None:
    """CLI entrypoint for dataset inspection"""
    parser = argparse.ArgumentParser(description="Inspect and visualize SST-2 data")
    parser.add_argument("--cache-dir", default="data/raw/hf_cache")
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="all",
    )
    parser.add_argument(
        "--save-plot",
        default="reports/figures/sst2_length_hist.png",
        help="Save a PNG histogram if matplotlib is installed.",
    )
    args = parser.parse_args()

    # Load dataset from HF cache dir (your project wrapper decides how it's cached).
    dataset = load_sst2(cache_dir=args.cache_dir)

    # Which splits to inspect in one run.
    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    # Collect lengths across all inspected splits for an overall histogram plot.
    all_lengths: list[int] = []
    for name in splits:
        split = dataset[name]
        summarize_split(name, split, bins=args.bins, sample_count=args.samples, seed=args.seed)
        all_lengths.extend([len(text.split()) for text in split["sentence"]])

    # Optional plot output to PNG.
    if args.save_plot:
        out_path = Path(args.save_plot)
        if save_length_plot(all_lengths, out_path, bins=args.bins):
            print(f"\nSaved histogram to {out_path}")
        else:
            print("\nPlot not saved (matplotlib not installed).")


if __name__ == "__main__":
    main()
