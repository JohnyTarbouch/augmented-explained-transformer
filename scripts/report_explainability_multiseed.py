from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


METHODS = {
    "ig": "consistency_baseline.csv",
    "lime": "lime_consistency.csv",
    "attention": "attention_consistency.csv",
}

METRICS = ["kendall_tau", "top_k_overlap", "cosine_similarity"]


def _read_metric_values(path: Path, metric: str) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if metric in row and row[metric] != "":
                values.append(float(row[metric]))
    return values


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {"mean": mean, "std": std}


def _load_seed_values(
    metrics_dir: Path,
    run_id: str,
    filename: str,
    metric: str,
) -> list[float]:
    path = metrics_dir / run_id / filename
    if not path.exists():
        return []
    return _read_metric_values(path, metric)


def _plot_boxplot(
    baseline_vals: list[float],
    augmented_vals: list[float],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.boxplot([baseline_vals, augmented_vals], labels=["baseline", "augmented"], showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_hist_overlay(
    baseline_vals: list[float],
    augmented_vals: list[float],
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(baseline_vals, bins=20, alpha=0.6, label="baseline", color="#2b6cb0")
    plt.hist(augmented_vals, bins=20, alpha=0.6, label="augmented", color="#c05621")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _aggregate_seed_means(seed_values: list[list[float]]) -> dict[str, float]:
    means = [np.mean(vals) for vals in seed_values if vals]
    if not means:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(means)), "std": float(np.std(means))}


def _plot_mean_std_overlay(
    baseline_seed_vals: list[list[float]],
    augmented_seed_vals: list[list[float]],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    base_means = [np.mean(vals) for vals in baseline_seed_vals if vals]
    aug_means = [np.mean(vals) for vals in augmented_seed_vals if vals]
    if not base_means or not aug_means:
        return

    base_mean = float(np.mean(base_means))
    base_std = float(np.std(base_means))
    aug_mean = float(np.mean(aug_means))
    aug_std = float(np.std(aug_means))

    labels = ["baseline", "augmented"]
    x = np.arange(len(labels))
    means = [base_mean, aug_mean]
    stds = [base_std, aug_std]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 4))
    plt.bar(x, means, yerr=stds, capsize=6, color=["#2b6cb0", "#c05621"], alpha=0.85)
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _iter_run_ids(prefix: str, seeds: Iterable[int]) -> list[str]:
    return [f"{prefix}_s{seed}" for seed in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot baseline vs augmented explainability across seeds."
    )
    parser.add_argument("--baseline-prefix", default="baseline")
    parser.add_argument("--augmented-prefix", default="augmented")
    parser.add_argument("--seeds", default="13,42,1337")
    parser.add_argument("--metrics-dir", default="reports/metrics")
    parser.add_argument("--figures-dir", default="reports/figures/compare")
    parser.add_argument(
        "--out",
        default="reports/figures/compare/compare_summary_multiseed.json",
        help="Output summary JSON.",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    out_path = Path(args.out)

    baseline_runs = _iter_run_ids(args.baseline_prefix, seeds)
    augmented_runs = _iter_run_ids(args.augmented_prefix, seeds)

    summary: dict[str, object] = {
        "baseline_prefix": args.baseline_prefix,
        "augmented_prefix": args.augmented_prefix,
        "seeds": seeds,
        "methods": {},
    }

    for method, filename in METHODS.items():
        summary["methods"][method] = {}
        for metric in METRICS:
            baseline_seed_vals: list[list[float]] = []
            augmented_seed_vals: list[list[float]] = []
            pooled_baseline: list[float] = []
            pooled_augmented: list[float] = []

            for run_id in baseline_runs:
                vals = _load_seed_values(metrics_dir, run_id, filename, metric)
                if vals:
                    baseline_seed_vals.append(vals)
                    pooled_baseline.extend(vals)

            for run_id in augmented_runs:
                vals = _load_seed_values(metrics_dir, run_id, filename, metric)
                if vals:
                    augmented_seed_vals.append(vals)
                    pooled_augmented.extend(vals)

            if not pooled_baseline or not pooled_augmented:
                continue

            metric_summary = {
                "pooled_baseline": _mean_std(pooled_baseline),
                "pooled_augmented": _mean_std(pooled_augmented),
                "seed_mean_baseline": _aggregate_seed_means(baseline_seed_vals),
                "seed_mean_augmented": _aggregate_seed_means(augmented_seed_vals),
                "num_seeds_baseline": len(baseline_seed_vals),
                "num_seeds_augmented": len(augmented_seed_vals),
            }
            summary["methods"][method][metric] = metric_summary

            _plot_boxplot(
                pooled_baseline,
                pooled_augmented,
                title=f"{method.upper()} {metric} (pooled across seeds)",
                ylabel=metric,
                path=figures_dir / f"{method}_{metric}_boxplot_multiseed.png",
            )
            _plot_hist_overlay(
                pooled_baseline,
                pooled_augmented,
                title=f"{method.upper()} {metric} distribution (pooled)",
                xlabel=metric,
                path=figures_dir / f"{method}_{metric}_hist_multiseed.png",
            )
            _plot_mean_std_overlay(
                baseline_seed_vals,
                augmented_seed_vals,
                title=f"{method.upper()} {metric} (mean ± std across seeds)",
                ylabel=metric,
                path=figures_dir / f"{method}_{metric}_meanstd_multiseed.png",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved multiseed compare summary to {out_path}")


if __name__ == "__main__":
    main()
