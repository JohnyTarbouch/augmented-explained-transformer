"""
Overlay sanity randomization curves (baseline vs augmented).

Reads per-level IG sanity summaries (single-seed or multiseed) and produces one
overlay plot comparing baseline (solid) vs augmented (dashed) across levels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> list[dict[str, object]]:
    """Load sanity summary JSON (list of per-level rows)."""
    # Expected format: a JSON array of dicts, one per randomization level.
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def load_multiseed_summary(path: Path) -> list[dict[str, object]]:
    """Load multiseed_summary.json and extract sanity levels."""
    # Multiseed format nests the sanity summary under: data["files"][...]["levels"].
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files", {}) if isinstance(data, dict) else {}
    sanity = files.get("sanity_ig_randomization_summary.json", {})
    levels = sanity.get("levels", []) if isinstance(sanity, dict) else []
    if not levels:
        raise ValueError(f"No sanity levels found in {path}")
    return levels


def align_levels(base: list[dict[str, object]], aug: list[dict[str, object]]) -> list[str]:
    """Align level names across baseline/augmented summaries."""
    # Union of level names, preserving first-seen order for stable plotting.
    base_levels = [str(row.get("level_name")) for row in base]
    aug_levels = [str(row.get("level_name")) for row in aug]
    levels = []
    for name in base_levels + aug_levels:
        if name not in levels:
            levels.append(name)
    return levels


def values_for(
    levels: list[str],
    rows: list[dict[str, object]],
    key: str,
) -> tuple[list[float], list[float]]:
    """Extract mean/std arrays across level order (std=0 for single-seed)."""
    # Map each level_name -> metric value, where value can be float or {mean,std}.
    mapping = {str(row.get("level_name")): row.get(key) for row in rows}
    means: list[float] = []
    stds: list[float] = []
    for level in levels:
        value = mapping.get(level)
        if isinstance(value, dict):
            # Multiseed rows store aggregates.
            means.append(float(value.get("mean", 0.0)))
            stds.append(float(value.get("std", 0.0)))
        elif value is None:
            # Missing metric for this level; keep shape aligned for plotting.
            means.append(float("nan"))
            stds.append(0.0)
        else:
            # Single-seed rows store a raw scalar.
            means.append(float(value))
            stds.append(0.0)
    return means, stds


def plot_overlay(
    levels: list[str],
    base: list[dict[str, object]],
    aug: list[dict[str, object]],
    out_path: Path,
) -> None:
    """Plot baseline vs augmented metric curves over randomization levels"""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    # Metrics to overlay; colors chosen to keep series consistent across runs.
    metrics = [
        ("mean_kendall_tau", "Kendall Tau", "#2b6cb0"),
        ("mean_top_k_overlap", "Top-k Overlap", "#c05621"),
        ("mean_cosine_similarity", "Cosine Similarity", "#2f855a"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.5))
    x = list(range(len(levels)))  # Numeric x-axis positions for categorical level labels.

    for metric_key, label, color in metrics:
        base_vals, base_stds = values_for(levels, base, metric_key)
        aug_vals, aug_stds = values_for(levels, aug, metric_key)

        # Baseline curve (solid) + uncertainty band.
        ax.plot(
            x,
            base_vals,
            marker="o",
            label=f"baseline (solid) {label}",
            color=color,
            linestyle="-",
        )
        ax.fill_between(
            x,
            [v - s for v, s in zip(base_vals, base_stds)],
            [v + s for v, s in zip(base_vals, base_stds)],
            color=color,
            alpha=0.12,
        )

        # Augmented curve (dashed) + uncertainty band.
        ax.plot(
            x,
            aug_vals,
            marker="o",
            label=f"augmented (dashed) {label}",
            color=color,
            linestyle="--",
        )
        ax.fill_between(
            x,
            [v - s for v, s in zip(aug_vals, aug_stds)],
            [v + s for v, s in zip(aug_vals, aug_stds)],
            color=color,
            alpha=0.08,
        )

    # Cosmetics / labeling.
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Randomization level")
    ax.set_ylabel("Similarity to trained IG")
    ax.set_title("IG sanity check: baseline vs augmented")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # Save output.
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sanity randomization curves.")
    parser.add_argument(
        "--baseline",
        default="reports/metrics/baseline/sanity_ig_randomization_summary.json",
        help="Baseline summary JSON.",
    )
    parser.add_argument(
        "--augmented",
        default="reports/metrics/augmented/sanity_ig_randomization_summary.json",
        help="Augmented summary JSON.",
    )
    parser.add_argument(
        "--baseline-multiseed",
        default=None,
        help="Multiseed summary JSON for baseline.",
    )
    parser.add_argument(
        "--augmented-multiseed",
        default=None,
        help="Multiseed summary JSON for augmented.",
    )
    parser.add_argument(
        "--out",
        default="reports/figures/compare/sanity_ig_randomization_overlay.png",
        help="Output plot path.",
    )
    args = parser.parse_args()

    # Resolve input/output paths.
    base_path = Path(args.baseline)
    aug_path = Path(args.augmented)
    base_multi = Path(args.baseline_multiseed) if args.baseline_multiseed else None
    aug_multi = Path(args.augmented_multiseed) if args.augmented_multiseed else None
    out_path = Path(args.out)

    # Prefer multiseed inputs if both provided; otherwise fall back to single summaries.
    if base_multi and aug_multi:
        if not base_multi.exists():
            raise FileNotFoundError(f"Baseline multiseed summary not found: {base_multi}")
        if not aug_multi.exists():
            raise FileNotFoundError(f"Augmented multiseed summary not found: {aug_multi}")
        base = load_multiseed_summary(base_multi)
        aug = load_multiseed_summary(aug_multi)
    else:
        if not base_path.exists():
            raise FileNotFoundError(f"Baseline summary not found: {base_path}")
        if not aug_path.exists():
            raise FileNotFoundError(f"Augmented summary not found: {aug_path}")
        base = load_summary(base_path)
        aug = load_summary(aug_path)

    # Align x-axis level order across both runs.
    levels = align_levels(base, aug)

    plot_overlay(levels, base, aug, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
