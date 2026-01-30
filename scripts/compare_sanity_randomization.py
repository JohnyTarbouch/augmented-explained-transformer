"""
Overlay sanity randomization curves (baseline vs augmented).
Generates a plot comparing the IG sanity randomization metrics between
the baseline and augmented models across different randomization levels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> list[dict[str, object]]:
    """Load sanity summary JSON (list of per-level rows)"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def align_levels(base: list[dict[str, object]], aug: list[dict[str, object]]) -> list[str]:
    """Align level names across baseline/augmented summaries"""
    base_levels = [str(row.get("level_name")) for row in base]
    aug_levels = [str(row.get("level_name")) for row in aug]
    levels = []
    for name in base_levels + aug_levels:
        if name not in levels:
            levels.append(name)
    return levels


def values_for(levels: list[str], rows: list[dict[str, object]], key: str) -> list[float | None]:
    """Extract a metric across level order, preserving missing values"""
    mapping = {str(row.get("level_name")): row.get(key) for row in rows}
    vals: list[float | None] = []
    for level in levels:
        value = mapping.get(level)
        vals.append(float(value) if value is not None else None)
    return vals


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

    metrics = [
        ("mean_kendall_tau", "Kendall Tau", "#2b6cb0"),
        ("mean_top_k_overlap", "Top-k Overlap", "#c05621"),
        ("mean_cosine_similarity", "Cosine Similarity", "#2f855a"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.5))
    x = list(range(len(levels)))
    for metric_key, label, color in metrics:
        base_vals = values_for(levels, base, metric_key)
        aug_vals = values_for(levels, aug, metric_key)
        ax.plot(
            x,
            base_vals,
            marker="o",
            label=f"baseline {label}",
            color=color,
            linestyle="-",
        )
        ax.plot(
            x,
            aug_vals,
            marker="o",
            label=f"augmented {label}",
            color=color,
            linestyle="--",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Randomization level")
    ax.set_ylabel("Similarity to trained IG")
    ax.set_title("IG sanity check: baseline vs augmented")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
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
        "--out",
        default="reports/figures/compare/sanity_ig_randomization_overlay.png",
        help="Output plot path.",
    )
    args = parser.parse_args()

    base_path = Path(args.baseline)
    aug_path = Path(args.augmented)
    out_path = Path(args.out)

    if not base_path.exists():
        raise FileNotFoundError(f"Baseline summary not found: {base_path}")
    if not aug_path.exists():
        raise FileNotFoundError(f"Augmented summary not found: {aug_path}")

    base = load_summary(base_path)
    aug = load_summary(aug_path)
    levels = align_levels(base, aug)

    plot_overlay(levels, base, aug, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
