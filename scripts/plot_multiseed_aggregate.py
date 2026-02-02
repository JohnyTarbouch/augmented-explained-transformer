"""Plot aggregate curves from multiseed summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_summary(path: Path) -> dict:
    """Load the multiseed summary"""
    # Multiseed summary JSON is produced by the multiseed aggregation script.
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_sanity(summary: dict, out_path: Path) -> bool:
    """Plot sanity randomization curves with mean/std"""
    # Looks for aggregated sanity output under:
    # summary["files"]["sanity_ig_randomization_summary.json"]["levels"]
    data = summary.get("files", {}).get("sanity_ig_randomization_summary.json")
    if not data:
        return False
    levels = data.get("levels", [])
    if not levels:
        return False

    # matplotlib is optional dependency; raise clearly if missing.
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    # X-axis is the randomization level ordering (kept stable by the aggregator).
    names = [str(level.get("level_name")) for level in levels]
    x = list(range(len(names)))

    # Each entry stores mean/std across seeds; we plot a line with a std band.
    metrics = [
        ("mean_kendall_tau", "Kendall Tau", "#2b6cb0"),
        ("mean_top_k_overlap", "Top-k Overlap", "#c05621"),
        ("mean_cosine_similarity", "Cosine Similarity", "#2f855a"),
    ]

    plt.figure(figsize=(8.5, 4.5))
    for key, label, color in metrics:
        vals = []
        stds = []
        for level in levels:
            # Aggregator may store either a dict {"mean":..., "std":...} or a raw scalar.
            stats = level.get(key) or {}
            vals.append(stats.get("mean", 0.0) if isinstance(stats, dict) else stats)
            stds.append(stats.get("std", 0.0) if isinstance(stats, dict) else 0.0)

        # Mean curve
        plt.plot(x, vals, marker="o", color=color, label=label)

        # Std band (mean ± std)
        plt.fill_between(
            x,
            [v - s for v, s in zip(vals, stds)],
            [v + s for v, s in zip(vals, stds)],
            color=color,
            alpha=0.15,
        )

    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Randomization level")
    plt.ylabel("Similarity to trained IG")
    plt.title("IG sanity check (mean ± std across seeds)")
    plt.legend(loc="upper right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _extract_curve(entry: dict, key: str) -> tuple[list[float], list[float]] | None:
    """Extract mean/std curve data from aggregate summary entries"""
    # Faithfulness aggregator may return curves as:
    # - dict with {"mean": [...], "std": [...]}  (preferred)
    # - list [...], interpreted as mean with std=0
    # - dict with {"values": [...]}, if lengths mismatched during aggregation
    stats = entry.get(key)
    if isinstance(stats, dict) and "mean" in stats:
        mean = stats.get("mean", [])
        std = stats.get("std", [0.0 for _ in mean])
        return mean, std
    if isinstance(stats, list):
        return stats, [0.0 for _ in stats]
    return None


def _plot_faithfulness(summary: dict, out_path: Path) -> bool:
    """Plot faithfulness AOPC curves with mean/std bands"""
    # Looks for aggregated faithfulness output under:
    # summary["files"]["faithfulness_aopc_summary.json"]
    data = summary.get("files", {}).get("faithfulness_aopc_summary.json")
    if not data:
        return False

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    # Fractions define the x-axis for AOPC curves.
    fractions = data.get("fractions")
    # Aggregated summaries may store fractions as {"values": ...} or {"mean": ...}
    # depending on how they were combined.
    if isinstance(fractions, dict):
        fractions = fractions.get("values") or fractions.get("mean")
    if not fractions:
        return False
    x = list(fractions)

    # Pull curves (mean/std) for IG and random baselines.
    comp = _extract_curve(data, "mean_comprehensiveness_curve")
    suff = _extract_curve(data, "mean_sufficiency_curve")
    rand_comp = _extract_curve(data, "mean_random_comprehensiveness_curve")
    rand_suff = _extract_curve(data, "mean_random_sufficiency_curve")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    if len(axes) != 2:
        axes = [axes]

    def plot_curve(ax, curve, rand_curve, title):
        # Plot IG curve with band; optionally overlay random baseline.
        if curve is None:
            return
        mean, std = curve
        ax.plot(x, mean, color="#2b6cb0", label="IG")
        ax.fill_between(
            x,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            color="#2b6cb0",
            alpha=0.15,
        )
        if rand_curve is not None:
            rmean, rstd = rand_curve
            ax.plot(x, rmean, color="#c05621", linestyle="--", label="random")
            ax.fill_between(
                x,
                [m - s for m, s in zip(rmean, rstd)],
                [m + s for m, s in zip(rmean, rstd)],
                color="#c05621",
                alpha=0.12,
            )
        ax.set_title(title)
        ax.set_xlabel("Fraction")
        ax.set_ylabel("Prob drop")
        ax.legend(fontsize=8)

    plot_curve(axes[0], comp, rand_comp, "Comprehensiveness (mean ± std)")
    plot_curve(axes[1], suff, rand_suff, "Sufficiency (mean ± std)")
    fig.suptitle("Faithfulness AOPC curves (mean ± std across seeds)")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main() -> None:
    """CLI entrypoint for aggregate plotting"""
    parser = argparse.ArgumentParser(description="Plot multiseed aggregate curves.")
    parser.add_argument(
        "--summary",
        default="reports/metrics/multiseed/baseline/multiseed_summary.json",
        help="Path to multiseed_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/figures/compare",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--prefix",
        default="baseline",
        help="Prefix for output filenames.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)
    summary = _load_summary(summary_path)

    # Output files are named by prefix so you can generate separate plots for baseline/augmented.
    sanity_out = out_dir / f"{args.prefix}_sanity_aggregate.png"
    faith_out = out_dir / f"{args.prefix}_faithfulness_aggregate.png"

    made_any = False
    if _plot_sanity(summary, sanity_out):
        print(f"Saved: {sanity_out}")
        made_any = True
    if _plot_faithfulness(summary, faith_out):
        print(f"Saved: {faith_out}")
        made_any = True

    if not made_any:
        print("No aggregate plots created (missing summaries).")


if __name__ == "__main__":
    main()
