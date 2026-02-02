"""
Analyze augmentation effects and label-flip proxies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"\w+")  # simple word-token matcher (alnum/underscore)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens"""
    # Lowercasing + regex tokenization keeps analysis lightweight and deterministic.
    return TOKEN_RE.findall(text.lower())


def _to_bool(value: object) -> bool:
    """Coerce string/bool values to bool"""
    # Accept common truthy spellings from CSVs (e.g., "1", "true", "yes").
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _mean_std(values: list[float]) -> dict[str, float]:
    """Compute mean/std for a list of values"""
    # Population std (divide by N), not sample std (N-1).
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": float(mean), "std": float(math.sqrt(var))}


def _js_divergence(p_counts: Counter[str], q_counts: Counter[str]) -> float:
    """Compute Jensen-Shannon divergence between token distributions"""
    # JSD is symmetric and bounded; uses log base 2 here.
    total_p = sum(p_counts.values())
    total_q = sum(q_counts.values())
    if total_p == 0 or total_q == 0:
        return 0.0
    vocab = set(p_counts) | set(q_counts)
    kl_p = 0.0
    kl_q = 0.0
    for token in vocab:
        p = p_counts[token] / total_p
        q = q_counts[token] / total_q
        m = 0.5 * (p + q)  # mixture distribution
        if p > 0:
            kl_p += p * math.log2(p / m)
        if q > 0:
            kl_q += q * math.log2(q / m)
    return 0.5 * (kl_p + kl_q)


def _change_ratio(orig_tokens: list[str], aug_tokens: list[str]) -> float:
    """Return fraction of tokens changed using sequence matching"""
    # Approximates "how much changed" via LCS-like matching from SequenceMatcher.
    if not orig_tokens:
        return 0.0
    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, orig_tokens, aug_tokens)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    changed = max(0, len(orig_tokens) - matched)
    return float(changed / len(orig_tokens))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV into a list of row dicts"""
    # Used for both SST-2 original/augmented CSVs and optional consistency CSV.
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_consistency_pairs(path: Path) -> list[dict[str, object]]:
    """Load text/aug_text/flip triples from a consistency CSV"""
    # Consistency CSV is expected to contain paired texts and an optional "flip" flag.
    rows = _load_csv_rows(path)
    pairs = []
    for row in rows:
        if "text" not in row or "aug_text" not in row:
            raise ValueError("Consistency CSV must include text and aug_text columns.")
        pairs.append(
            {
                "text": row["text"],
                "aug_text": row["aug_text"],
                "flip": _to_bool(row.get("flip", False)),  # prediction-flip proxy if provided
            }
        )
    return pairs


def _token_distribution(texts: Iterable[str]) -> Counter[str]:
    """Count tokens across a collection of text"""
    # Unigram counts aggregated over all texts.
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_tokenize(text))
    return counts


def _plot_hist(values_a: list[float], values_b: list[float], out_path: Path) -> bool:
    """Plot histograms for no-flip vs flip change ratios"""
    # Returns False if matplotlib isn't available or there is nothing to plot.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not values_a and not values_b:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if values_a:
        plt.hist(values_a, bins=20, alpha=0.6, label="no-flip", color="#2b6cb0")
    if values_b:
        plt.hist(values_b, bins=20, alpha=0.6, label="flip", color="#c05621")
    plt.title("Change ratio (original vs augmented)")
    plt.xlabel("Token change ratio")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _plot_top_tokens(diff: Counter[str], out_path: Path, title: str, top_k: int) -> bool:
    """Plot top-k token frequency delta."""
    # Used for visualizing normalized token-frequency shifts (and flip-case deltas).
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not diff:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = diff.most_common(top_k)
    tokens = [t for t, _ in items][::-1]  # reverse for nicer barh ordering
    values = [v for _, v in items][::-1]

    plt.figure(figsize=(8, max(4, 0.25 * len(tokens))))
    plt.barh(tokens, values, color="#2b6cb0")
    plt.title(title)
    plt.xlabel("Frequency delta")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _plot_js_divergence(value: float, out_path: Path) -> bool:
    """Plot a single-bar JS divergence summary"""
    # Simple single-value summary chart for the distribution shift statistic.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.bar(["JS divergence"], [value], color="#2b6cb0")
    plt.ylim(0.0, max(0.1, value * 1.2))  # keep plot readable even for tiny values
    plt.ylabel("Value")
    plt.title("Unigram JS divergence (orig vs augmented)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze augmentation effects and potential label noise.")
    parser.add_argument("--original-csv", default="data/interim/sst2_augmented/train_original.csv")
    parser.add_argument("--augmented-csv", default="data/interim/sst2_augmented/train_augmented.csv")
    parser.add_argument(
        "--consistency-csv",
        default=None,
        help="Optional consistency CSV with paired original/augmented texts.",
    )
    parser.add_argument(
        "--out-metrics",
        default="reports/metrics/compare/augmentation_distribution_summary.json",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures/compare",
    )
    parser.add_argument("--top-k", type=int, default=30)  # number of tokens to show in token-delta plots
    args = parser.parse_args()

    out_metrics = Path(args.out_metrics)
    figures_dir = Path(args.figures_dir)

    summary: dict[str, object] = {
        "note": "Flip rates use model predictions from consistency CSV (proxy for noise/difficulty).",
    }

    # Distribution shift on full original vs augmented CSVs.
    orig_rows = _load_csv_rows(Path(args.original_csv))
    aug_rows = _load_csv_rows(Path(args.augmented_csv))
    orig_texts = [row["sentence"] for row in orig_rows if row.get("sentence")]
    aug_texts = [row["sentence"] for row in aug_rows if row.get("sentence")]

    # Token-level distribution shift (unigram JSD).
    orig_counts = _token_distribution(orig_texts)
    aug_counts = _token_distribution(aug_texts)
    js = _js_divergence(orig_counts, aug_counts)

    # Length stats (tokenized length).
    orig_lengths = [len(_tokenize(text)) for text in orig_texts]
    aug_lengths = [len(_tokenize(text)) for text in aug_texts]

    summary["distribution_shift"] = {
        "original_rows": len(orig_rows),
        "augmented_rows": len(aug_rows),
        "js_divergence_unigram": js,
        "original_length": _mean_std(orig_lengths),
        "augmented_length": _mean_std(aug_lengths),
    }

    # Compute normalized frequency deltas: p_aug(token) - p_orig(token).
    total_orig = sum(orig_counts.values())
    total_aug = sum(aug_counts.values())
    diff_counts = Counter()
    for token in set(orig_counts) | set(aug_counts):
        if total_orig == 0 or total_aug == 0:
            continue
        diff = (aug_counts[token] / total_aug) - (orig_counts[token] / total_orig)
        if diff != 0:
            diff_counts[token] = diff

    increased = sorted(diff_counts.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]
    decreased = sorted(diff_counts.items(), key=lambda kv: kv[1])[: args.top_k]

    summary["top_tokens"] = {
        "increased": increased,
        "decreased": decreased,
    }

    # Save figures for token deltas + JSD.
    _plot_top_tokens(
        Counter({k: v for k, v in diff_counts.items() if v > 0}),
        figures_dir / "augmentation_top_tokens.png",
        "Tokens more frequent after augmentation (normalized)",
        args.top_k,
    )
    _plot_js_divergence(js, figures_dir / "augmentation_js_divergence.png")

    # Paired analysis (flip rate / difficulty proxy) if consistency CSV is provided.
    if args.consistency_csv:
        pairs = _load_consistency_pairs(Path(args.consistency_csv))
        change_ratios = []
        flip_change_ratios = []
        noflip_change_ratios = []
        flip_count = 0

        flip_token_diff = Counter()
        for pair in pairs:
            orig_tokens = _tokenize(pair["text"])
            aug_tokens = _tokenize(pair["aug_text"])
            ratio = _change_ratio(orig_tokens, aug_tokens)
            change_ratios.append(ratio)
            if pair["flip"]:
                # "Flip" here is a proxy signal (e.g., model prediction changed).
                flip_count += 1
                flip_change_ratios.append(ratio)
                # Tokens net-added in augmented vs original for flip cases.
                flip_token_diff.update((Counter(aug_tokens) - Counter(orig_tokens)))
            else:
                noflip_change_ratios.append(ratio)

        summary["paired_consistency_proxy"] = {
            "pairs": len(pairs),
            "flip_rate": float(flip_count / len(pairs)) if pairs else 0.0,
            "change_ratio": _mean_std(change_ratios),
            "change_ratio_flip": _mean_std(flip_change_ratios),
            "change_ratio_no_flip": _mean_std(noflip_change_ratios),
        }

        # Compare change-ratio distributions for flip vs no-flip.
        _plot_hist(
            noflip_change_ratios,
            flip_change_ratios,
            figures_dir / "flip_change_ratio_hist.png",
        )
        _plot_top_tokens(
            flip_token_diff,
            figures_dir / "flip_changed_tokens.png",
            "Tokens added in flip cases (augmented vs original)",
            args.top_k,
        )

    # Persist metrics JSON (and ensure output folder exists).
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {out_metrics}")


if __name__ == "__main__":
    main()  # CLI entrypoint
