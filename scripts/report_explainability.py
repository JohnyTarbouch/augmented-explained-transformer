"""Compare baseline vs augmented explainability outputs and generate plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from statistics import NormalDist

import torch


def _normalize_token(token: str) -> str:
    """Normalize tokens for counting/overlap."""
    return token.lower().strip(".,!?;:\"'()[]{}")


def _load_metric_csv(path: Path) -> dict[str, list[float]]:
    """Load metric columns from a CSV into lists."""
    metrics = {
        "kendall_tau": [],
        "top_k_overlap": [],
        "cosine_similarity": [],
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in metrics:
                if key in row and row[key] != "":
                    metrics[key].append(float(row[key]))
    return metrics


def _load_metric_rows(path: Path) -> dict[int, dict[str, float]]:
    """Load per-row metrics keyed by example id."""
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "id" not in row:
                continue
            row_id = int(row["id"])
            rows[row_id] = {}
            for key in ("kendall_tau", "top_k_overlap", "cosine_similarity"):
                if key in row and row[key] != "":
                    rows[row_id][key] = float(row[key])
    return rows


def _load_text_pairs(path: Path) -> dict[int, tuple[str, str]]:
    """Load (text, aug_text) pairs by id from a consistency CSV."""
    pairs: dict[int, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "id" not in row:
                continue
            if "text" not in row or "aug_text" not in row:
                continue
            pairs[int(row["id"])] = (row["text"], row["aug_text"])
    return pairs


def _tokenize_simple(text: str) -> list[str]:
    """Simple word tokenizer for change ratio."""
    return re.findall(r"\w+", text.lower())


def _token_change_ratio(text: str, aug_text: str) -> float:
    """Compute fraction of tokens changed between two texts."""
    tokens = _tokenize_simple(text)
    aug_tokens = _tokenize_simple(aug_text)
    if not tokens and not aug_tokens:
        return 0.0
    matcher = SequenceMatcher(None, tokens, aug_tokens)
    matched = sum(match.size for match in matcher.get_matching_blocks())
    total = max(len(tokens), len(aug_tokens))
    if total == 0:
        return 0.0
    return 1.0 - (matched / float(total))


def _save_boxplot(
    baseline: list[float],
    augmented: list[float],
    *,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    """Save a baseline vs augmented boxplot."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.boxplot([baseline, augmented], labels=["baseline", "augmented"], showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_hist_overlay(
    baseline: list[float],
    augmented: list[float],
    *,
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    """Save an overlaid histogram for baseline vs augmented values."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(baseline, bins=20, alpha=0.6, label="baseline", color="#2b6cb0")
    plt.hist(augmented, bins=20, alpha=0.6, label="augmented", color="#c05621")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_hist_single(values: list[float], *, title: str, xlabel: str, path: Path) -> None:
    """Save a single histogram plot."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20, color="#2b6cb0", edgecolor="#1a202c")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_bar_groups(
    labels: list[str],
    baseline_vals: list[float],
    augmented_vals: list[float],
    *,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    """Save grouped bar chart for baseline vs augmented bars."""
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    width = 0.38

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, baseline_vals, width, label="baseline", color="#2b6cb0")
    plt.bar(x + width / 2, augmented_vals, width, label="augmented", color="#c05621")
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_scatter(
    x_vals: list[float],
    y_vals: list[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    color: str,
    label: str,
) -> None:
    """Save a scatter plot of x vs y."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(x_vals, y_vals, alpha=0.5, s=12, color=color, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _paired_t_test(baseline: list[float], augmented: list[float]) -> dict[str, float]:
    """Paired t-test (normal approximation) with Cohen's d."""
    if len(baseline) != len(augmented) or len(baseline) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "cohen_d": 0.0, "n": len(baseline)}

    diffs = np.array(augmented, dtype=float) - np.array(baseline, dtype=float)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    n = len(diffs)
    if std_diff == 0:
        return {"t_stat": 0.0, "p_value": 1.0, "cohen_d": 0.0, "n": n}

    t_stat = mean_diff / (std_diff / np.sqrt(n))
    # Normal approximation for p-value (n is typically large here).
    z = abs(t_stat)
    p_value = 2 * (1 - NormalDist().cdf(z))
    cohen_d = mean_diff / std_diff
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohen_d": float(cohen_d),
        "n": n,
    }


def _predict_prob(
    model, tokenizer, text: str, target_label: int, *, device: str, max_length: int
) -> float:
    """Predict probability for a target label on a single text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0, target_label].item())


def _top_k_indices(scores: list[float], k: int) -> set[int]:
    """Return indices of top-k absolute scores."""
    if not scores:
        return set()
    scores_arr = np.abs(np.array(scores, dtype=float))
    idx = np.argsort(-scores_arr)[:k]
    return set(int(i) for i in idx)


def _remove_indices(words: list[str], indices: set[int]) -> str:
    """Remove words by index and return a string."""
    return " ".join([w for i, w in enumerate(words) if i not in indices]).strip()


def _keep_indices(words: list[str], indices: set[int]) -> str:
    """Keep words by index and return a string."""
    return " ".join([w for i, w in enumerate(words) if i in indices]).strip()

def _load_ig_samples(path: Path) -> dict[int, dict[str, object]]:
    """Load IG JSONL samples keyed by id."""
    samples: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            samples[int(record["id"])] = record
    return samples


def _top_tokens_for_counts(words: list[str], scores: list[float], k: int) -> list[str]:
    """Return normalized top-k tokens for counting."""
    scores_arr = np.abs(np.array(scores, dtype=float))
    idx = np.argsort(-scores_arr)[:k]
    return [_normalize_token(words[i]) for i in idx if _normalize_token(words[i])]


def _top_tokens_with_scores(words: list[str], scores: list[float], k: int) -> list[tuple[str, float]]:
    """Return top-k (token, score) pairs."""
    scores_arr = np.abs(np.array(scores, dtype=float))
    idx = np.argsort(-scores_arr)[:k]
    return [(words[i], float(scores[i])) for i in idx]


def _token_counts(
    records: Iterable[dict[str, object]],
    *,
    k: int,
    stopwords: set[str],
) -> Counter:
    """Count top-k tokens across IG samples with stopword filtering."""
    counts: Counter = Counter()
    for record in records:
        words = record.get("words", [])
        scores = record.get("word_attributions", [])
        if not words or not scores:
            continue
        tokens = _top_tokens_for_counts(words, scores, k=k)
        for tok in tokens:
            if tok in stopwords:
                continue
            counts[tok] += 1
    return counts


def _plot_top_tokens(
    baseline_counts: Counter,
    augmented_counts: Counter,
    *,
    top_n: int,
    title: str,
    path: Path,
) -> None:
    """Plot top token frequency deltas (baseline vs augmented)."""
    import matplotlib.pyplot as plt

    tokens = [tok for tok, _ in (baseline_counts + augmented_counts).most_common(top_n)]
    if not tokens:
        return
    baseline_vals = [baseline_counts.get(tok, 0) for tok in tokens]
    augmented_vals = [augmented_counts.get(tok, 0) for tok in tokens]

    x = np.arange(len(tokens))
    width = 0.38

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, baseline_vals, width, label="baseline", color="#2b6cb0")
    plt.bar(x + width / 2, augmented_vals, width, label="augmented", color="#c05621")
    plt.xticks(x, tokens, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Top-k frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _label_name(label: int | None) -> str:
    """Pretty label name for 0/1 with fallback."""
    mapping = {0: "negative", 1: "positive"}
    if label is None:
        return "unknown"
    return f"{label} ({mapping.get(label, str(label))})"


def _save_example_plot(
    record_base: dict[str, object],
    record_aug: dict[str, object],
    *,
    top_k: int,
    path: Path,
    example_id: int,
    gold_label: int | None,
) -> None:
    """Save a side-by-side bar chart for one baseline vs augmented example."""
    import matplotlib.pyplot as plt

    words_base = record_base.get("words", [])
    scores_base = record_base.get("word_attributions", [])
    words_aug = record_aug.get("words", [])
    scores_aug = record_aug.get("word_attributions", [])
    if not words_base or not scores_base or not words_aug or not scores_aug:
        return

    tokens_base = _top_tokens_with_scores(words_base, scores_base, k=top_k)
    tokens_aug = _top_tokens_with_scores(words_aug, scores_aug, k=top_k)
    base_labels = [tok for tok, _ in tokens_base]
    base_vals = [score for _, score in tokens_base]
    aug_labels = [tok for tok, _ in tokens_aug]
    aug_vals = [score for _, score in tokens_aug]

    text = record_base.get("text", "")
    pred_base = record_base.get("pred_label")
    pred_aug = record_aug.get("pred_label")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].barh(base_labels, base_vals, color="#2b6cb0")
    axes[0].set_title("Baseline top tokens")
    axes[1].barh(aug_labels, aug_vals, color="#c05621")
    axes[1].set_title("Augmented top tokens")
    for ax in axes:
        ax.invert_yaxis()

    title = (
        f"ID {example_id} | Gold: {_label_name(gold_label)} | "
        f"Pred base: {_label_name(pred_base)} | Pred aug: {_label_name(pred_aug)}"
    )
    fig.suptitle(title, fontsize=10)
    if text:
        fig.text(
            0.5,
            0.01,
            textwrap.fill(str(text), width=110),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout(rect=[0, 0.05, 1, 0.9])
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs augmented explainability outputs.")
    parser.add_argument("--baseline-run", default="baseline", help="Run id for baseline outputs.")
    parser.add_argument("--augmented-run", default="augmented", help="Run id for augmented outputs.")
    parser.add_argument(
        "--baseline-model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Model name or path for baseline faithfulness.",
    )
    parser.add_argument(
        "--augmented-model",
        default="models/augmented",
        help="Model name or path for augmented faithfulness.",
    )
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--metrics-dir", default="reports/metrics")
    parser.add_argument("--figures-dir", default="reports/figures/compare")
    parser.add_argument("--attrib-dir", default="reports/attributions")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "ig": "consistency_baseline.csv",
        "lime": "lime_consistency.csv",
        "attention": "attention_consistency.csv",
    }

    summary = {
        "baseline_run": args.baseline_run,
        "augmented_run": args.augmented_run,
        "methods": {},
        "stats": {},
        "faithfulness": {},
    }
    for method, filename in methods.items():
        baseline_path = metrics_dir / args.baseline_run / filename
        augmented_path = metrics_dir / args.augmented_run / filename
        if not baseline_path.exists() or not augmented_path.exists():
            continue

        base_metrics = _load_metric_csv(baseline_path)
        aug_metrics = _load_metric_csv(augmented_path)
        base_rows = _load_metric_rows(baseline_path)
        aug_rows = _load_metric_rows(augmented_path)
        shared_ids = sorted(set(base_rows) & set(aug_rows))

        summary["methods"][method] = {}
        summary["stats"][method] = {}
        for metric_name in base_metrics:
            baseline_vals = base_metrics[metric_name]
            augmented_vals = aug_metrics.get(metric_name, [])
            if not baseline_vals or not augmented_vals:
                continue
            summary["methods"][method][metric_name] = {
                "baseline_mean": float(np.mean(baseline_vals)),
                "augmented_mean": float(np.mean(augmented_vals)),
                "delta": float(np.mean(augmented_vals) - np.mean(baseline_vals)),
            }

            _save_boxplot(
                baseline_vals,
                augmented_vals,
                title=f"{method.upper()} {metric_name} (baseline vs augmented)",
                ylabel=metric_name,
                path=figures_dir / f"{method}_{metric_name}_boxplot.png",
            )
            _save_hist_overlay(
                baseline_vals,
                augmented_vals,
                title=f"{method.upper()} {metric_name} distribution",
                xlabel=metric_name,
                path=figures_dir / f"{method}_{metric_name}_hist.png",
            )

            if shared_ids:
                paired_base = [base_rows[i][metric_name] for i in shared_ids if metric_name in base_rows[i]]
                paired_aug = [aug_rows[i][metric_name] for i in shared_ids if metric_name in aug_rows[i]]
                n = min(len(paired_base), len(paired_aug))
                paired_base = paired_base[:n]
                paired_aug = paired_aug[:n]
                stats = _paired_t_test(paired_base, paired_aug)
                summary["stats"][method][metric_name] = {
                    **stats,
                    "mean_diff": float(np.mean(np.array(paired_aug) - np.array(paired_base)))
                    if n > 0
                    else 0.0,
                    "test": "paired_t_normal_approx",
                }

    attrib_dir = Path(args.attrib_dir)
    base_ig_path = attrib_dir / args.baseline_run / "ig_samples.jsonl"
    aug_ig_path = attrib_dir / args.augmented_run / "ig_samples.jsonl"
    gold_labels: dict[int, int] = {}
    try:
        from aet.data.datasets import load_sst2

        dataset = load_sst2(cache_dir=args.cache_dir)
        if args.split in dataset and "label" in dataset[args.split].column_names:
            gold_labels = {
                int(i): int(label) for i, label in enumerate(dataset[args.split]["label"])
            }
    except Exception:
        gold_labels = {}

    if base_ig_path.exists() and aug_ig_path.exists():
        base_samples = _load_ig_samples(base_ig_path)
        aug_samples = _load_ig_samples(aug_ig_path)
        shared_ids = sorted(set(base_samples) & set(aug_samples))

        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "while",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "as",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        }

        base_counts = _token_counts(
            [base_samples[i] for i in shared_ids], k=args.top_k, stopwords=stopwords
        )
        aug_counts = _token_counts(
            [aug_samples[i] for i in shared_ids], k=args.top_k, stopwords=stopwords
        )

        overlap_scores: list[float] = []
        for ex_id in shared_ids:
            base_rec = base_samples[ex_id]
            aug_rec = aug_samples[ex_id]
            base_tokens = set(
                _top_tokens_for_counts(
                    base_rec.get("words", []),
                    base_rec.get("word_attributions", []),
                    k=args.top_k,
                )
            )
            aug_tokens = set(
                _top_tokens_for_counts(
                    aug_rec.get("words", []),
                    aug_rec.get("word_attributions", []),
                    k=args.top_k,
                )
            )
            if not base_tokens or not aug_tokens:
                continue
            overlap_scores.append(len(base_tokens & aug_tokens) / float(args.top_k))

        _plot_top_tokens(
            base_counts,
            aug_counts,
            top_n=args.top_n,
            title=f"IG top-{args.top_k} token frequency",
            path=figures_dir / "ig_top_tokens.png",
        )

        if overlap_scores:
            _save_hist_single(
                overlap_scores,
                title=f"IG top-{args.top_k} overlap (baseline vs augmented)",
                xlabel="Top-k overlap",
                path=figures_dir / "ig_topk_overlap_hist.png",
            )
            summary["ig_topk_overlap_mean"] = float(np.mean(overlap_scores))
            summary["ig_topk_overlap_std"] = float(np.std(overlap_scores))

        top_base = {tok for tok, _ in base_counts.most_common(args.top_n)}
        top_aug = {tok for tok, _ in aug_counts.most_common(args.top_n)}
        if top_base or top_aug:
            summary["ig_top_tokens_jaccard"] = float(
                len(top_base & top_aug) / max(1, len(top_base | top_aug))
            )

        example_dir = figures_dir / "examples"
        for ex_id in shared_ids[: args.examples]:
            _save_example_plot(
                base_samples[ex_id],
                aug_samples[ex_id],
                top_k=min(args.top_k, 10),
                path=example_dir / f"ig_example_{ex_id}.png",
                example_id=ex_id,
                gold_label=gold_labels.get(ex_id),
            )

        try:
            from aet.models.distilbert import load_model_and_tokenizer
            from aet.utils.device import resolve_device

            device = resolve_device(args.device)
            tokenizer_base, model_base = load_model_and_tokenizer(args.baseline_model, num_labels=2)
            tokenizer_aug, model_aug = load_model_and_tokenizer(args.augmented_model, num_labels=2)
            model_base.to(device)
            model_aug.to(device)
            model_base.eval()
            model_aug.eval()

            comp_base: list[float] = []
            suff_base: list[float] = []
            comp_aug: list[float] = []
            suff_aug: list[float] = []

            for ex_id in shared_ids:
                base_rec = base_samples[ex_id]
                aug_rec = aug_samples[ex_id]
                words_base = base_rec.get("words", [])
                scores_base = base_rec.get("word_attributions", [])
                words_aug = aug_rec.get("words", [])
                scores_aug = aug_rec.get("word_attributions", [])
                if not words_base or not scores_base or not words_aug or not scores_aug:
                    continue

                idx_base = _top_k_indices(scores_base, k=args.top_k)
                idx_aug = _top_k_indices(scores_aug, k=args.top_k)

                base_text = base_rec.get("text", " ".join(words_base))
                aug_text = aug_rec.get("text", " ".join(words_aug))
                base_removed = _remove_indices(words_base, idx_base)
                base_kept = _keep_indices(words_base, idx_base)
                aug_removed = _remove_indices(words_aug, idx_aug)
                aug_kept = _keep_indices(words_aug, idx_aug)

                if not base_removed or not base_kept or not aug_removed or not aug_kept:
                    continue

                base_label = int(base_rec.get("pred_label", 0))
                aug_label = int(aug_rec.get("pred_label", 0))

                base_orig_prob = _predict_prob(
                    model_base,
                    tokenizer_base,
                    base_text,
                    base_label,
                    device=device,
                    max_length=args.max_length,
                )
                base_removed_prob = _predict_prob(
                    model_base,
                    tokenizer_base,
                    base_removed,
                    base_label,
                    device=device,
                    max_length=args.max_length,
                )
                base_kept_prob = _predict_prob(
                    model_base,
                    tokenizer_base,
                    base_kept,
                    base_label,
                    device=device,
                    max_length=args.max_length,
                )
                comp_base.append(base_orig_prob - base_removed_prob)
                suff_base.append(base_orig_prob - base_kept_prob)

                aug_orig_prob = _predict_prob(
                    model_aug,
                    tokenizer_aug,
                    aug_text,
                    aug_label,
                    device=device,
                    max_length=args.max_length,
                )
                aug_removed_prob = _predict_prob(
                    model_aug,
                    tokenizer_aug,
                    aug_removed,
                    aug_label,
                    device=device,
                    max_length=args.max_length,
                )
                aug_kept_prob = _predict_prob(
                    model_aug,
                    tokenizer_aug,
                    aug_kept,
                    aug_label,
                    device=device,
                    max_length=args.max_length,
                )
                comp_aug.append(aug_orig_prob - aug_removed_prob)
                suff_aug.append(aug_orig_prob - aug_kept_prob)

            if comp_base and comp_aug:
                _save_boxplot(
                    comp_base,
                    comp_aug,
                    title=f"Faithfulness (Comprehensiveness, top-{args.top_k})",
                    ylabel="orig_prob - removed_prob",
                    path=figures_dir / "faithfulness_comprehensiveness_boxplot.png",
                )
                _save_hist_overlay(
                    comp_base,
                    comp_aug,
                    title="Faithfulness (Comprehensiveness) distribution",
                    xlabel="orig_prob - removed_prob",
                    path=figures_dir / "faithfulness_comprehensiveness_hist.png",
                )
                summary["faithfulness"]["comprehensiveness"] = {
                    "baseline_mean": float(np.mean(comp_base)),
                    "augmented_mean": float(np.mean(comp_aug)),
                    "delta": float(np.mean(comp_aug) - np.mean(comp_base)),
                    "stats": _paired_t_test(comp_base, comp_aug),
                }

            if suff_base and suff_aug:
                _save_boxplot(
                    suff_base,
                    suff_aug,
                    title=f"Faithfulness (Sufficiency, top-{args.top_k})",
                    ylabel="orig_prob - kept_prob",
                    path=figures_dir / "faithfulness_sufficiency_boxplot.png",
                )
                _save_hist_overlay(
                    suff_base,
                    suff_aug,
                    title="Faithfulness (Sufficiency) distribution",
                    xlabel="orig_prob - kept_prob",
                    path=figures_dir / "faithfulness_sufficiency_hist.png",
                )
                summary["faithfulness"]["sufficiency"] = {
                    "baseline_mean": float(np.mean(suff_base)),
                    "augmented_mean": float(np.mean(suff_aug)),
                    "delta": float(np.mean(suff_aug) - np.mean(suff_base)),
                    "stats": _paired_t_test(suff_base, suff_aug),
                }

                if gold_labels:
                    consistency_path = metrics_dir / args.baseline_run / "consistency_baseline.csv"
                    if not consistency_path.exists():
                        consistency_path = None
                    text_pairs = _load_text_pairs(consistency_path) if consistency_path else {}
                ids_with_text = [i for i in shared_ids if i in text_pairs]
                base_orig_correct = 0
                base_aug_correct = 0
                base_flip = 0
                aug_orig_correct = 0
                aug_aug_correct = 0
                aug_flip = 0
                total = 0

                for ex_id in ids_with_text:
                    gold = gold_labels.get(ex_id)
                    if gold is None:
                        continue
                    text, aug_text = text_pairs[ex_id]
                    total += 1

                    base_pred = int(base_samples[ex_id].get("pred_label", 0))
                    inputs_base_aug = tokenizer_base(
                        aug_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=args.max_length,
                    )
                    inputs_base_aug = {k: v.to(device) for k, v in inputs_base_aug.items()}
                    with torch.no_grad():
                        base_pred_aug = int(
                            torch.argmax(model_base(**inputs_base_aug).logits, dim=-1).item()
                        )

                    aug_pred = int(aug_samples[ex_id].get("pred_label", 0))
                    inputs_aug_aug = tokenizer_aug(
                        aug_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=args.max_length,
                    )
                    inputs_aug_aug = {k: v.to(device) for k, v in inputs_aug_aug.items()}
                    with torch.no_grad():
                        aug_pred_aug = int(
                            torch.argmax(model_aug(**inputs_aug_aug).logits, dim=-1).item()
                        )

                    if base_pred == gold:
                        base_orig_correct += 1
                    if base_pred_aug == gold:
                        base_aug_correct += 1
                    if base_pred != base_pred_aug:
                        base_flip += 1

                    if aug_pred == gold:
                        aug_orig_correct += 1
                    if aug_pred_aug == gold:
                        aug_aug_correct += 1
                    if aug_pred != aug_pred_aug:
                        aug_flip += 1

                if total > 0:
                    summary["label_preservation"] = {
                        "total": total,
                        "baseline_orig_acc": base_orig_correct / total,
                        "baseline_aug_acc": base_aug_correct / total,
                        "baseline_flip_rate": base_flip / total,
                        "augmented_orig_acc": aug_orig_correct / total,
                        "augmented_aug_acc": aug_aug_correct / total,
                        "augmented_flip_rate": aug_flip / total,
                    }
                    _save_bar_groups(
                        ["orig_acc", "aug_acc"],
                        [base_orig_correct / total, base_aug_correct / total],
                        [aug_orig_correct / total, aug_aug_correct / total],
                        title="Label preservation accuracy",
                        ylabel="Accuracy",
                        path=figures_dir / "label_preservation_accuracy.png",
                    )
                    _save_bar_groups(
                        ["flip_rate"],
                        [base_flip / total],
                        [aug_flip / total],
                        title="Prediction flip rate",
                        ylabel="Flip rate",
                        path=figures_dir / "label_preservation_flip.png",
                    )
        except Exception:
            summary["faithfulness_error"] = "faithfulness computation failed"

    summary_path = figures_dir / "compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    for method, metrics in summary.get("stats", {}).items():
        for metric_name, stats in metrics.items():
            delta = stats.get("mean_diff", 0.0)
            p_value = stats.get("p_value", 1.0)
            cohen_d = stats.get("cohen_d", 0.0)
            n = stats.get("n", 0)
            direction = "higher" if delta >= 0 else "lower"
            signif = "significant" if p_value < 0.05 else "not significant"
            lines.append(
                f"{method.upper()} {metric_name}: delta={delta:.4f} ({direction}), "
                f"d={cohen_d:.3f}, p={p_value:.4f}, n={n} ({signif})"
            )

    faith = summary.get("faithfulness", {})
    if "comprehensiveness" in faith:
        comp = faith["comprehensiveness"]
        delta = comp.get("delta", 0.0)
        stats = comp.get("stats", {})
        lines.append(
            "FAITHFULNESS comprehensiveness: "
            f"delta={delta:.4f} (higher is better), "
            f"d={stats.get('cohen_d', 0.0):.3f}, p={stats.get('p_value', 1.0):.4f}, "
            f"n={stats.get('n', 0)}"
        )
    if "sufficiency" in faith:
        suff = faith["sufficiency"]
        delta = suff.get("delta", 0.0)
        stats = suff.get("stats", {})
        lines.append(
            "FAITHFULNESS sufficiency: "
            f"delta={delta:.4f} (lower is better), "
            f"d={stats.get('cohen_d', 0.0):.3f}, p={stats.get('p_value', 1.0):.4f}, "
            f"n={stats.get('n', 0)}"
        )

    label_pres = summary.get("label_preservation")
    if label_pres:
        lines.append(
            "LABEL preservation (orig/aug acc, flip rate): "
            f"baseline={label_pres.get('baseline_orig_acc', 0.0):.3f}/"
            f"{label_pres.get('baseline_aug_acc', 0.0):.3f}, "
            f"flip={label_pres.get('baseline_flip_rate', 0.0):.3f}; "
            f"augmented={label_pres.get('augmented_orig_acc', 0.0):.3f}/"
            f"{label_pres.get('augmented_aug_acc', 0.0):.3f}, "
            f"flip={label_pres.get('augmented_flip_rate', 0.0):.3f}"
        )

    if lines:
        summary_txt = figures_dir / "compare_summary.txt"
        summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    base_consistency = metrics_dir / args.baseline_run / "consistency_baseline.csv"
    aug_consistency = metrics_dir / args.augmented_run / "consistency_baseline.csv"
    if base_consistency.exists() and aug_consistency.exists():
        base_pairs = _load_text_pairs(base_consistency)
        aug_pairs = _load_text_pairs(aug_consistency)

        base_rows = _load_metric_rows(base_consistency)
        aug_rows = _load_metric_rows(aug_consistency)

        for metric_name in ("kendall_tau", "top_k_overlap", "cosine_similarity"):
            base_x = []
            base_y = []
            for ex_id, (text, aug_text) in base_pairs.items():
                if ex_id not in base_rows or metric_name not in base_rows[ex_id]:
                    continue
                base_x.append(_token_change_ratio(text, aug_text))
                base_y.append(base_rows[ex_id][metric_name])

            aug_x = []
            aug_y = []
            for ex_id, (text, aug_text) in aug_pairs.items():
                if ex_id not in aug_rows or metric_name not in aug_rows[ex_id]:
                    continue
                aug_x.append(_token_change_ratio(text, aug_text))
                aug_y.append(aug_rows[ex_id][metric_name])

            if base_x and aug_x:
                _save_scatter(
                    base_x,
                    base_y,
                    title=f"IG consistency vs change ratio ({metric_name})",
                    xlabel="Token change ratio",
                    ylabel=metric_name,
                    path=figures_dir / f"ig_change_scatter_{metric_name}_baseline.png",
                    color="#2b6cb0",
                    label="baseline",
                )
                _save_scatter(
                    aug_x,
                    aug_y,
                    title=f"IG consistency vs change ratio ({metric_name})",
                    xlabel="Token change ratio",
                    ylabel=metric_name,
                    path=figures_dir / f"ig_change_scatter_{metric_name}_augmented.png",
                    color="#c05621",
                    label="augmented",
                )


if __name__ == "__main__":
    main()
