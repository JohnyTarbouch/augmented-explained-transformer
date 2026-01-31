"""
Run Integrated Gradients on a single example (for testing).

Usage:
  python scripts/ig_single_example.py --augment --top-k 5

Compare two models and save a single plot (baseline vs augmented model):
  python scripts/ig_single_example.py --augment --model distilbert-base-uncased-finetuned-sst-2-english --compare-model models/augmented_augmented_s42 --out reports/figures/compare/examples/ig_compare_example.png
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from aet.data.augment import augment_text
from aet.explain.integrated_gradients import compute_integrated_gradients, predict_label
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device


LABELS = {0: "negative", 1: "positive"}


def top_k_words(ig, k: int) -> list[tuple[str, float]]:
    """Return top-k words by absolute attribution"""
    pairs = list(zip(ig.words, ig.word_attributions))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:k]


def _plot_compare(
    *,
    text: str,
    aug_text: str,
    base_name: str,
    aug_name: str,
    base_orig,
    base_aug,
    aug_orig,
    aug_aug,
    top_k: int,
    out_path: Path,
) -> None:
    """Save a 2x2 bar plot comparing IG attributions for two models."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    panels = [
        (f"{base_name} (orig)", base_orig),
        (f"{base_name} (aug)", base_aug),
        (f"{aug_name} (orig)", aug_orig),
        (f"{aug_name} (aug)", aug_aug),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()
    for ax, (title, ig) in zip(axes, panels):
        tokens = top_k_words(ig, k=top_k)
        labels = [t for t, _ in tokens][::-1]
        scores = [s for _, s in tokens][::-1]
        ax.barh(labels, scores, color="#2b6cb0")
        ax.set_title(title, fontsize=9)
        ax.invert_yaxis()

    fig.suptitle("IG top-k attributions (baseline vs augmented)", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "Orig: " + textwrap.fill(text, width=80) + "\nAug: " + textwrap.fill(aug_text, width=80),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """CLI entrypoint for single-example IG"""
    parser = argparse.ArgumentParser(description="IG for one text and its augmented version")
    parser.add_argument(
        "--text",
        default="The movie was surprisingly good and the acting was excellent.",
        help="Input sentence.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply synonym replacement to create augmented text.",
    )
    parser.add_argument(
        "--aug-text",
        default=None,
        help="Optional custom augmented sentence (skips synonym replacement).",
    )
    parser.add_argument("--replace-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--compare-model",
        default=None,
        help="Optional second model to compare (e.g., augmented checkpoint).",
    )
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for a combined plot (requires --compare-model).",
    )
    args = parser.parse_args()

    text = args.text
    if args.aug_text is not None:
        aug_text = args.aug_text
    elif args.augment:
        aug_text = augment_text(text, replace_prob=args.replace_prob, seed=args.seed)
    else:
        aug_text = text

    device = resolve_device(args.device)
    tokenizer, model = load_model_and_tokenizer(args.model, num_labels=2)
    model.to(device)
    model.eval()

    pred = predict_label(model, tokenizer, text, device=device, max_length=args.max_length)
    ig_orig = compute_integrated_gradients(
        model,
        tokenizer,
        text,
        target_label=pred,
        n_steps=args.n_steps,
        device=device,
        max_length=args.max_length,
    )
    ig_aug = compute_integrated_gradients(
        model,
        tokenizer,
        aug_text,
        target_label=pred,
        n_steps=args.n_steps,
        device=device,
        max_length=args.max_length,
    )

    print("Original:", text)
    print("Augmented:", aug_text)
    print("Pred label:", pred, LABELS.get(pred, str(pred)))
    print("Top words (orig):", top_k_words(ig_orig, k=args.top_k))
    print("Top words (aug):", top_k_words(ig_aug, k=args.top_k))

    if args.compare_model:
        tokenizer_cmp, model_cmp = load_model_and_tokenizer(args.compare_model, num_labels=2)
        model_cmp.to(device)
        model_cmp.eval()

        pred_cmp = predict_label(model_cmp, tokenizer_cmp, text, device=device, max_length=args.max_length)
        ig_cmp_orig = compute_integrated_gradients(
            model_cmp,
            tokenizer_cmp,
            text,
            target_label=pred_cmp,
            n_steps=args.n_steps,
            device=device,
            max_length=args.max_length,
        )
        ig_cmp_aug = compute_integrated_gradients(
            model_cmp,
            tokenizer_cmp,
            aug_text,
            target_label=pred_cmp,
            n_steps=args.n_steps,
            device=device,
            max_length=args.max_length,
        )

        print("\n[Compare model]")
        print("Pred label:", pred_cmp, LABELS.get(pred_cmp, str(pred_cmp)))
        print("Top words (orig):", top_k_words(ig_cmp_orig, k=args.top_k))
        print("Top words (aug):", top_k_words(ig_cmp_aug, k=args.top_k))

        if args.out:
            _plot_compare(
                text=text,
                aug_text=aug_text,
                base_name=Path(args.model).name,
                aug_name=Path(args.compare_model).name,
                base_orig=ig_orig,
                base_aug=ig_aug,
                aug_orig=ig_cmp_orig,
                aug_aug=ig_cmp_aug,
                top_k=args.top_k,
                out_path=Path(args.out),
            )
            print(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
