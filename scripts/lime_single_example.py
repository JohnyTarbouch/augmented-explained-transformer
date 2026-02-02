"""
Run LIME on a single example (for testing)
"""

from __future__ import annotations

import argparse

from aet.data.augment import augment_text
from aet.explain.lime_explainer import compute_lime_explanation
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device


# Human-readable labels for SST-2 style binary sentiment.
LABELS = {0: "negative", 1: "positive"}


def top_k_words(tokens: list[str], weights: list[float], k: int) -> list[tuple[str, float]]:
    """Return top-k tokens by absolute weight"""
    # Pair each token with its LIME weight, then sort by |weight| descending.
    # Using abs() surfaces both strongly positive and strongly negative contributions.
    pairs = list(zip(tokens, weights))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:k]


def main() -> None:
    """CLI entrypoint for single-example LIME"""
    parser = argparse.ArgumentParser(description="LIME for one text and its augmented version")
    parser.add_argument(
        "--text",
        default="I love ChatGPT but the UI is not great.",
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

    # LIME parameters:
    # - num_samples controls how many perturbed texts LIME generates (quality vs runtime)
    # - max_features controls how many tokens LIME keeps in its explanation
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--max-features", type=int, default=None)

    parser.add_argument(
        "--model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Model name or local path.",
    )
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    # Decide which augmented text to use:
    # 1) --aug-text overrides everything
    # 2) else if --augment is set, apply wordnet synonym replacement
    # 3) else fall back to the original text
    text = args.text
    if args.aug_text is not None:
        aug_text = args.aug_text
    elif args.augment:
        aug_text = augment_text(text, replace_prob=args.replace_prob, seed=args.seed)
    else:
        aug_text = text

    # Load model/tokenizer and move model to the chosen device.
    device = resolve_device(args.device)
    tokenizer, model = load_model_and_tokenizer(args.model, num_labels=2)
    model.to(device)
    model.eval()

    # Compute LIME on original text.
    lime_orig = compute_lime_explanation(
        model,
        tokenizer,
        text,
        num_samples=args.num_samples,
        max_features=args.max_features,
        seed=args.seed,
        device=device,
        max_length=args.max_length,
        class_names=["negative", "positive"],
    )

    # Compute LIME on augmented text (use a different seed so perturbation sampling differs).
    lime_aug = compute_lime_explanation(
        model,
        tokenizer,
        aug_text,
        num_samples=args.num_samples,
        max_features=args.max_features,
        seed=args.seed + 1,
        device=device,
        max_length=args.max_length,
        class_names=["negative", "positive"],
    )

    # Print results (quick sanity check / debugging output).
    print("Original:", text)
    print("Augmented:", aug_text)
    print(
        "Pred label:",
        lime_orig.pred_label,
        LABELS.get(lime_orig.pred_label, str(lime_orig.pred_label)),
    )
    print("Top words (orig):", top_k_words(lime_orig.tokens, lime_orig.token_weights, k=args.top_k))
    print("Top words (aug):", top_k_words(lime_aug.tokens, lime_aug.token_weights, k=args.top_k))


if __name__ == "__main__":
    main()
