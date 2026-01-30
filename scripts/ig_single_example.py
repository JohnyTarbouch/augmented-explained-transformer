"""
Run Integrated Gradients on a single example (for testing)
"""

from __future__ import annotations

import argparse

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
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=128)
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


if __name__ == "__main__":
    main()
