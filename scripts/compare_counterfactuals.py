from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def index_by_text(items: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for item in items:
        text = item.get("text")
        if text is None:
            continue
        if text not in indexed:
            indexed[text] = item
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TextAttack counterfactuals.")
    parser.add_argument(
        "--baseline",
        default="reports/metrics/baseline/counterfactual_textfooler.jsonl",
        help="Path to baseline counterfactuals JSONL.",
    )
    parser.add_argument(
        "--augmented",
        default="reports/metrics/augmented/counterfactual_textfooler.jsonl",
        help="Path to augmented counterfactuals JSONL.",
    )
    parser.add_argument(
        "--out",
        default="reports/metrics/counterfactual_comparison.jsonl",
        help="Output JSONL for paired examples.",
    )
    parser.add_argument("--max", type=int, default=10, help="Max paired examples to save.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    augmented_path = Path(args.augmented)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_items = load_jsonl(baseline_path)
    augmented_items = load_jsonl(augmented_path)
    baseline_by_text = index_by_text(baseline_items)
    augmented_by_text = index_by_text(augmented_items)

    common_texts = list(set(baseline_by_text) & set(augmented_by_text))
    rng = random.Random(args.seed)
    rng.shuffle(common_texts)
    selected = common_texts[: max(0, args.max)]

    with out_path.open("w", encoding="utf-8") as handle:
        for text in selected:
            base = baseline_by_text[text]
            aug = augmented_by_text[text]
            record = {
                "text": text,
                "baseline_counterfactual": base.get("counterfactual"),
                "augmented_counterfactual": aug.get("counterfactual"),
                "gold_label": base.get("gold_label"),
                "baseline_orig_pred": base.get("orig_pred"),
                "augmented_orig_pred": aug.get("orig_pred"),
                "baseline_adv_pred": base.get("adv_pred"),
                "augmented_adv_pred": aug.get("adv_pred"),
                "baseline_num_changed": base.get("num_changed"),
                "augmented_num_changed": aug.get("num_changed"),
                "baseline_change_ratio": base.get("change_ratio"),
                "augmented_change_ratio": aug.get("change_ratio"),
                "baseline_num_queries": base.get("num_queries"),
                "augmented_num_queries": aug.get("num_queries"),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Paired examples: {len(selected)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
