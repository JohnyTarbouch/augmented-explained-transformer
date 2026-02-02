"""
Run multiple seeds and optionally aggregate summaries.
13,21,42,1337
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from aet.pipelines import (
    attention_pipeline,
    consistency_pipeline,
    counterfactual_pipeline,
    eval_pipeline,
    explain_pipeline,
    faithfulness_pipeline,
    lime_pipeline,
    robustness_pipeline,
    sanity_pipeline,
    train_pipeline,
)

# Map stage names (CLI) -> pipeline module implementing run(cfg).
STAGE_MAP = {
    "train": train_pipeline,
    "eval": eval_pipeline,
    "explain": explain_pipeline,
    "consistency": consistency_pipeline,
    "robustness": robustness_pipeline,
    "counterfactual": counterfactual_pipeline,
    "attention": attention_pipeline,
    "lime": lime_pipeline,
    "faithfulness": faithfulness_pipeline,
    "sanity": sanity_pipeline,
}

# Config sections that may have a `model_path` override. When training per seed,
# we often want all these stages to point at that seed's trained checkpoint.
MODEL_PATH_SECTIONS = [
    "evaluation",
    "explain",
    "consistency",
    "attention",
    "lime",
    "robustness",
    "counterfactual",
    "faithfulness",
    "sanity",
]


def _is_number(value: Any) -> bool:
    """Return True for numeric (non-bool) values."""
    # bool is a subclass of int, so explicitly exclude it.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _aggregate_numeric_list(values: list[list[float]]) -> dict[str, list[float]]:
    """Aggregate lists of equal length into element-wise mean/std."""
    # This is used for curves/vectors where we expect each seed to produce
    # the same-length list (e.g., AOPC curves over fixed fractions).
    #
    # If lengths disagree, we avoid silently mixing mismatched indices and instead
    # return raw values so the caller can diagnose the mismatch.
    length = len(values[0])
    for v in values:
        if len(v) != length:
            return {"values": values}

    means = []
    stds = []
    for idx in range(length):
        col = [v[idx] for v in values]
        mean = sum(col) / len(col)
        var = sum((x - mean) ** 2 for x in col) / len(col)
        means.append(float(mean))
        stds.append(float(var**0.5))
    return {"mean": means, "std": stds}


def _aggregate_values(values: list[Any]) -> Any:
    """Aggregate numbers, lists, or dicts recursively."""
    if not values:
        return None

    # 1) Scalar numeric values -> mean/std summary.
    if all(_is_number(v) for v in values):
        mean = sum(float(v) for v in values) / len(values)
        var = sum((float(v) - mean) ** 2 for v in values) / len(values)
        return {"mean": float(mean), "std": float(var**0.5)}

    # 2) List-of-lists of numeric values -> element-wise aggregation.
    if all(isinstance(v, list) for v in values) and all(
        all(_is_number(x) for x in v) for v in values
    ):
        return _aggregate_numeric_list(values)

    # 3) Dicts -> aggregate per key recursively.
    if all(isinstance(v, dict) for v in values):
        return _aggregate_dicts(values)

    # 4) Anything else / mixed types -> keep raw values (no assumptions).
    return {"values": values}


def _aggregate_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of dicts by key."""
    # We union keys so missing keys in some seeds don't break aggregation.
    keys = set()
    for d in dicts:
        keys.update(d.keys())

    result: dict[str, Any] = {}
    for key in keys:
        vals = [d[key] for d in dicts if key in d]
        result[key] = _aggregate_values(vals)
    return result


def _aggregate_sanity(list_of_lists: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate sanity randomization summaries by level name."""
    # Sanity summaries are lists of rows, each row keyed by a "level_name".
    # We aggregate across seeds per level_name (e.g., "classifier", "last_3", ...).
    by_level: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []

    # Preserve level order as it appears in the first seen summary.
    for rows in list_of_lists:
        for row in rows:
            name = str(row.get("level_name"))
            if name not in by_level:
                by_level[name] = []
                order.append(name)
            by_level[name].append(row)

    aggregated_levels = []
    for name in order:
        rows = by_level.get(name, [])

        # Only aggregate specific numeric summary fields.
        numeric_fields = {}
        for field in ["mean_kendall_tau", "mean_top_k_overlap", "mean_cosine_similarity"]:
            vals = [r.get(field) for r in rows if r.get(field) is not None]
            if vals:
                numeric_fields[field] = _aggregate_values(vals)

        aggregated_levels.append(
            {
                "level_name": name,
                **numeric_fields,
                "num_seeds": len(rows),
            }
        )
    return {"levels": aggregated_levels}


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    # Safe loader prevents arbitrary code execution via YAML tags.
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _parse_seeds(seed_str: str) -> list[int]:
    """Parse comma-separated seed string into list of ints."""
    # Example: "13,42,1337" -> [13, 42, 1337]
    return [int(s.strip()) for s in seed_str.split(",") if s.strip()]


def _collect_summaries(run_id: str) -> dict[str, Any]:
    """Collect all *_summary.json files for a run id."""
    # Convention: per-run metrics land in reports/metrics/<run_id>/.
    base_dir = Path("reports/metrics") / run_id
    summaries: dict[str, Any] = {}
    if not base_dir.exists():
        return summaries

    # Standard summary files (pattern: *_summary.json).
    for path in base_dir.glob("*_summary.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        summaries[path.name] = data

    # Sanity summary is a list; include it explicitly if present.
    # (This is redundant with glob if it matches, but harmless and makes intent explicit.)
    sanity_path = base_dir / "sanity_ig_randomization_summary.json"
    if sanity_path.exists():
        summaries[sanity_path.name] = json.loads(sanity_path.read_text(encoding="utf-8"))
    return summaries


def _aggregate_runs(run_ids: list[str], out_dir: Path) -> Path:
    """Aggregate summaries across multiple run ids and write JSON."""
    # Load all summaries per run.
    per_run = {rid: _collect_summaries(rid) for rid in run_ids}

    # Determine all summary filenames that appear in any run.
    all_files = set()
    for data in per_run.values():
        all_files.update(data.keys())

    aggregated: dict[str, Any] = {
        "runs": run_ids,
        "files": {},
    }

    # Aggregate file-by-file to keep structure consistent.
    for fname in sorted(all_files):
        payloads = [per_run[rid][fname] for rid in run_ids if fname in per_run[rid]]
        if not payloads:
            continue

        # Sanity has a special schema (list of dict rows), so aggregate differently.
        if fname == "sanity_ig_randomization_summary.json":
            aggregated["files"][fname] = _aggregate_sanity(payloads)
        else:
            aggregated["files"][fname] = _aggregate_values(payloads)

    # Write the aggregated output under the requested directory.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiseed_summary.json"
    out_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    return out_path


def _set_run_overrides(
    cfg: dict[str, Any],
    seed: int,
    run_id: str,
    stages: list[str],
    force_model_path: bool,
) -> None:
    """Inject seed/run_id and update model_path/output_dir for each stage."""
    # Ensure project section exists and set the run identifiers.
    cfg.setdefault("project", {})
    cfg["project"]["seed"] = seed
    cfg["project"]["run_id"] = run_id

    if "train" in stages:
        # Ensure per-seed training output dirs are unique.
        # This prevents collisions when running multiple seeds back-to-back.
        training = cfg.setdefault("training", {})
        output_dir = training.get("output_dir")
        if output_dir:
            training["output_dir"] = f"{output_dir}_{run_id}"

        # Point downstream stages at the per-seed model unless explicitly overridden.
        # If force_model_path is set, we always overwrite stage model_path.
        for section in MODEL_PATH_SECTIONS:
            if section in cfg:
                section_cfg = cfg[section] or {}
                if force_model_path or section_cfg.get("model_path") in (None, ""):
                    section_cfg["model_path"] = training.get("output_dir")
                    cfg[section] = section_cfg


def _parse_stages(stages_str: str) -> list[str]:
    """Parse comma-separated stages, supporting 'all'."""
    # Example: "train,eval" -> ["train", "eval"]
    # "all" expands to all known pipeline stages in STAGE_MAP.
    stages = [s.strip() for s in stages_str.split(",") if s.strip()]
    if "all" in stages:
        return list(STAGE_MAP.keys())
    return stages


def main() -> None:
    """CLI entrypoint for multi-seed experiments."""
    parser = argparse.ArgumentParser(description="Run full multi-seed experiments.")
    parser.add_argument(
        "--configs",
        default="configs/baseline.yaml,configs/augmented.yaml",
        help="Comma-separated config paths.",
    )
    parser.add_argument("--seeds", default="13,42,1337")
    parser.add_argument(
        "--stages",
        default="train,eval,explain,consistency,attention,lime,faithfulness,sanity,robustness,counterfactual",
        help="Comma-separated list of stages or 'all'.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate mean/std across seeds.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/metrics/multiseed",
        help="Where to write aggregated summary.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs if a stage fails.",
    )
    parser.add_argument(
        "--force-model-path",
        action="store_true",
        help="Force all stages to use the per-seed trained model path.",
    )
    args = parser.parse_args()

    # Parse CLI lists.
    configs = [Path(p.strip()) for p in args.configs.split(",") if p.strip()]
    seeds = _parse_seeds(args.seeds)
    stages = _parse_stages(args.stages)

    errors: list[str] = []

    # Run each config for each seed, stage-by-stage.
    for cfg_path in configs:
        cfg = _load_yaml(cfg_path)

        # Default run id: config-defined run_id, otherwise use filename stem.
        base_run_id = cfg.get("project", {}).get("run_id", cfg_path.stem)
        run_ids: list[str] = []

        for seed in seeds:
            run_id = f"{base_run_id}_s{seed}"
            run_ids.append(run_id)

            # Use a deep copy so per-seed overrides don't leak into other seeds.
            cfg_seed = copy.deepcopy(cfg)
            _set_run_overrides(cfg_seed, seed, run_id, stages, args.force_model_path)

            # Run each requested stage for this seed.
            for stage in stages:
                if stage not in STAGE_MAP:
                    raise ValueError(f"Unknown stage: {stage}")
                try:
                    STAGE_MAP[stage].run(cfg_seed)
                except Exception as exc:
                    # Optionally keep going across seeds/stages for long sweeps.
                    msg = f"{cfg_path} seed={seed} stage={stage} failed: {exc}"
                    if args.continue_on_error:
                        errors.append(msg)
                        continue
                    raise

        # Optional aggregation across all seeds for this config.
        if args.aggregate:
            out_path = _aggregate_runs(run_ids, Path(args.out_dir) / base_run_id)
            print(f"Wrote aggregated summary: {out_path}")

    # Print collected errors if any were tolerated.
    if errors:
        print("\nErrors during run:")
        for msg in errors:
            print(f"- {msg}")


if __name__ == "__main__":
    main()
