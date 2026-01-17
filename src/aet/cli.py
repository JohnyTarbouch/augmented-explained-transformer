from __future__ import annotations

import argparse

from aet.config import load_config
from aet.pipelines import consistency_pipeline, explain_pipeline, train_pipeline
from aet.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Augmented-Explained-Transformer CLI")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--stage",
        choices=["train", "explain", "consistency"],
        default="train",
        help="Pipeline stage to run.",
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    if args.stage == "train":
        train_pipeline.run(cfg)
    elif args.stage == "explain":
        explain_pipeline.run(cfg)
    else:
        consistency_pipeline.run(cfg)


if __name__ == "__main__":
    main()
