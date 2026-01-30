"""
Download SST-2 via Hugging Face and save csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    """Entrypoint to download and materialize SST-2 splits"""
    parser = argparse.ArgumentParser(description="Download SST-2 to CSV")
    parser.add_argument("--out-dir", default="data/raw/sst2")
    parser.add_argument("--cache-dir", default="data/raw/hf_cache")
    args = parser.parse_args()

    dataset = load_dataset("glue", "sst2", cache_dir=args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, split_ds in dataset.items():
        # Write each split to a separate csv file.
        split_ds.to_csv(out_dir / f"{split}.csv", index=False)


if __name__ == "__main__":
    main()
