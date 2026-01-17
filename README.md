# Augmented-Explained-Transformer

Research codebase for measuring explanation consistency of DistilBERT sentiment analysis under text augmentation.

## Project layout
- `configs/`: experiment configs
- `data/`: raw/interim/processed/external datasets
- `models/`: saved checkpoints
- `reports/`: figures and tables
- `paper/`: LaTeX report
- `src/aet/`: library code
- `scripts/`: entry-point scripts
- `tests/`: unit tests
- `notebooks/`: exploration notebooks

## Quickstart
1. `python -m pip install -U pip`
2. `python -m pip install -e .[dev]`
3. `python -m aet.cli --config configs/base.yaml --stage train`

## Data
- `scripts/download_sst2.py` downloads SST-2 CSVs into `data/raw/sst2/`.
- Keep large artifacts out of git see `.gitignore`.

## Next steps
- pipelines `src/aet/pipelines/`.
- IG computation `src/aet/explain/`.
- metrics `src/aet/metrics/`.
