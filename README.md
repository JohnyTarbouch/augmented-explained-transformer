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

## Commands (so far)
Setup:
```powershell
python -m pip install -U pip
python -m pip install -e .[dev]
```

GPU install (optional, CUDA 12.1):
```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Download data:
```powershell
python scripts/download_sst2.py
```

Inspect data:
```powershell
python scripts/inspect_sst2.py
```

Enable WordNet augmentation:
```powershell
python -m nltk.downloader wordnet
```

Run pipelines:
```powershell
python -m aet.cli --config configs/base.yaml --stage train
python -m aet.cli --config configs/base.yaml --stage eval
python -m aet.cli --config configs/base.yaml --stage explain
python -m aet.cli --config configs/base.yaml --stage consistency
```

Single example IG (original vs augmented):
```powershell
python scripts/ig_single_example.py --augment
```

Open notebook:
```powershell
jupyter notebook notebooks/inspect_sst2.ipynb
```

Optional: install plotting for histograms
```powershell
python -m pip install matplotlib
```
