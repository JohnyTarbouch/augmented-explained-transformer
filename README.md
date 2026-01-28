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

Create augmented SST-2 CSVs:
```powershell
python scripts/augment_sst2.py --split train --combined --augment-fraction 0.1
python scripts/augment_sst2.py --split validation --augment-fraction 0.1
```

Back-translation augmentation (optional):
```powershell
python scripts/augment_sst2.py --split train --combined --augment-fraction 0.1 --method backtranslation
```

LLM-based augmentation (optional):
```powershell
set-content -path .env -value "LLM_BASE_URL=""https://chat-ai.cluster.uni-hannover.de/v1""`nLLM_API_KEY=""your-key"""
python scripts/augment_sst2.py --split train --combined --augment-fraction 0.1 --method llm --llm-model llama-3.3-70b-instruct
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
python -m aet.cli --config configs/base.yaml --stage faithfulness
python -m aet.cli --config configs/base.yaml --stage sanity
python -m aet.cli --config configs/base.yaml --stage robustness
python -m aet.cli --config configs/base.yaml --stage counterfactual
python -m aet.cli --config configs/base.yaml --stage lime
```

Baseline vs augmented configs:
```powershell
python -m aet.cli --config configs/baseline.yaml --stage eval
python -m aet.cli --config configs/augmented.yaml --stage eval
```

Per-run outputs (avoid overwriting reports):
```yaml
project:
  run_id: baseline
```

Train on augmented CSVs (set in config first):
```yaml
training:
  train_data_path: data/interim/sst2_augmented/train_combined.csv
  eval_data_path: data/interim/sst2_augmented/validation_original.csv
```

LoRA variant (optional):
```yaml
training:
  lora:
    enabled: true
    r: 8
    alpha: 16
    dropout: 0.1
    target_modules: [q_lin, k_lin, v_lin, out_lin]
```

Explain an augmented CSV (set `explain.data_path` first):
```powershell
python -m aet.cli --config configs/base.yaml --stage explain
```

Single example IG (original vs augmented):
```powershell
python scripts/ig_single_example.py --augment
```

Single example LIME (original vs augmented):
```powershell
python scripts/lime_single_example.py --augment
```

Open notebook:
```powershell
jupyter notebook notebooks/inspect_sst2.ipynb
```

Optional: install plotting for histograms
```powershell
python -m pip install matplotlib
```

TextFooler robustness (optional):
```powershell
python -m pip install textattack
python -m aet.cli --config configs/baseline.yaml --stage robustness
python -m aet.cli --config configs/augmented.yaml --stage robustness
```

TextFooler counterfactuals:
```powershell
python -m aet.cli --config configs/baseline.yaml --stage counterfactual
python -m aet.cli --config configs/augmented.yaml --stage counterfactual
```

Compare counterfactual pairs:
```powershell
python scripts/compare_counterfactuals.py --max 10
```

Attention analysis (consistency + attention vs IG):
```powershell
python -m aet.cli --config configs/baseline.yaml --stage attention
python -m aet.cli --config configs/augmented.yaml --stage attention
```

LIME consistency (original vs augmented):
```powershell
python -m pip install -e .[lime]
python -m aet.cli --config configs/baseline.yaml --stage lime
python -m aet.cli --config configs/augmented.yaml --stage lime
```

Explainability comparison report (baseline vs augmented):
```powershell
python -m aet.cli --config configs/baseline.yaml --stage explain
python -m aet.cli --config configs/augmented.yaml --stage explain
python scripts/report_explainability.py --baseline-run baseline --augmented-run augmented
```
Outputs:
- `reports/figures/compare/` plots + example overlays
- `reports/figures/compare/compare_summary.json`
- `reports/figures/compare/compare_summary.txt`

Sanity overlay (baseline vs augmented):
```powershell
python scripts/compare_sanity_randomization.py
```
Output:
- `reports/figures/compare/sanity_ig_randomization_overlay.png`















## Multi-seed experiments (baseline + augmented) + compare plots

Baseline (no fine-tuning, multi-seed full pipeline):
```powershell
python scripts/run_full_multiseed.py --configs configs/baseline.yaml --seeds 13,21,42,1337 --stages eval,explain,consistency,attention,lime,faithfulness,sanity --aggregate
```

Augmented (fine-tuning, multi-seed full pipeline):
```powershell
python scripts/run_full_multiseed.py --configs configs/augmented.yaml --seeds 13,21,42,1337 --stages train,eval,explain,consistency,attention,lime,faithfulness,sanity --aggregate --force-model-path
```

Aggregated summaries:
- `reports/metrics/multiseed/<run_id>/multiseed_summary.json`

Mean ± std curves (sanity + faithfulness):
```powershell
python scripts/plot_multiseed_aggregate.py --summary reports/metrics/multiseed/baseline/multiseed_summary.json --prefix baseline
python scripts/plot_multiseed_aggregate.py --summary reports/metrics/multiseed/augmented/multiseed_summary.json --prefix augmented
```
Outputs (in `reports/figures/compare/`):
- `baseline_sanity_aggregate.png`
- `baseline_faithfulness_aggregate.png`
- `augmented_sanity_aggregate.png`
- `augmented_faithfulness_aggregate.png`

Pooled compare plots across seeds (IG / LIME / Attention):
```powershell
python scripts/report_explainability_multiseed.py --baseline-prefix baseline --augmented-prefix augmented --seeds 13,42,1337
```
Outputs (in `reports/figures/compare/`):
- `ig_*_boxplot_multiseed.png`, `ig_*_hist_multiseed.png`, `ig_*_meanstd_multiseed.png`
- `lime_*_boxplot_multiseed.png`, `lime_*_hist_multiseed.png`, `lime_*_meanstd_multiseed.png`
- `attention_*_boxplot_multiseed.png`, `attention_*_hist_multiseed.png`, `attention_*_meanstd_multiseed.png`
Summary:
- `reports/figures/compare/compare_summary_multiseed.json`

## Augmentation + label-flip analysis

Token distribution before vs after augmentation (top-k + JS divergence):
```powershell
python scripts/analyze_augmentation.py --original-csv data/interim/sst2_augmented/train_original.csv --augmented-csv data/interim/sst2_augmented/train_augmented.csv
```

Flip analysis (tokens changed and change-ratio distribution):
```powershell
python scripts/analyze_augmentation.py --original-csv data/interim/sst2_augmented/train_original.csv --augmented-csv data/interim/sst2_augmented/train_augmented.csv --consistency-csv reports/metrics/baseline_s42/consistency_baseline.csv
```

Outputs:
- `reports/figures/compare/augmentation_top_tokens.png`
- `reports/figures/compare/flip_change_ratio_hist.png`
- `reports/figures/compare/flip_changed_tokens.png`
- `reports/metrics/compare/augmentation_distribution_summary.json`

