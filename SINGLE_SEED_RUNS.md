# Single Seed Experiments

This guide explains how to run the pipeline for a **single seed**, which is useful for debugging.

## How to Run

All stages are run through the main CLI:
```powershell
python -m aet.cli --config <cfg> --stage <stage>
```

**Available Stages (in order):**
`train`, `eval`, `explain`, `consistency`, `attention`, `lime`, `faithfulness`, `sanity`, `robustness`, `counterfactual`

---

## 1. Baseline Run (Fine-tuning)

This runs the pipeline on the standard SST-2 validation set using the pre-trained `distilbert-base-uncased-finetuned-sst-2-english`.

**Command (Powershell Loop):**
```powershell
$stages = "train","eval","explain","consistency","attention","lime","faithfulness","sanity","robustness","counterfactual"
foreach ($s in $stages) { 
    Write-Host "Running stage: $s"
    python -m aet.cli --config configs/baseline.yaml --stage $s 
}
```

---

## 2. Augmented Run (With Fine-tuning)

This fine-tunes DistilBERT on your augmented dataset before running the explainability pipeline.

### Prerequisites
Ensure you have generated the augmented data first:
```powershell
# Create augmented training and validation sets
python scripts/augment_sst2.py --split train --combined --augment-fraction 0.1
python scripts/augment_sst2.py --split validation --augment-fraction 0.1
```

### Command (Powershell Loop):
```powershell
$stages = "train","eval","explain","consistency","attention","lime","faithfulness","sanity","robustness","counterfactual"
foreach ($s in $stages) { 
    Write-Host "Running stage: $s"
    python -m aet.cli --config configs/augmented.yaml --stage $s 
}
```

---

