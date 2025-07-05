# NeurIPS-Ariel-Data-Decoder

This repository contains a full pipeline to compete in the 2025 NeurIPS Ariel Spectral Prediction Challenge. It implements the V18 Finalist Model, including full calibration-aware preprocessing, uncertainty modeling, ensemble prediction, and Kaggle-ready submission packaging.

## Key Features
- Calibration-aware physical preprocessing: (signal - dark) / flat
- ConvNeXt + ViT architecture with uncertainty quantification
- Ensemble prediction with MC Dropout and Test-Time Augmentation
- Fully compatible with Kaggle / Colab

## To Train
```bash
python scripts/train_v18.py
```

## To Generate Submission
```bash
python scripts/run_submission.py
```