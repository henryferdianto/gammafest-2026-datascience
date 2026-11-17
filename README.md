# Gammafest 2026 — National Data Science Competition

![ci](https://github.com/henryferdianto/gammafest-2026-datascience/actions/workflows/ci.yml/badge.svg)

> **National Finalist (Top 30) — IPB University**

## Overview

An end-to-end tabular machine learning pipeline for the Gammafest national data science competition, featuring domain-driven feature engineering, outlier treatment, stratified cross-validation and model ensembling.

## Approach

1. **EDA** and domain-driven feature engineering.
2. **Outlier treatment** and robust preprocessing.
3. **Model benchmark**: gradient boosting variants (LightGBM, XGBoost, CatBoost).
4. **Ensembling** and stratified cross-validation.
5. **Submission** generation with metric-focused tuning.

## Repository Structure

```
gammafest-2026-datascience/
├── src/features.py
├── src/train.py
├── src/predict.py
├── requirements.txt
└── README.md
```

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `LightGBM` · `XGBoost` · `CatBoost`

## Getting Started

```bash
git clone https://github.com/henryferdianto/gammafest-2026-datascience.git
cd gammafest-2026-datascience
pip install -r requirements.txt
```

```bash
python src/train.py --train data/raw/train.csv --target target
python src/predict.py --test data/raw/test.csv
```


<!-- completed -->

## Quickstart with Sample Data

The repository ships a synthetic data generator so the full pipeline can be run
without any private dataset:

```bash
make sample && python -m src.train && python -m src.predict
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

## Project Status

This repository contains the project documentation and a runnable reference
implementation. A synthetic data generator is included so every pipeline stage
can be executed end-to-end; replace the sample data with your own datasets in
`data/raw/` to reproduce real results.

## License

MIT License — see [LICENSE](LICENSE).


---

**Author:** Rifky Henry Ferdianto — Data Science undergraduate at ITERA  
GitHub: [@henryferdianto](https://github.com/henryferdianto) · Portfolio: [rifky-henry-ferdianto](https://github.com/henryferdianto/rifky-henry-ferdianto)
