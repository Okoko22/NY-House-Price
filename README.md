# NY House Price Prediction

A Python package for cleaning, training, evaluating, and predicting New York housing prices using a reusable CLI and modular pipeline.

## Overview

This repository contains a production-oriented machine learning package for NYC house price prediction. The code is organized into a package under `ny_house_price/` with separate responsibilities for data cleaning, feature engineering, model training, and command-line execution.

## What’s included

- `ny_house_price/data.py` - raw dataset cleaning, validation, and CSV export
- `ny_house_price/features.py` - feature engineering and preprocessing pipeline construction
- `ny_house_price/models.py` - train/test splitting, ensemble training, evaluation, and model persistence
- `ny_house_price/cli.py` - Typer-based CLI exposing the pipeline commands
- `pyproject.toml` - package metadata and dependency declarations
- `uv.lock` - uv-managed lockfile for reproducible dependency resolution

## Getting started

Use the repository root as the working directory. Dependencies are declared in `pyproject.toml` and managed through `uv`.

```bash
uv run python -m ny_house_price.cli --help
```

> Note: This project is designed to run within the existing Python virtual environment managed outside the repository. Do not sync or overwrite that environment here.

## CLI commands

Run the package via `uv run python -m ny_house_price.cli` and choose one of the available commands.

```bash
uv run python -m ny_house_price.cli clean
uv run python -m ny_house_price.cli train
uv run python -m ny_house_price.cli evaluate
uv run python -m ny_house_price.cli predict --input-path mydata/sample_input.csv
uv run python -m ny_house_price.cli run-all
```

### Command descriptions

- `clean` - read `mydata/NY-House-Dataset.csv`, clean it, and save the result to `mydata/df_clean.csv`
- `train` - train the ensemble model from cleaned data and save it to `ny_house_ensemble.pkl`
- `evaluate` - evaluate the saved model using a hold-out split from the cleaned dataset
- `predict` - load a saved model and predict prices for a provided input CSV
- `run-all` - execute clean, train, and evaluate in one flow

## Recommended workflow

1. Clean the dataset:

   ```bash
   uv run python -m ny_house_price.cli clean
   ```

2. Train the model:

   ```bash
   uv run python -m ny_house_price.cli train
   ```

3. Evaluate model quality:

   ```bash
   uv run python -m ny_house_price.cli evaluate
   ```

4. Predict new values from a CSV file:

   ```bash
   uv run python -m ny_house_price.cli predict --input-path mydata/sample_input.csv
   ```

## Notes

- The codebase is package-oriented, not notebook-oriented.
- Model persistence defaults to `ny_house_ensemble.pkl`.
- Cleaned dataset output defaults to `mydata/df_clean.csv`.
- The package is meant to be executed through the `ny_house_price` module and uv-managed workflow.
