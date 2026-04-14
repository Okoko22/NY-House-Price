# NY House Price Prediction

NYC house-price pipeline with a modular Python package and Rich-powered CLI.

The code in `ny_house_price/` is a refactor of the notebook workflow in `final_draft.ipynb`, and the CLI is the operational entry point.

## What is actually in this repo

- `ny_house_price/data.py`: data cleaning rules and canonical cleaned dataset output.
- `ny_house_price/features.py`: `FeatureEngineer` + sklearn preprocessing pipeline.
- `ny_house_price/models.py`: train/test split, ensemble training, model load/save, prediction wrapper.
- `ny_house_price/cli.py`: Typer app with Rich output (`clean`, `train`, `evaluate`, `predict`, `run-all`).
- `ny_house_ensemble.pkl`: pre-trained ensemble artifact used directly by `predict`.

## How prediction works right now

`predict` loads the existing `ny_house_ensemble.pkl` and runs inference on an already-cleaned feature schema (same shape as training features).

```bash
uv run python -m ny_house_price.cli predict --input-path mydata/sample_input.csv
```

The CLI prints a Rich table preview by default, or writes a CSV if `--output-path` is provided.

> [!IMPORTANT]
> Input rows for prediction must follow the cleaned feature contract used by the model (`type`, `beds`, `bath`, `propertysqft`, `latitude`, `longitude`, `borough`, `zip_code`, `broker_name`, etc.).  
> `zip_code` is normalized to string during prediction to match the model pipeline expectations.

## CLI commands

```bash
uv run python -m ny_house_price.cli clean
uv run python -m ny_house_price.cli train
uv run python -m ny_house_price.cli evaluate
uv run python -m ny_house_price.cli predict --input-path mydata/sample_input.csv
uv run python -m ny_house_price.cli run-all
```

## Data and artifact paths

- Raw source data: `mydata/NY-House-Dataset.csv`
- Cleaned dataset: `mydata/df_clean.csv`
- Sample inference data: `mydata/sample_input.csv`
- Ensemble model artifact: `ny_house_ensemble.pkl`

## Typical usage

```bash
# 1) Clean raw data
uv run python -m ny_house_price.cli clean

# 2) Train and save a new model
uv run python -m ny_house_price.cli train

# 3) Evaluate hold-out MAPE
uv run python -m ny_house_price.cli evaluate

# 4) Predict from CSV
uv run python -m ny_house_price.cli predict --input-path mydata/sample_input.csv
```

## Environment notes

- Dependencies are pinned in `pyproject.toml` for reproducibility with this model/code path.
- `uv` is the expected execution path for CLI commands in this repository.
