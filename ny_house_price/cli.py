from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .data import clean_dataset, load_raw_data, print_cleaning_report, save_clean_data
from .models import (
    evaluate_model,
    load_clean_data,
    load_model,
    save_model,
    split_train_test,
    train_ensemble,
)

app = typer.Typer(help="NY House Price pipeline CLI")


@app.command()
def clean(
    raw_path: Path = Path("mydata/NY-House-Dataset.csv"),
    output_path: Path = Path("mydata/df_clean.csv"),
) -> None:
    """Clean the raw dataset and save a cleaned CSV."""
    df = load_raw_data(raw_path)
    df_clean, report = clean_dataset(df)
    save_clean_data(df_clean, output_path)
    print_cleaning_report(report)
    typer.echo(f"Cleaned dataset saved to: {output_path}")


@app.command()
def train(
    clean_path: Path = Path("mydata/df_clean.csv"),
    model_path: Path = Path("ny_house_ensemble.pkl"),
    use_gpu: bool = False,
) -> None:
    """Train an ensemble model from cleaned data and save it."""
    if not clean_path.exists():
        typer.echo(f"Clean file not found: {clean_path}")
        raise typer.Exit(code=1)

    df = load_clean_data(clean_path)
    X_train, X_test, y_train, y_test = split_train_test(df)
    model = train_ensemble(X_train, y_train, use_gpu=use_gpu)
    save_model(model, model_path)
    metrics = evaluate_model(model, X_test, y_test)
    typer.echo(f"Model trained and saved to: {model_path}")
    typer.echo(f"Hold-out MAPE: {metrics['mape']:.4f}")


@app.command()
def evaluate(
    clean_path: Path = Path("mydata/df_clean.csv"),
    model_path: Path = Path("ny_house_ensemble.pkl"),
) -> None:
    """Evaluate an existing model against the cleaned hold-out test split."""
    if not clean_path.exists():
        typer.echo(f"Clean file not found: {clean_path}")
        raise typer.Exit(code=1)
    if not model_path.exists():
        typer.echo(f"Model file not found: {model_path}")
        raise typer.Exit(code=1)

    df = load_clean_data(clean_path)
    _, X_test, _, y_test = split_train_test(df)
    model = load_model(model_path)
    metrics = evaluate_model(model, X_test, y_test)
    typer.echo(f"Model evaluated on hold-out set")
    typer.echo(f"Hold-out MAPE: {metrics['mape']:.4f}")


@app.command()
def predict(
    model_path: Path = Path("ny_house_ensemble.pkl"),
    input_path: Path = None,
    output_path: Path = None,
) -> None:
    """Load a model and predict target prices for input rows."""
    if not model_path.exists():
        typer.echo(f"Model file not found: {model_path}")
        raise typer.Exit(code=1)
    if input_path is None or not input_path.exists():
        typer.echo("Please provide an existing --input-path CSV file.")
        raise typer.Exit(code=1)

    model = load_model(model_path)
    df = pd.read_csv(input_path)
    if "price" in df.columns:
        X = df.drop(columns=["price"])
    else:
        X = df.copy()

    predictions = model.predict(X)
    output = df.copy()
    output["predicted_price"] = predictions
    if output_path is not None:
        output.to_csv(output_path, index=False)
        typer.echo(f"Predictions written to: {output_path}")
    else:
        typer.echo(output.to_string(index=False))


@app.command()
def run_all(
    raw_path: Path = Path("mydata/NY-House-Dataset.csv"),
    clean_path: Path = Path("mydata/df_clean.csv"),
    model_path: Path = Path("ny_house_ensemble.pkl"),
    use_gpu: bool = False,
) -> None:
    """Run the full clean-train-evaluate flow."""
    df = load_raw_data(raw_path)
    df_clean, report = clean_dataset(df)
    save_clean_data(df_clean, clean_path)
    print_cleaning_report(report)

    X_train, X_test, y_train, y_test = split_train_test(df_clean)
    model = train_ensemble(X_train, y_train, use_gpu=use_gpu)
    save_model(model, model_path)
    metrics = evaluate_model(model, X_test, y_test)

    typer.echo(f"Full pipeline complete")
    typer.echo(f"Cleaned data saved to: {clean_path}")
    typer.echo(f"Model saved to: {model_path}")
    typer.echo(f"Hold-out MAPE: {metrics['mape']:.4f}")
