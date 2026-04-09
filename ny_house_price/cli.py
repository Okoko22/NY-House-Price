from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .data import clean_dataset, load_raw_data, save_clean_data
from .features import FeatureEngineer, SanitiseFeatureNames
from .models import (
    evaluate_model,
    load_clean_data,
    load_model,
    predict_prices,
    save_model,
    split_train_test,
    train_ensemble,
)

app = typer.Typer(help="NY House Price pipeline CLI")
console = Console()

# Compatibility symbols for legacy pickled models.
_ = (FeatureEngineer, SanitiseFeatureNames)


def _report_table(report: dict[str, int]) -> Table:
    table = Table(title="Cleaning Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Initial rows", f"{report['initial_rows']:,}")
    table.add_row("Dropped duplicates", f"{report['dropped_duplicates']:,}")
    table.add_row("Dropped sentinels", f"{report['dropped_sentinels']:,}")
    table.add_row("Dropped corrupt bath", f"{report['dropped_corrupt_bath']:,}")
    table.add_row("Dropped non-transactional", f"{report['dropped_nontransactional_types']:,}")
    table.add_row("Final rows", f"{report['final_rows']:,}")
    table.add_row("Total dropped", f"{report['total_dropped']:,}")
    return table


@app.command()
def clean(
    raw_path: Path = Path("mydata/NY-House-Dataset.csv"),
    output_path: Path = Path("mydata/df_clean.csv"),
) -> None:
    df = load_raw_data(raw_path)
    df_clean, report = clean_dataset(df)
    save_clean_data(df_clean, output_path)
    console.print(_report_table(report))
    console.print(f"[bold green]Saved cleaned dataset:[/bold green] {output_path}")


@app.command()
def train(
    clean_path: Path = Path("mydata/df_clean.csv"),
    model_path: Path = Path("ny_house_ensemble.pkl"),
    use_gpu: bool = False,
) -> None:
    if not clean_path.exists():
        console.print(f"[bold red]Missing clean dataset:[/bold red] {clean_path}")
        raise typer.Exit(code=1)

    df = load_clean_data(clean_path)
    X_train, X_test, y_train, y_test = split_train_test(df)
    model = train_ensemble(X_train, y_train, use_gpu=use_gpu)
    save_model(model, model_path)
    metrics = evaluate_model(model, X_test, y_test)
    console.print(f"[bold green]Model saved:[/bold green] {model_path}")
    console.print(f"[bold]Hold-out MAPE:[/bold] {metrics['mape']:.4f}")


@app.command()
def evaluate(
    clean_path: Path = Path("mydata/df_clean.csv"),
    model_path: Path = Path("ny_house_ensemble.pkl"),
) -> None:
    if not clean_path.exists():
        console.print(f"[bold red]Missing clean dataset:[/bold red] {clean_path}")
        raise typer.Exit(code=1)
    if not model_path.exists():
        console.print(f"[bold red]Missing model file:[/bold red] {model_path}")
        raise typer.Exit(code=1)

    df = load_clean_data(clean_path)
    _, X_test, _, y_test = split_train_test(df)
    model = load_model(model_path)
    metrics = evaluate_model(model, X_test, y_test)
    console.print("[bold green]Evaluation complete[/bold green]")
    console.print(f"[bold]Hold-out MAPE:[/bold] {metrics['mape']:.4f}")


@app.command()
def predict(
    model_path: Path = Path("ny_house_ensemble.pkl"),
    input_path: Path = Path("mydata/sample_input.csv"),
    output_path: Path | None = None,
) -> None:
    if not model_path.exists():
        console.print(f"[bold red]Missing model file:[/bold red] {model_path}")
        raise typer.Exit(code=1)
    if not input_path.exists():
        console.print(f"[bold red]Missing input CSV:[/bold red] {input_path}")
        raise typer.Exit(code=1)

    model = load_model(model_path)
    input_df = load_clean_data(input_path)
    predictions_df = predict_prices(model, input_df)

    if output_path is not None:
        predictions_df.to_csv(output_path, index=False)
        console.print(f"[bold green]Predictions written:[/bold green] {output_path}")
        return

    preview = Table(title="Prediction Preview")
    preview.add_column("Row", justify="right")
    preview.add_column("Predicted Price", justify="right", style="magenta")
    for idx, value in predictions_df["predicted_price"].head(10).items():
        preview.add_row(str(idx), f"{value:,.2f}")
    console.print(preview)


@app.command("run-all")
def run_all(
    raw_path: Path = Path("mydata/NY-House-Dataset.csv"),
    clean_path: Path = Path("mydata/df_clean.csv"),
    model_path: Path = Path("ny_house_ensemble.pkl"),
    use_gpu: bool = False,
) -> None:
    clean(raw_path=raw_path, output_path=clean_path)
    train(clean_path=clean_path, model_path=model_path, use_gpu=use_gpu)
    evaluate(clean_path=clean_path, model_path=model_path)


if __name__ == "__main__":
    app()
