from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from .features import build_pipeline


def load_clean_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_test(
    df: pd.DataFrame,
    target: str = "price",
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    df["price_cat"] = pd.qcut(df[target], q=5, labels=False, duplicates="drop")
    train_data, test_data = train_test_split(
        df,
        test_size=test_size,
        stratify=df["price_cat"],
        random_state=random_state,
    )
    train_data = train_data.drop(columns=["price_cat"])
    test_data = test_data.drop(columns=["price_cat"])
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]
    return X_train, X_test, y_train, y_test


def _get_base_regressors(use_gpu: bool = False) -> dict[str, object]:
    if use_gpu:
        return {
            "LightGBM": LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=42,
                verbosity=-1,
                device="gpu",
            ),
            "XGBoost": XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
                tree_method="hist",
                device="cuda",
            ),
            "CatBoost": CatBoostRegressor(
                iterations=400,
                learning_rate=0.05,
                random_seed=42,
                verbose=0,
                task_type="GPU",
                devices="0",
            ),
            "Hist GB": HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, random_state=42
            ),
        }
    return {
        "LightGBM": LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            tree_method="hist",
        ),
        "CatBoost": CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
            thread_count=0,
        ),
        "Hist GB": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=42
        ),
    }


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_gpu: bool = False,
) -> VotingRegressor:
    regressors = _get_base_regressors(use_gpu=use_gpu)
    pipelines = {}
    for name, reg in regressors.items():
        pipe = build_pipeline(reg)
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

    ensemble = VotingRegressor(
        estimators=[(name, pipe) for name, pipe in pipelines.items()], n_jobs=1
    )
    ensemble.fit(X_train, y_train)
    return ensemble


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, predictions)
    return {"mape": float(mape)}


def predict_prices(model: object, df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=["price"]) if "price" in df.columns else df.copy()
    if "zip_code" in features.columns:
        features["zip_code"] = features["zip_code"].astype(str)
    output = df.copy()
    output["predicted_price"] = model.predict(features)
    return output


def save_model(model: object, path: Path) -> None:
    joblib.dump(model, path)


def load_model(path: Path) -> object:
    return joblib.load(path)
