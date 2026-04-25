from __future__ import annotations

import __main__
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from ny_house_price.features import FeatureEngineer, SanitiseFeatureNames

#  Page config ─
st.set_page_config(
    layout="wide",
    page_title="NY House Price Predictor",
    page_icon="🏙️",
    initial_sidebar_state="expanded",
)

#  CSS ─
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Barlow+Condensed:wght@400;600;700&family=Space+Grotesk:wght@300;400;500&display=swap');

    :root {
        --bg:        #080C12;
        --surface:   #0E1420;
        --surface2:  #121A28;
        --border:    #1A2235;
        --border-hi: #243050;
        --text:      #D6E4F7;
        --muted:     #5A6A8A;
        --accent:    #2F80ED;
        --accent2:   #56CCF2;
        --success:   #27AE60;
        --danger:    #EB5757;
        --radius:    6px;
        --font-head: 'Barlow Condensed', sans-serif;
        --font-mono: 'IBM Plex Mono', monospace;
        --font-body: 'Space Grotesk', sans-serif;
    }

    html, body, [class*="css"] {
        font-family: var(--font-body);
        background-color: var(--bg) !important;
        color: var(--text);
    }
    .main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

    /*  Header  */
    .app-header { display: flex; align-items: baseline; gap: 0.9rem; margin-bottom: 0.2rem; }
    .app-title {
        font-family: var(--font-head); font-size: 2.2rem; font-weight: 700;
        letter-spacing: 0.02em; color: var(--text); margin: 0; text-transform: uppercase;
    }
    .app-title span { color: var(--accent2); }
    .app-badge {
        font-family: var(--font-mono); font-size: 0.6rem; font-weight: 500;
        letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent);
        border: 1px solid var(--accent); padding: 2px 7px; border-radius: 3px;
    }
    .app-subtitle {
        font-family: var(--font-mono); font-size: 0.68rem;
        color: var(--muted); letter-spacing: 0.06em; margin-bottom: 1.8rem;
    }

    /*  Sidebar  */
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    .sidebar-label {
        font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.16em;
        text-transform: uppercase; color: var(--muted); padding: 1.2rem 0 0.5rem;
    }
    [data-testid="stSidebar"] label {
        font-family: var(--font-mono) !important; font-size: 0.68rem !important;
        color: var(--muted) !important; letter-spacing: 0.03em;
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
        font-family: var(--font-mono) !important; font-size: 0.62rem !important;
        color: var(--accent2) !important;
    }

    /*  Tabs  */
    [data-testid="stTabs"] { border-bottom: 1px solid var(--border); }
    button[data-baseweb="tab"] {
        font-family: var(--font-mono) !important; font-size: 0.68rem !important;
        letter-spacing: 0.1em !important; text-transform: uppercase !important;
        color: var(--muted) !important; padding: 0.55rem 1.1rem !important;
        background: transparent !important; border: none !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent2) !important;
        border-bottom: 2px solid var(--accent2) !important;
    }

    /*  KPI grid  */
    .kpi-row {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 1px; background: var(--border);
        border: 1px solid var(--border); border-radius: var(--radius);
        overflow: hidden; margin-bottom: 1.5rem;
    }
    .kpi-card { background: var(--surface); padding: 1.3rem 1.5rem; }
    .kpi-label {
        font-family: var(--font-mono); font-size: 0.58rem;
        letter-spacing: 0.16em; text-transform: uppercase;
        color: var(--muted); margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-family: var(--font-head); font-size: 1.9rem; font-weight: 700;
        color: var(--text); letter-spacing: 0.01em; line-height: 1;
    }
    .kpi-sub { font-family: var(--font-mono); font-size: 0.6rem; color: var(--muted); margin-top: 0.3rem; }

    /*  Prediction card  */
    .pred-result {
        background: linear-gradient(135deg, #0E1828 0%, #0E1420 100%);
        border: 1px solid var(--border-hi); border-left: 3px solid var(--accent2);
        border-radius: var(--radius); padding: 1.6rem 2rem; margin: 0.5rem 0 1.2rem;
        position: relative; overflow: hidden;
    }
    .pred-result::before {
        content: ''; position: absolute; top: 0; right: 0;
        width: 180px; height: 180px;
        background: radial-gradient(circle at top right, rgba(47,128,237,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .pred-label {
        font-family: var(--font-mono); font-size: 0.58rem;
        letter-spacing: 0.18em; text-transform: uppercase;
        color: var(--muted); margin-bottom: 0.45rem;
    }
    .pred-value {
        font-family: var(--font-head); font-size: 3.2rem; font-weight: 700;
        color: var(--accent2); letter-spacing: 0.02em; line-height: 1;
    }
    .pred-sub { font-family: var(--font-mono); font-size: 0.66rem; color: var(--muted); margin-top: 0.6rem; }

    /*  Section headings  */
    .section-head {
        font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.18em;
        text-transform: uppercase; color: var(--accent); margin: 1.6rem 0 0.8rem;
        display: flex; align-items: center; gap: 0.6rem;
    }
    .section-head::after { content: ''; flex: 1; height: 1px; background: var(--border); }

    /*  Info pills  */
    .pill-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.6rem; }
    .info-pill {
        display: inline-flex; align-items: center; gap: 0.35rem;
        font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.07em;
        color: var(--muted); background: var(--surface2);
        border: 1px solid var(--border); border-radius: 3px; padding: 3px 9px;
    }
    .info-pill .dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }

    /*  Empty state  */
    .empty-state {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 3.5rem 2rem; text-align: center;
    }
    .empty-icon { font-size: 2.2rem; margin-bottom: 0.8rem; }
    .empty-title {
        font-family: var(--font-head); font-size: 1rem; font-weight: 600;
        color: var(--text); margin-bottom: 0.4rem; letter-spacing: 0.03em;
    }
    .empty-sub { font-family: var(--font-mono); font-size: 0.66rem; color: var(--muted); }

    /*  Buttons  */
    .stButton > button {
        font-family: var(--font-mono) !important; font-size: 0.68rem !important;
        letter-spacing: 0.12em !important; text-transform: uppercase !important;
        background: var(--accent) !important; color: #fff !important;
        border: none !important; border-radius: var(--radius) !important;
        padding: 0.6rem 1.4rem !important; font-weight: 500 !important;
        transition: background 0.15s ease !important;
    }
    .stButton > button:hover { background: #1a6fd4 !important; }

    /*  Sliders / selects  */
    [data-testid="stSlider"] > div > div > div { background: #1A2235 !important; }
    [data-testid="stSlider"] [data-testid="stThumbValue"] {
        background: #0E1420 !important;
        color: #D6E4F7 !important;
        border: 1px solid #243050 !important;
        border-radius: 3px !important;
        padding: 1px 4px !important;
    }
    [data-testid="stSelectbox"] > div > div {
        background-color: var(--surface2) !important; border-color: var(--border-hi) !important;
        font-family: var(--font-mono) !important; font-size: 0.74rem !important;
    }

    /*  st.metric  */
    [data-testid="stMetric"] {
        background: var(--surface2); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] p {
        font-family: var(--font-mono) !important; font-size: 0.6rem !important;
        letter-spacing: 0.12em !important; text-transform: uppercase !important; color: var(--muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--font-head) !important; font-size: 1.7rem !important;
        font-weight: 700 !important; color: var(--text) !important;
    }
    [data-testid="stMetricDelta"] { font-family: var(--font-mono) !important; font-size: 0.62rem !important; }

    /*  Misc  */
    [data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
    [data-testid="stExpander"] {
        background: var(--surface2) !important; border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stExpander"] summary {
        font-family: var(--font-mono) !important; font-size: 0.68rem !important;
        letter-spacing: 0.06em !important; color: var(--muted) !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] p {
        font-family: var(--font-mono) !important; font-size: 0.6rem !important; color: var(--muted) !important;
    }
    .stSpinner > div { border-top-color: var(--accent2) !important; }
    [data-testid="stFileUploader"] {
        border: 1px dashed var(--border-hi) !important;
        border-radius: var(--radius) !important; background: var(--surface2) !important;
    }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }
    .app-footer {
        margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
        font-family: var(--font-mono); font-size: 0.58rem; color: #2A3550;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#  Plotly theme
_COLORS = ["#2F80ED", "#56CCF2", "#27AE60", "#EB5757", "#9B51E0", "#F2994A", "#6FCF97"]
_BASE_LAYOUT: dict[str, Any] = dict(
    paper_bgcolor="#0E1420",
    plot_bgcolor="#080C12",
    font=dict(family="IBM Plex Mono, monospace", color="#5A6A8A", size=10),
    xaxis=dict(gridcolor="#1A2235", zerolinecolor="#1A2235", tickfont=dict(size=9)),
    yaxis=dict(gridcolor="#1A2235", zerolinecolor="#1A2235", tickfont=dict(size=9)),
    margin=dict(l=12, r=12, t=36, b=12),
    title_font=dict(family="Barlow Condensed, sans-serif", size=14, color="#D6E4F7"),
    colorway=_COLORS,
)


def _themed(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(**{**_BASE_LAYOUT, "height": height})
    return fig


#  Dataclass ─
@dataclass(frozen=True)
class ArtifactPaths:
    model_path: Path
    data_path: Path
    study_path: Path | None


#  Artifact inference
def infer_paths(base_dir: Path) -> ArtifactPaths:
    pkl_files = sorted(base_dir.glob("*.pkl"))
    model_candidates = [p for p in pkl_files if "ensemble" in p.stem.lower()]
    compat_candidates = [p for p in model_candidates if "compat" in p.stem.lower()]
    primary_candidates = [p for p in model_candidates if p not in compat_candidates]
    model_path = (
        primary_candidates or compat_candidates or model_candidates or pkl_files
    )[0]
    csv_files = sorted((base_dir / "mydata").glob("*.csv"))
    clean_candidates = [p for p in csv_files if "clean" in p.stem.lower()]
    data_path = (clean_candidates or csv_files)[0]
    study_candidates = [p for p in pkl_files if "stud" in p.stem.lower()]
    return ArtifactPaths(
        model_path=model_path,
        data_path=data_path,
        study_path=study_candidates[0] if study_candidates else None,
    )


#  Pipeline introspection
def _final_estimator(pipe: Any) -> Any:
    """Last step of a Pipeline, or the object itself if not a Pipeline."""
    return pipe.steps[-1][1] if isinstance(pipe, Pipeline) else pipe


def _pre_steps(pipe: Any) -> list[Any]:
    """All steps before the last in a Pipeline."""
    if isinstance(pipe, Pipeline) and len(pipe.steps) > 1:
        return [t for _, t in pipe.steps[:-1]]
    return []


def _transform_chain(steps: list[Any], X: Any) -> Any:
    for step in steps:
        X = step.transform(X)
    return X


def _to_dense(X: Any) -> Any:
    return X.toarray() if hasattr(X, "toarray") else X


def _extract_required_features(model: Any) -> list[str]:
    engineered = {
        "bed_bath_ratio",
        "total_rooms",
        "size_category",
        "broker_listing_count",
        "zip_listing_count",
    }

    def _clean(cols: list[str]) -> list[str]:
        out: list[str] = []
        for c in cols:
            name = str(c)
            low = name.lower()
            if low.startswith("unnamed:"):
                continue
            if name in engineered or name.startswith("cluster_sim_"):
                continue
            if name not in out:
                out.append(name)
        return out

    def _from_pipeline(pipe: Pipeline) -> list[str]:
        if hasattr(pipe, "feature_names_in_"):
            cleaned = _clean([str(c) for c in pipe.feature_names_in_])
            if cleaned:
                return cleaned
        pre = pipe.named_steps.get("pre") if hasattr(pipe, "named_steps") else None
        if pre is not None and hasattr(pre, "feature_names_in_"):
            cleaned = _clean([str(c) for c in pre.feature_names_in_])
            if cleaned:
                return cleaned
        for _, step in pipe.steps:
            if hasattr(step, "feature_names_in_"):
                cleaned = _clean([str(c) for c in step.feature_names_in_])
                if cleaned:
                    return cleaned
        return []

    if hasattr(model, "feature_names_in_"):
        cleaned = _clean([str(c) for c in model.feature_names_in_])
        if cleaned:
            return cleaned
    if isinstance(model, Pipeline) and hasattr(model, "feature_names_in_"):
        cleaned = _clean([str(c) for c in model.feature_names_in_])
        if cleaned:
            return cleaned
    if isinstance(model, Pipeline):
        cleaned = _from_pipeline(model)
        if cleaned:
            return cleaned

    estimators_fitted = list(getattr(model, "estimators_", []) or [])
    estimators_named = [e for _, e in getattr(model, "estimators", [])]
    for est in estimators_fitted + estimators_named:
        if hasattr(est, "feature_names_in_"):
            cleaned = _clean([str(c) for c in est.feature_names_in_])
            if cleaned:
                return cleaned
        if isinstance(est, Pipeline):
            cleaned = _from_pipeline(est)
            if cleaned:
                return cleaned
    return []


def _extract_preprocessor(model: Any) -> Any | None:
    if isinstance(model, Pipeline):
        if hasattr(model, "named_steps") and "pre" in model.named_steps:
            return model.named_steps["pre"]
        steps = _pre_steps(model)
        return steps[0] if steps else None
    for _, est in getattr(model, "estimators", []):
        if isinstance(est, Pipeline):
            if hasattr(est, "named_steps") and "pre" in est.named_steps:
                return est.named_steps["pre"]
            steps = _pre_steps(est)
            if steps:
                return steps[0]
    return None


#  Loaders ─
@st.cache_resource
def load_model_bundle(
    base_dir: Path,
) -> tuple[Any, Any | None, list[str], ArtifactPaths]:
    try:
        paths = infer_paths(base_dir)
    except Exception as exc:
        st.error(f"Failed to infer artifact paths: {exc}")
        raise
    try:
        __main__.FeatureEngineer = FeatureEngineer
        __main__.SanitiseFeatureNames = SanitiseFeatureNames
        try:
            model = joblib.load(paths.model_path)
        except Exception:
            compat_fallback = paths.model_path.with_name("ny_house_ensemble_compat.pkl")
            if compat_fallback.exists() and compat_fallback != paths.model_path:
                model = joblib.load(compat_fallback)
                paths = ArtifactPaths(
                    model_path=compat_fallback,
                    data_path=paths.data_path,
                    study_path=paths.study_path,
                )
            else:
                raise
        return (
            model,
            _extract_preprocessor(model),
            _extract_required_features(model),
            paths,
        )
    except Exception as exc:
        st.error(f"Failed to load model from {paths.model_path}: {exc}")
        raise


@st.cache_data
def load_training_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
        return df.drop(columns=unnamed) if unnamed else df
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        raise


@st.cache_data
def build_feature_specs(
    df: pd.DataFrame, required_features: list[str]
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for col in required_features:
        if col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if clean.empty:
                continue
            q01, q99 = clean.quantile([0.01, 0.99]).tolist()
            if q01 == q99:
                q01, q99 = float(clean.min()), float(clean.max())
            rounded = np.isclose(clean, np.round(clean)).mean() >= 0.98
            count_like = any(t in col.lower() for t in ("bed", "bath", "room", "count"))
            specs[col] = {
                "kind": "numeric",
                "min": float(q01),
                "max": float(q99),
                "default": float(clean.median()),
                "integer": bool(rounded or count_like),
            }
        else:
            choices = sorted([str(v) for v in s.dropna().unique()])
            specs[col] = {"kind": "categorical", "choices": choices or [""]}
    return specs


#  Column inference
def infer_target_column(df: pd.DataFrame, required_features: list[str]) -> str:
    for col in df.columns:
        if str(col).strip().lower() == "price":
            return col
    remaining = [c for c in df.columns if c not in required_features]
    numeric = [c for c in remaining if pd.api.types.is_numeric_dtype(df[c])]
    return (numeric or remaining or [df.columns[-1]])[0]


def infer_sqft_feature(features: list[str]) -> str | None:
    hits = [c for c in features if "sqft" in c.lower() or "square" in c.lower()]
    return hits[0] if hits else None


def infer_location_feature(df: pd.DataFrame) -> str | None:
    obj_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    preferred = [
        c
        for c in obj_cols
        if any(k in c.lower() for k in ("borough", "local", "zip", "state"))
    ]
    return (preferred or obj_cols or [None])[0]


def infer_color_feature(df: pd.DataFrame) -> str | None:
    obj_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return obj_cols[0] if obj_cols else None


def infer_borough_feature(features: list[str]) -> str | None:
    for c in features:
        if "borough" in c.lower():
            return c
    return None


def infer_zip_feature(features: list[str]) -> str | None:
    for c in features:
        if "zip" in c.lower():
            return c
    return None


def infer_lat_feature(features: list[str]) -> str | None:
    for c in features:
        if "lat" in c.lower():
            return c
    return None


def infer_lon_feature(features: list[str]) -> str | None:
    for c in features:
        if "lon" in c.lower() or "lng" in c.lower():
            return c
    return None


def infer_sublocality_feature(features: list[str]) -> str | None:
    for c in features:
        if "sublocal" in c.lower():
            return c
    return None


def infer_locality_feature(features: list[str]) -> str | None:
    for c in features:
        if "locality" in c.lower() and "sublocal" not in c.lower():
            return c
    return None


@st.cache_data
def build_borough_constraints(
    df: pd.DataFrame,
    borough_col: str | None,
    zip_col: str | None,
    lat_col: str | None,
    lon_col: str | None,
    categorical_cols: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if borough_col is None or borough_col not in df.columns:
        return {}
    constraints: dict[str, dict[str, Any]] = {}
    for borough in sorted(df[borough_col].dropna().astype(str).unique()):
        sub = df[df[borough_col].astype(str) == borough]
        item: dict[str, Any] = {}
        if zip_col and zip_col in sub.columns:
            item["zip_choices"] = sorted(
                sub[zip_col].dropna().astype(str).unique().tolist()
            )
        for col in categorical_cols or []:
            if col and col in sub.columns:
                item[f"{col}_choices"] = sorted(
                    sub[col].dropna().astype(str).unique().tolist()
                )
        if (
            lat_col
            and lat_col in sub.columns
            and pd.api.types.is_numeric_dtype(sub[lat_col])
        ):
            lat = pd.to_numeric(sub[lat_col], errors="coerce").dropna()
            if not lat.empty:
                item["lat_min"], item["lat_max"] = [
                    float(v) for v in lat.quantile([0.01, 0.99])
                ]
                item["lat_default"] = float(lat.median())
        if (
            lon_col
            and lon_col in sub.columns
            and pd.api.types.is_numeric_dtype(sub[lon_col])
        ):
            lon = pd.to_numeric(sub[lon_col], errors="coerce").dropna()
            if not lon.empty:
                item["lon_min"], item["lon_max"] = [
                    float(v) for v in lon.quantile([0.01, 0.99])
                ]
                item["lon_default"] = float(lon.median())
        constraints[borough] = item
    return constraints


def sanitize_feature_names(columns: list[str]) -> list[str]:
    return [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in columns]


#  Feature importance
def extract_feature_importance(
    model: Any, train_df: pd.DataFrame, required_features: list[str], target_col: str
) -> pd.DataFrame:
    try:
        sample = train_df[required_features + [target_col]].dropna().head(800)
        if not sample.empty:
            X = sample[required_features]
            y = sample[target_col]
            perm = permutation_importance(
                model,
                X,
                y,
                n_repeats=5,
                random_state=42,
                scoring="neg_mean_absolute_percentage_error",
            )
            return (
                pd.DataFrame(
                    {
                        "feature": required_features,
                        "importance": np.abs(perm.importances_mean),
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
    except Exception:
        pass
    return pd.DataFrame(columns=["feature", "importance"])


#  Metrics helpers ─
def load_study_best_params(study_path: Path | None) -> dict[str, Any]:
    if study_path is None:
        return {}
    try:
        studies = joblib.load(study_path)
        if not isinstance(studies, dict):
            return {}
        return {
            str(k): v.best_params
            for k, v in studies.items()
            if getattr(v, "best_params", None)
        }
    except Exception:
        return {}


def load_metrics_from_files(base_dir: Path) -> dict[str, float] | None:
    for path in sorted(base_dir.glob("**/*metrics*.json")) + sorted(
        base_dir.glob("**/*metrics*.csv")
    ):
        try:
            if path.suffix == ".json":
                payload = json.loads(path.read_text())
                if {"mape", "r2"}.issubset(payload):
                    return {k: float(payload[k]) for k in ("mape", "r2")}
            else:
                df = pd.read_csv(path)
                lm = {c.lower(): c for c in df.columns}
                if {"mape", "r2"}.issubset(lm):
                    row = df.iloc[0]
                    return {k: float(row[lm[k]]) for k in ("mape", "r2")}
        except Exception:
            continue
    return None


def compute_metrics(
    model: Any, df: pd.DataFrame, required_features: list[str], target_col: str
) -> dict[str, float]:
    X, y = df[required_features].copy(), df[target_col].copy()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = model.predict(X_test)
    return {
        "mape": float(mean_absolute_percentage_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


#  Prediction
def predict(inputs: dict[str, Any], model: Any, required_features: list[str]) -> float:
    row = pd.DataFrame([inputs]).reindex(columns=required_features)
    return float(model.predict(row)[0])


#  SHAP
def build_shap_figure(
    model: Any,
    train_df: pd.DataFrame,
    required_features: list[str],
    input_row: pd.DataFrame,
) -> plt.Figure | None:
    """
    Extracts the tree regressor from the first VotingRegressor member pipeline,
    transforms background + query row through all preceding steps, then runs
    TreeExplainer directly on the raw estimator. Works for LightGBM, XGBoost,
    CatBoost, and HistGradientBoosting pipelines regardless of step naming.
    """
    pipe = model if isinstance(model, Pipeline) else None
    if pipe is None:
        estimators_named = getattr(model, "estimators", [])
        estimators_fitted = list(getattr(model, "estimators_", []) or [])
        if estimators_named:
            pipe = estimators_named[0][1]
        elif estimators_fitted:
            pipe = estimators_fitted[0]
    if pipe is None or not isinstance(pipe, Pipeline):
        return None

    reg = _final_estimator(pipe)
    if hasattr(reg, "regressor_"):
        reg = reg.regressor_
    elif hasattr(reg, "regressor"):
        reg = reg.regressor
    pre = _pre_steps(pipe)

    try:
        bg_raw = train_df[required_features].head(min(200, len(train_df)))
        bg_x = _to_dense(_transform_chain(pre, bg_raw))
        row_x = _to_dense(_transform_chain(pre, input_row))
        n_cols = int(np.asarray(bg_x).shape[1])
        feat_names: list[str]
        if hasattr(reg, "feature_names_in_") and len(reg.feature_names_in_) == n_cols:
            feat_names = [str(n) for n in reg.feature_names_in_]
        else:
            names_from_steps: list[str] = []
            for step in reversed(pre):
                if hasattr(step, "get_feature_names_out"):
                    try:
                        names_from_steps = [
                            str(n) for n in step.get_feature_names_out()
                        ]
                        break
                    except Exception:
                        continue
            if names_from_steps and len(names_from_steps) == n_cols:
                feat_names = sanitize_feature_names(names_from_steps)
            else:
                feat_names = [f"f{i}" for i in range(n_cols)]
        bg_df = pd.DataFrame(bg_x, columns=feat_names)
        row_df = pd.DataFrame(row_x, columns=feat_names)

        if hasattr(reg, "feature_importances_"):
            try:
                explainer = shap.TreeExplainer(reg, data=bg_df)
                shap_values = explainer(row_df)
            except Exception:
                explainer = shap.Explainer(reg.predict, bg_df, algorithm="permutation")
                max_evals = max(2 * int(row_df.shape[1]) + 1, 1024)
                shap_values = explainer(row_df, max_evals=max_evals)
        else:
            explainer = shap.Explainer(reg.predict, bg_df, algorithm="permutation")
            max_evals = max(2 * int(row_df.shape[1]) + 1, 1024)
            shap_values = explainer(row_df, max_evals=max_evals)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0E1420")
        ax.set_facecolor("#080C12")
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)

        fig.patch.set_facecolor("#0E1420")
        for ax_ in fig.axes:
            ax_.set_facecolor("#080C12")
            ax_.tick_params(colors="#5A6A8A", labelsize=8)
            ax_.xaxis.label.set_color("#5A6A8A")
            for spine in ax_.spines.values():
                spine.set_edgecolor("#1A2235")

        plt.tight_layout()
        return fig

    except Exception as exc:
        st.warning(f"SHAP computation failed: {exc}")
        return None


#  Formatting
def fmt_usd(value: float) -> str:
    return f"${value:,.0f}"


#  Sidebar ─
def render_sidebar(
    required_features: list[str],
    specs: dict[str, dict[str, Any]],
    train_df: pd.DataFrame,
) -> tuple[dict[str, Any], bool]:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:1rem 0 0.2rem">
                <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.15rem;
                             font-weight:700;color:#D6E4F7;letter-spacing:0.04em;
                             text-transform:uppercase;">Property Features</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        inputs: dict[str, Any] = {}
        numeric_feats = [
            f for f in required_features if specs.get(f, {}).get("kind") == "numeric"
        ]
        cat_feats = [
            f
            for f in required_features
            if specs.get(f, {}).get("kind") == "categorical"
        ]
        other_feats = [f for f in required_features if f not in specs]
        borough_feat = infer_borough_feature(required_features)
        zip_feat = infer_zip_feature(required_features)
        lat_feat = infer_lat_feature(required_features)
        lon_feat = infer_lon_feature(required_features)
        sublocality_feat = infer_sublocality_feature(required_features)
        locality_feat = infer_locality_feature(required_features)
        constrained_cats = [c for c in [zip_feat, sublocality_feat, locality_feat] if c]
        constraints = build_borough_constraints(
            train_df, borough_feat, zip_feat, lat_feat, lon_feat, constrained_cats
        )
        selected_borough: str | None = None

        if borough_feat and borough_feat in cat_feats:
            st.markdown('<p class="sidebar-label">Location</p>', unsafe_allow_html=True)
            borough_choices = specs[borough_feat]["choices"]
            selected_borough = st.selectbox(
                borough_feat.replace("_", " ").title(),
                options=borough_choices,
                key=f"sl_{borough_feat}",
            )
            inputs[borough_feat] = selected_borough

        if numeric_feats:
            st.markdown('<p class="sidebar-label">Numeric</p>', unsafe_allow_html=True)
            for feat in numeric_feats:
                s = specs[feat]
                min_v, max_v, default_v = (
                    float(s["min"]),
                    float(s["max"]),
                    float(s["default"]),
                )
                if selected_borough and feat == lat_feat:
                    c = constraints.get(selected_borough, {})
                    min_v = float(c.get("lat_min", min_v))
                    max_v = float(c.get("lat_max", max_v))
                    default_v = float(c.get("lat_default", default_v))
                if selected_borough and feat == lon_feat:
                    c = constraints.get(selected_borough, {})
                    min_v = float(c.get("lon_min", min_v))
                    max_v = float(c.get("lon_max", max_v))
                    default_v = float(c.get("lon_default", default_v))
                if s.get("integer"):
                    min_i, max_i = int(round(min_v)), int(round(max_v))
                    if min_i == max_i:
                        max_i = min_i + 1
                    default_i = int(round(default_v))
                    default_i = min(max(default_i, min_i), max_i)
                    inputs[feat] = st.slider(
                        feat.replace("_", " ").title(),
                        min_value=min_i,
                        max_value=max_i,
                        value=default_i,
                        step=1,
                    )
                else:
                    inputs[feat] = st.slider(
                        feat.replace("_", " ").title(),
                        min_value=min_v,
                        max_value=max_v,
                        value=min(max(default_v, min_v), max_v),
                    )
        if cat_feats:
            st.markdown(
                '<p class="sidebar-label">Categorical</p>', unsafe_allow_html=True
            )
            for feat in cat_feats:
                if feat == borough_feat:
                    continue
                options = specs[feat]["choices"]
                if selected_borough and feat in constrained_cats:
                    b_opts = constraints.get(selected_borough, {}).get(
                        f"{feat}_choices", []
                    )
                    filtered = [z for z in options if z in set(b_opts)]
                    options = filtered or options
                    inputs[feat] = st.selectbox(
                        feat.replace("_", " ").title(),
                        options=options,
                        key=f"sl_{feat}_{selected_borough}",
                    )
                else:
                    inputs[feat] = st.selectbox(
                        feat.replace("_", " ").title(), options=options
                    )
        if other_feats:
            st.markdown('<p class="sidebar-label">Other</p>', unsafe_allow_html=True)
            for feat in other_feats:
                inputs[feat] = st.text_input(feat.replace("_", " ").title(), value="")

        st.markdown("<div style='height:1rem'/>", unsafe_allow_html=True)
        run = st.button("Run Prediction", use_container_width=True)

    return inputs, run


#  Tab: Predict
def show_predict_tab(
    model: Any,
    train_df: pd.DataFrame,
    required_features: list[str],
    target_col: str,
    inputs: dict[str, Any],
    should_predict: bool,
) -> None:
    if not should_predict:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">🏙️</div>
                <div class="empty-title">Configure inputs and run prediction</div>
                <div class="empty-sub">Adjust property features in the sidebar, then click Run Prediction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Running model inference..."):
        y_pred = predict(inputs, model, required_features)

    sqft_feat = infer_sqft_feature(required_features)
    sqft_val = float(inputs.get(sqft_feat, 0)) if sqft_feat else 0.0
    ppsf = y_pred / sqft_val if sqft_val > 0 else None
    borough_feat = infer_borough_feature(required_features)
    baseline_label = "dataset median"
    median = float(train_df[target_col].median())
    comparison_series = train_df[target_col].dropna()
    if borough_feat and borough_feat in train_df.columns:
        selected_borough = str(inputs.get(borough_feat, "")).strip()
        if selected_borough:
            borough_slice = train_df[
                train_df[borough_feat].astype(str) == selected_borough
            ]
            if not borough_slice.empty:
                median = float(borough_slice[target_col].median())
                comparison_series = borough_slice[target_col].dropna()
                baseline_label = f"{selected_borough} median"
    delta_pct = (y_pred - median) / median * 100
    delta_abs = y_pred - median
    percentile = (
        float((comparison_series <= y_pred).mean() * 100)
        if not comparison_series.empty
        else np.nan
    )
    q25 = (
        float(comparison_series.quantile(0.25))
        if not comparison_series.empty
        else np.nan
    )
    q75 = (
        float(comparison_series.quantile(0.75))
        if not comparison_series.empty
        else np.nan
    )
    iqr = q75 - q25 if np.isfinite(q75) and np.isfinite(q25) else np.nan
    iqr_units = (y_pred - median) / iqr if iqr and np.isfinite(iqr) else np.nan
    d_color = "#27AE60" if delta_pct >= 0 else "#EB5757"
    d_arrow = "▲" if delta_pct >= 0 else "▼"

    st.markdown(
        f"""
        <div class="pred-result">
            <div class="pred-label">Predicted price</div>
            <div class="pred-value">{fmt_usd(y_pred)}</div>
            <div class="pred-sub">
                <span style="color:{d_color};">{d_arrow} {abs(delta_pct):.1f}% vs {baseline_label}</span>
                &nbsp;·&nbsp; Median: {fmt_usd(median)}
                {"&nbsp;·&nbsp; " + fmt_usd(ppsf) + " / sqft" if ppsf else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Delta vs baseline", fmt_usd(delta_abs))
    m2.metric(
        "Price percentile", f"{percentile:.1f}%" if np.isfinite(percentile) else "N/A"
    )
    m3.metric(
        "IQR units from median", f"{iqr_units:.2f}" if np.isfinite(iqr_units) else "N/A"
    )

    # Gauge
    gauge_max = max(median * 2.0, y_pred * 1.1, 1.0)
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=y_pred,
            number={
                "prefix": "$",
                "valueformat": ",.0f",
                "font": {
                    "family": "Barlow Condensed, sans-serif",
                    "size": 30,
                    "color": "#56CCF2",
                },
            },
            delta={
                "reference": median,
                "relative": True,
                "font": {"family": "IBM Plex Mono, monospace", "size": 10},
                "increasing": {"color": "#27AE60"},
                "decreasing": {"color": "#EB5757"},
            },
            title={
                "text": f"vs {baseline_label}",
                "font": {
                    "family": "IBM Plex Mono, monospace",
                    "size": 9,
                    "color": "#5A6A8A",
                },
            },
            gauge={
                "axis": {
                    "range": [0, gauge_max],
                    "tickfont": {
                        "size": 8,
                        "family": "IBM Plex Mono, monospace",
                        "color": "#5A6A8A",
                    },
                    "tickcolor": "#1A2235",
                },
                "bar": {"color": "#56CCF2", "thickness": 0.22},
                "bgcolor": "#080C12",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, median * 0.7], "color": "#0C1018"},
                    {"range": [median * 0.7, median * 1.3], "color": "#0E1828"},
                    {"range": [median * 1.3, gauge_max], "color": "#0C1018"},
                ],
                "threshold": {
                    "line": {"color": "#2F80ED", "width": 2},
                    "thickness": 0.8,
                    "value": median,
                },
            },
        )
    )
    fig_gauge.update_layout(
        paper_bgcolor="#0E1420",
        plot_bgcolor="#0E1420",
        margin=dict(l=20, r=20, t=30, b=20),
        height=220,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # SHAP
    st.markdown(
        '<div class="section-head">Feature Attribution (SHAP)</div>',
        unsafe_allow_html=True,
    )
    input_row = pd.DataFrame([inputs]).reindex(columns=required_features)
    shap_fig = build_shap_figure(model, train_df, required_features, input_row)
    if shap_fig is not None:
        st.pyplot(shap_fig, clear_figure=True)
    else:
        st.info(
            "SHAP waterfall unavailable — pipeline transformation could not be resolved."
        )


#  Tab: Batch
def show_batch_tab(model: Any, required_features: list[str]) -> None:
    st.markdown(
        '<div class="section-head">Batch CSV Prediction</div>', unsafe_allow_html=True
    )

    col_up, col_schema = st.columns([3, 2])
    with col_up:
        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"], label_visibility="collapsed"
        )
    with col_schema:
        with st.expander(f"Required columns ({len(required_features)})"):
            st.code(", ".join(required_features), language="text")

    if uploaded is None:
        st.markdown(
            """<div class="empty-state" style="padding:2rem">
                <div class="empty-sub">Drop a CSV with the required columns to get batch predictions</div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Cannot read CSV: {exc}")
        return

    missing = [c for c in required_features if c not in batch_df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    pred_df = batch_df.copy()
    pred_df["PREDICTED_PRICE"] = model.predict(pred_df[required_features])
    prices = pred_df["PREDICTED_PRICE"]

    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Rows Predicted</div>
                <div class="kpi-value">{len(pred_df):,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Median Price</div>
                <div class="kpi-value" style="color:#56CCF2">{fmt_usd(prices.median())}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Range</div>
                <div class="kpi-value" style="font-size:1.2rem;color:#D6E4F7">
                    {fmt_usd(prices.min())} – {fmt_usd(prices.max())}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(pred_df, use_container_width=True)
    st.download_button(
        "↓  Download predictions CSV",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name="batch_predictions.csv",
        mime="text/csv",
    )

    st.markdown(
        '<div class="section-head">Price Distribution</div>', unsafe_allow_html=True
    )
    hist = go.Figure(
        go.Histogram(x=prices, nbinsx=40, marker_color="#2F80ED", marker_line_width=0)
    )
    st.plotly_chart(_themed(hist, height=260), use_container_width=True)


#  Tab: Model Info ─
def show_model_info_tab(
    base_dir: Path,
    model: Any,
    train_df: pd.DataFrame,
    required_features: list[str],
    target_col: str,
    study_path: Path | None,
) -> None:
    st.markdown(
        '<div class="section-head">Global Model Performance</div>',
        unsafe_allow_html=True,
    )
    metrics = load_metrics_from_files(base_dir)
    note = "Loaded from metrics file."
    if metrics is None:
        metrics = compute_metrics(model, train_df, required_features, target_col)
        note = "Computed on 20% held-out split from training data."
    c1, c2 = st.columns(2)
    c1.metric("MAPE", f"{metrics['mape']:.4f}")
    c2.metric("R²", f"{metrics['r2']:.4f}")
    st.caption(f"{note} (global validation, not per-listing error)")

    st.markdown(
        '<div class="section-head">Ensemble Members</div>', unsafe_allow_html=True
    )
    algo_rows: list[dict[str, Any]] = []
    if isinstance(model, Pipeline):
        reg = _final_estimator(model)
        if hasattr(reg, "regressor_"):
            reg = reg.regressor_
        elif hasattr(reg, "regressor"):
            reg = reg.regressor
        algo_rows.append(
            {
                "name": "Model",
                "algorithm": type(reg).__name__,
                "params": reg.get_params() if hasattr(reg, "get_params") else {},
            }
        )
    else:
        members = list(getattr(model, "estimators", []))
        if not members and getattr(model, "estimators_", None):
            members = [
                (f"Member {i+1}", est) for i, est in enumerate(model.estimators_)
            ]
        for name, est in members:
            reg = _final_estimator(est) if isinstance(est, Pipeline) else est
            if hasattr(reg, "regressor_"):
                reg = reg.regressor_
            elif hasattr(reg, "regressor"):
                reg = reg.regressor
            algo_rows.append(
                {
                    "name": name,
                    "algorithm": type(reg).__name__,
                    "params": reg.get_params() if hasattr(reg, "get_params") else {},
                }
            )
    st.dataframe(
        pd.DataFrame(
            [{"Member": r["name"], "Algorithm": r["algorithm"]} for r in algo_rows]
        ),
        use_container_width=True,
    )
    with st.expander("Hyperparameters"):
        for row in algo_rows:
            st.markdown(f"**{row['name']} — {row['algorithm']}**")
            st.json(row["params"])

    st.markdown(
        '<div class="section-head">Feature Importances</div>',
        unsafe_allow_html=True,
    )
    fi_df = extract_feature_importance(model, train_df, required_features, target_col)
    if not fi_df.empty:
        top = fi_df.head(25).sort_values("importance", ascending=True)
        fig_imp = go.Figure(
            go.Bar(
                x=top["importance"],
                y=top["feature"],
                orientation="h",
                marker_color="#2F80ED",
                marker_line_width=0,
            )
        )
        st.plotly_chart(
            _themed(fig_imp, height=max(300, len(top) * 22)), use_container_width=True
        )
    else:
        st.info("Feature importances not available.")

    best_params = load_study_best_params(study_path)
    if best_params:
        st.markdown(
            '<div class="section-head">Optuna Best Params</div>', unsafe_allow_html=True
        )
        st.json(best_params)


#  Tab: Data Explorer
def show_data_explorer_tab(
    df: pd.DataFrame, target_col: str, required_features: list[str]
) -> None:
    missing_pct = df.isna().mean().mean() * 100
    miss_color = "#EB5757" if missing_pct > 5 else "#27AE60"

    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Rows</div>
                <div class="kpi-value">{df.shape[0]:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Columns</div>
                <div class="kpi-value" style="color:#56CCF2">{df.shape[1]}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Missing Values</div>
                <div class="kpi-value" style="color:{miss_color}">{missing_pct:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-head">Column Schema</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.astype(str).values}),
        use_container_width=True,
        height=220,
    )

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        st.markdown(
            '<div class="section-head">Missing Values by Column</div>',
            unsafe_allow_html=True,
        )
        fig_m = go.Figure(
            go.Bar(
                x=missing.index,
                y=missing.values,
                marker_color="#EB5757",
                marker_line_width=0,
            )
        )
        st.plotly_chart(_themed(fig_m, height=240), use_container_width=True)

    sqft_feat = infer_sqft_feature(required_features)
    color_feat = infer_color_feature(df)
    if sqft_feat and sqft_feat in df.columns:
        st.markdown(
            f'<div class="section-head">{sqft_feat.replace("_"," ").title()} vs {target_col}</div>',
            unsafe_allow_html=True,
        )
        fig_sc = px.scatter(
            df,
            x=sqft_feat,
            y=target_col,
            color=color_feat if color_feat in df.columns else None,
            opacity=0.5,
            color_discrete_sequence=_COLORS,
        )
        fig_sc.update_traces(marker_size=3)
        st.plotly_chart(_themed(fig_sc, height=340), use_container_width=True)

    loc_col = infer_location_feature(df)
    if loc_col and loc_col in df.columns:
        st.markdown(
            f'<div class="section-head">{target_col} by {loc_col.replace("_"," ").title()}</div>',
            unsafe_allow_html=True,
        )
        top_locs = df[loc_col].astype(str).value_counts().head(10).index.tolist()
        box_df = df[df[loc_col].astype(str).isin(top_locs)]
        fig_box = px.box(
            box_df,
            x=loc_col,
            y=target_col,
            color=loc_col,
            color_discrete_sequence=_COLORS,
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(_themed(fig_box, height=340), use_container_width=True)

    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        st.markdown(
            '<div class="section-head">Correlation Heatmap</div>',
            unsafe_allow_html=True,
        )
        corr = num_df.corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            text_auto=False,
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale=[[0, "#EB5757"], [0.5, "#0E1420"], [1, "#2F80ED"]],
        )
        st.plotly_chart(_themed(fig_corr, height=480), use_container_width=True)


#  Main
def main() -> None:
    base_dir = Path(__file__).parent

    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">NY <span>House Price</span> Predictor</h1>
            <span class="app-badge">Ensemble · v2</span>
        </div>
        <div class="app-subtitle">
            VotingRegressor &nbsp;·&nbsp; LightGBM &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; CatBoost &nbsp;·&nbsp; HistGB
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, preprocessor, required_features, paths = load_model_bundle(base_dir)
    except Exception:
        return
    try:
        train_df = load_training_data(paths.data_path)
    except Exception:
        return

    if not required_features:
        st.error(
            "Could not infer the model's trained input features from the pickle artifact. "
            "This app will not guess from CSV columns."
        )
        return
    target_col = infer_target_column(train_df, required_features)
    specs = build_feature_specs(train_df, required_features)

    members = [name for name, _ in getattr(model, "estimators", [])]
    pre_type = type(preprocessor).__name__ if preprocessor else "None"
    st.markdown(
        f"""
        <div class="pill-row">
            <span class="info-pill"><span class="dot" style="background:#27AE60"></span>Model loaded</span>
            <span class="info-pill"><span class="dot" style="background:#2F80ED"></span>{len(required_features)} features</span>
            <span class="info-pill"><span class="dot" style="background:#56CCF2"></span>{len(members)} members: {", ".join(members)}</span>
            <span class="info-pill"><span class="dot" style="background:#9B51E0"></span>{pre_type}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    inputs, should_predict = render_sidebar(required_features, specs, train_df)

    tab_predict, tab_batch, tab_model, tab_explorer = st.tabs(
        ["  Predict  ", "  Batch  ", "  Model Info  ", "  Data Explorer  "]
    )
    with tab_predict:
        show_predict_tab(
            model, train_df, required_features, target_col, inputs, should_predict
        )
    with tab_batch:
        show_batch_tab(model, required_features)
    with tab_model:
        show_model_info_tab(
            base_dir, model, train_df, required_features, target_col, paths.study_path
        )
    with tab_explorer:
        show_data_explorer_tab(train_df, target_col, required_features)

    st.markdown(
        f'<div class="app-footer">{paths.model_path.name} &nbsp;·&nbsp; {paths.data_path.name}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
