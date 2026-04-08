from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
)


@dataclass
class FeatureEngineer(BaseEstimator, TransformerMixin):
    bins: np.ndarray = field(
        default_factory=lambda: np.logspace(np.log10(200), np.log10(65000), num=6)
    )
    labels: list[str] = field(
        default_factory=lambda: ["Tiny", "Small", "Medium", "Large", "Luxury"]
    )
    cluster_features: list[str] = field(
        default_factory=lambda: ["latitude", "longitude"]
    )
    n_clusters: int = 10
    gamma: float = 1.0
    random_state: int = 42

    def fit(self, X, y=None, **fit_params):
        X = X.copy()
        self.broker_stats_ = (
            X.groupby("broker_name")
            .size()
            .sub(1)
            .clip(lower=0)
            .to_frame("broker_listing_count")
        )
        self.zip_stats_ = (
            X.groupby("zip_code")
            .size()
            .sub(1)
            .clip(lower=0)
            .to_frame("zip_listing_count")
        )

        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X[self.cluster_features])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["bed_bath_ratio"] = X["beds"] / X["bath"].replace(0, np.nan)
        X["total_rooms"] = X["beds"] + X["bath"]
        X["size_category"] = pd.cut(
            X["propertysqft"],
            bins=self.bins,
            labels=self.labels,
            include_lowest=True,
            right=False,
        )
        X = X.merge(
            self.broker_stats_, left_on="broker_name", right_index=True, how="left"
        )
        X = X.merge(self.zip_stats_, left_on="zip_code", right_index=True, how="left")
        X["broker_listing_count"] = X["broker_listing_count"].fillna(0)
        X["zip_listing_count"] = X["zip_listing_count"].fillna(0)

        similarities = rbf_kernel(
            X[self.cluster_features], self.kmeans_.cluster_centers_, gamma=self.gamma
        )
        for i in range(similarities.shape[1]):
            X[f"cluster_sim_{i}"] = similarities[:, i]
        return X

    def get_feature_names_out(self, input_features=None):
        sim_cols = [f"cluster_sim_{i}" for i in range(self.n_clusters)]
        new_cols = [
            "bed_bath_ratio",
            "total_rooms",
            "size_category",
            "broker_listing_count",
            "zip_listing_count",
            *sim_cols,
        ]
        return list(input_features) + new_cols if input_features is not None else new_cols


class SanitiseFeatureNames(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in X.columns]
        return X


YEO_JOHNSON_FEATURES = ["propertysqft"]
NUMERIC_FEATURES = [
    "beds",
    "bath",
    "latitude",
    "longitude",
    "bed_bath_ratio",
    "total_rooms",
    "broker_listing_count",
    "zip_listing_count",
    *[f"cluster_sim_{i}" for i in range(10)],
]
ORDINAL_FEATURES = ["size_category"]
SIZE_CATEGORIES = [["Tiny", "Small", "Medium", "Large", "Luxury"]]
CATEGORICAL_FEATURES = ["borough", "broker_name", "type", "zip_code"]


PREPROCESSOR = ColumnTransformer(
    transformers=[
        (
            "yeo",
            make_pipeline(SimpleImputer(strategy="median"), PowerTransformer(method="yeo-johnson")),
            YEO_JOHNSON_FEATURES,
        ),
        (
            "num",
            make_pipeline(SimpleImputer(strategy="median"), RobustScaler()),
            NUMERIC_FEATURES,
        ),
        (
            "ord",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(categories=SIZE_CATEGORIES),
            ),
            ORDINAL_FEATURES,
        ),
        (
            "cat",
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value="Unknown"),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
).set_output(transform="pandas")


LOG_TRANSFORMER = FunctionTransformer(np.log1p, inverse_func=np.expm1)


def build_pipeline(regressor, target_transformer=None):
    if target_transformer is None:
        target_transformer = PowerTransformer(method="yeo-johnson")
    return Pipeline(
        [
            ("fe", FeatureEngineer()),
            ("pre", PREPROCESSOR),
            ("sanitise", SanitiseFeatureNames()),
            (
                "model",
                TransformedTargetRegressor(regressor=regressor, transformer=target_transformer),
            ),
        ]
    )
