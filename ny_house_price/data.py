from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd

PRICE_SENTINEL = 2_147_483_647
SQFT_SENTINEL = 65_535
BATH_CORRUPT_FLOAT = 2.3738608579684373
NON_SALE_TYPES = {"Pending", "Contingent", "Coming Soon", "Foreclosure"}

DROP_COLS = [
    "ADDRESS",
    "MAIN_ADDRESS",
    "STREET_NAME",
    "LONG_NAME",
    "FORMATTED_ADDRESS",
    "ADMINISTRATIVE_AREA_LEVEL_2",
]


def extract_borough(state_str: str) -> str:
    borough_map = {
        "brooklyn": "Brooklyn",
        "bronx": "Bronx",
        "staten island": "Staten Island",
        "queens": "Queens",
        "jackson heights": "Queens",
        "elmhurst": "Queens",
        "rego park": "Queens",
        "woodside": "Queens",
        "flushing": "Queens",
        "jamaica": "Queens",
        "astoria": "Queens",
        "manhattan": "Manhattan",
        "new york": "Manhattan",
    }
    city_part = str(state_str).split(",")[0].strip().lower()
    return borough_map.get(city_part, "Other")


def extract_zip(state_str: str) -> str:
    match = re.search(r"\b(\d{5})\b", str(state_str))
    return match.group(1) if match else ""


def clean_broker_name(raw: str) -> str:
    name = re.sub(r"^Brokered by\s+", "", str(raw), flags=re.IGNORECASE).strip()
    name = re.sub(r"\s+-\s*\d+.*$", "", name).strip()
    return re.sub(r"\s{2,}", " ", name)


FRANCHISE_PATTERNS: list[tuple[str, str]] = [
    (r"douglas elliman", "Douglas Elliman"),
    (r"sotheby", "Sotheby's International Realty"),
    (r"corcoran", "Corcoran"),
    (r"keller williams", "Keller Williams"),
    (r"re\s*/?\s*max|remax", "RE/MAX"),
    (r"\bexp realty\b|exprealty", "eXp Realty"),
    (r"compass", "COMPASS"),
    (r"serhant", "Serhant"),
    (r"coldwell banker", "Coldwell Banker"),
    (r"century 21", "Century 21"),
    (r"winzone", "Winzone Realty"),
    (r"brown harris stevens", "Brown Harris Stevens"),
    (r"nest seekers", "Nest Seekers International"),
    (r"e realty international", "E Realty International"),
    (r"robert defalco", "Robert DeFalco Realty"),
]

BROKER_MIN_LISTINGS = 10


def normalise_broker(name: str) -> str:
    lower = str(name).lower()
    for pattern, canonical in FRANCHISE_PATTERNS:
        if re.search(pattern, lower):
            return canonical
    return name


def bucket_broker_column(series: pd.Series, min_count: int = BROKER_MIN_LISTINGS) -> pd.Series:
    counts = series.value_counts()
    rare = counts[counts < min_count].index
    return series.where(~series.isin(rare), other="Other")


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    report: dict[str, int] = {"initial_rows": len(df)}

    before = len(df)
    df = df.drop_duplicates()
    report["dropped_duplicates"] = before - len(df)

    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    before = len(df)
    df = df[df["price"] != PRICE_SENTINEL]
    df = df[df["propertysqft"] != SQFT_SENTINEL]
    report["dropped_sentinels"] = before - len(df)

    before = len(df)
    df = df[~np.isclose(df["bath"], BATH_CORRUPT_FLOAT)]
    report["dropped_corrupt_bath"] = before - len(df)

    before = len(df)
    df = df[~df["type"].isin(NON_SALE_TYPES)]
    report["dropped_nontransactional_types"] = before - len(df)

    df["borough"] = df["state"].apply(extract_borough)
    df["zip_code"] = df["state"].apply(extract_zip)
    df = df.drop(columns=["state"])

    df["broker_name"] = (
        df["brokertitle"]
        .apply(clean_broker_name)
        .apply(normalise_broker)
        .pipe(bucket_broker_column)
        .astype("category")
    )
    df = df.drop(columns=["brokertitle"])

    type_order = [
        "Co-op for sale",
        "Condo for sale",
        "Condop for sale",
        "Townhouse for sale",
        "House for sale",
        "Multi-family home for sale",
        "Land for sale",
        "Mobile house for sale",
        "For sale",
    ]
    df["type"] = pd.Categorical(df["type"], categories=type_order, ordered=False)

    existing_drop = [c for c in [col.lower() for col in DROP_COLS] if c in df.columns]
    df = df.drop(columns=existing_drop)

    df["beds"] = df["beds"].astype("int16")
    df["bath"] = df["bath"].astype("float32")
    df["price"] = df["price"].astype("int64")
    df["propertysqft"] = df["propertysqft"].astype("float32")
    df["latitude"] = df["latitude"].astype("float32")
    df["longitude"] = df["longitude"].astype("float32")
    df["zip_code"] = df["zip_code"].astype(str)
    df["borough"] = df["borough"].astype("category")

    df = df.reset_index(drop=True)
    report["final_rows"] = len(df)
    report["total_dropped"] = report["initial_rows"] - report["final_rows"]
    return df, report


def print_cleaning_report(report: dict[str, int]) -> None:
    width = 52
    print("\n" + "=" * width)
    print("  NY HOUSE DATASET - CLEANING REPORT")
    print("=" * width)
    print(f"Initial rows: {report['initial_rows']:>6}")
    print(f"Dropped duplicates: {report['dropped_duplicates']:>6}")
    print(f"Dropped sentinel values: {report['dropped_sentinels']:>6}")
    print(f"Dropped corrupt BATH float: {report['dropped_corrupt_bath']:>6}")
    n_dropped = report["dropped_nontransactional_types"]
    print(f"Dropped non-transactional: {n_dropped:>6}")
    print("-" * width)
    print(f"Final rows: {report['final_rows']:>6}")
    print(
        f"Total removed: {report['total_dropped']:>6} "
        f"({report['total_dropped']/report['initial_rows']*100:.1f}%)"
    )
    print("=" * width + "\n")


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_clean_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
