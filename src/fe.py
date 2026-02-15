# src/fe.py
"""
Leak-safe Feature Engineering + Preprocessing for Bank Marketing (bank-full.csv).

Key fixes vs v1:
- NO fitting/transforming inside build_preprocessor() to "discover columns" (prevents leakage).
- Uses fixed, known column lists for bank-full.csv schema.
- Numeric pipeline uses StandardScaler() correctly.
- OneHotEncoder stays sparse; numeric block stays dense; ColumnTransformer outputs sparse matrix.
- GaussianNB compatibility handled in train.py (dense conversion).

Exports:
- load_bank_data(csv_path, sep=";") -> (X, y, df)
- build_preprocessor() -> sklearn Pipeline (FE -> ColumnTransformer)
- get_feature_names(preprocessor_pipeline) -> list[str]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# -----------------------------
# Bank schema (bank-full.csv)
# -----------------------------
TARGET_COL = "y"

NUMERIC_COLS_RAW = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_COLS_RAW = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

# Engineered columns (created by FeatureEngineeringTransformer)
NUMERIC_COLS_ENGINEERED = [
    "contacted_before",
    "campaign_intensity",
    "is_long_call",
]

CATEGORICAL_COLS_ENGINEERED = [
    "age_group",
]


# -----------------------------
# Data Loading
# -----------------------------
def load_bank_data(csv_path: str, sep: str = ";") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads bank-full.csv and returns:
      X: features dataframe (raw columns)
      y: target series (0/1)
      df: full dataframe including y (cleaned target)
    """
    df = pd.read_csv(csv_path, sep=sep)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found in dataset.")

    # strip whitespace in object columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # encode target
    df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0})
    if df[TARGET_COL].isna().any():
        bad = df.loc[df[TARGET_COL].isna(), TARGET_COL].unique()
        raise ValueError(f"Unexpected target values in '{TARGET_COL}': {bad}")

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    return X, y, df


# -----------------------------
# Feature Engineering Transformer
# -----------------------------
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Performs deterministic feature engineering (no target use).

    Steps:
    - Replace "unknown" in categorical columns with NaN (so imputer can handle)
    - Rare category grouping per categorical column (fit on train only)
    - Create engineered features:
        * age_group (bins)
        * contacted_before = (pdays != -1)
        * campaign_intensity = campaign / (previous + 1)
        * is_long_call = duration > median(duration) (median learned on train)
    - Apply log1p to skewed numeric columns: balance, duration, campaign

    NOTE: This transformer expects a pandas DataFrame and returns a pandas DataFrame.
    """

    def __init__(self, rare_thresh: float = 0.01):
        self.rare_thresh = rare_thresh
        self._duration_median_: float | None = None
        self._rare_maps_: dict[str, set[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # Handle unknowns
        for c in CATEGORICAL_COLS_RAW:
            if c in X.columns:
                X[c] = X[c].replace({"unknown": np.nan})

        # Duration median for long-call flag
        if "duration" in X.columns:
            dur = pd.to_numeric(X["duration"], errors="coerce")
            self._duration_median_ = float(np.nanmedian(dur.values))
        else:
            self._duration_median_ = None

        # Rare category maps (train-only)
        self._rare_maps_.clear()
        n = len(X)
        if n > 0:
            for c in CATEGORICAL_COLS_RAW + CATEGORICAL_COLS_ENGINEERED:
                if c in X.columns:
                    vc = X[c].value_counts(dropna=True)
                    rare = set(vc[vc / n < self.rare_thresh].index.astype(str).tolist())
                    self._rare_maps_[c] = rare

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Ensure expected raw cols exist (guard early for upload/test)
        missing_raw = [c for c in (NUMERIC_COLS_RAW + CATEGORICAL_COLS_RAW) if c not in X.columns]
        if missing_raw:
            raise ValueError(f"Missing required columns in input data: {missing_raw}")

        # Standardize object cols and replace unknown
        for c in CATEGORICAL_COLS_RAW:
            X[c] = X[c].astype(str).str.strip()
            X[c] = X[c].replace({"unknown": np.nan})

        # Numeric coercion
        for c in NUMERIC_COLS_RAW:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        # -----------------------------
        # Engineered features
        # -----------------------------
        # age_group
        age = X["age"]
        X["age_group"] = pd.cut(
            age,
            bins=[0, 30, 45, 60, np.inf],
            labels=["18_30", "31_45", "46_60", "60_plus"],
            include_lowest=True,
        ).astype("object")

        # contacted_before
        X["contacted_before"] = (X["pdays"] != -1).astype(int)

        # campaign_intensity
        X["campaign_intensity"] = X["campaign"] / (X["previous"].fillna(0) + 1)

        # is_long_call
        if self._duration_median_ is not None:
            X["is_long_call"] = (X["duration"] > self._duration_median_).astype(int)
        else:
            X["is_long_call"] = 0

        # -----------------------------
        # Rare category grouping (after adding engineered categorical)
        # -----------------------------
        for c, rare_set in self._rare_maps_.items():
            if c in X.columns and rare_set:
                # Only replace non-null rare categories
                X[c] = X[c].where(~X[c].isin(list(rare_set)), other="Other")

        # -----------------------------
        # Log transforms (safe)
        # -----------------------------
        for col in ["balance", "duration", "campaign"]:
            if col in X.columns:
                # values might be negative for balance; log1p requires >= -1
                # So we shift only if needed:
                v = X[col].astype(float)
                min_v = np.nanmin(v.values)
                if np.isfinite(min_v) and min_v < -1:
                    # shift to make >= -1
                    shift = (-1 - min_v) + 1.0
                    v = v + shift
                X[col] = np.log1p(v)

        return X


# -----------------------------
# Preprocessor builder (LEAK-SAFE)
# -----------------------------
def build_preprocessor() -> Pipeline:
    """
    Returns a Pipeline:
      FE -> ColumnTransformer(num pipeline + cat pipeline)

    IMPORTANT:
    - Does not fit on any data
    - Column lists are based on known bank-full.csv schema + engineered features
    """
    fe = FeatureEngineeringTransformer(rare_thresh=0.01)

    numeric_cols = NUMERIC_COLS_RAW + NUMERIC_COLS_ENGINEERED
    categorical_cols = CATEGORICAL_COLS_RAW + CATEGORICAL_COLS_ENGINEERED

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),  # correct for dense numeric block
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.3,  # encourage sparse output when OHE dominates
    )

    full = Pipeline(
        steps=[
            ("fe", fe),
            ("preprocess", ct),
        ]
    )
    return full


def get_feature_names(preprocessor_pipeline: Pipeline) -> list[str]:
    """
    Extracts output feature names after preprocessing (after one-hot).
    """
    if not hasattr(preprocessor_pipeline, "named_steps"):
        return []

    if "preprocess" not in preprocessor_pipeline.named_steps:
        return []

    ct = preprocessor_pipeline.named_steps["preprocess"]
    try:
        names = ct.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return []
