"""Possum total-length regression (from ``linearRegression.ipynb``).

Predicts a possum's total body length (``totlngth``, cm) from its other body
measurements with scikit-learn ``LinearRegression``. Missing values are filled
with the training-set column mean, as in the notebook.

Run ``python possum_model.py`` to train and print the evaluation metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parent / "possum.csv"
TARGET = "totlngth"
# Notebook features: every column except the target, `sex` and `Pop`.
# `case` (row ID) is also excluded: it carries no biological information.
FEATURES = ["site", "age", "hdlngth", "skullw", "taill", "footlgth", "earconch", "eye", "chest", "belly"]
FEATURE_LABELS = {
    "site": "Trapping site (1–7)",
    "age": "Age (years)",
    "hdlngth": "Head length (mm)",
    "skullw": "Skull width (mm)",
    "taill": "Tail length (cm)",
    "footlgth": "Foot length (mm)",
    "earconch": "Ear conch length (mm)",
    "eye": "Eye distance (mm)",
    "chest": "Chest girth (cm)",
    "belly": "Belly girth (cm)",
}
RANDOM_STATE = 42


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def build_model() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="mean")), ("regressor", LinearRegression())])


def _metrics(y_true, y_pred) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE) -> dict:
    """80/20 split as in the notebook (now reproducible), plus 5-fold CV R²."""
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = build_model().fit(X_train, y_train)
    # Shuffle folds: the CSV is ordered by site, so unshuffled folds would hold out whole sites.
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_r2 = cross_val_score(build_model(), X, y, cv=cv, scoring="r2")
    return {
        "model": model,
        "train": _metrics(y_train, model.predict(X_train)),
        "test": _metrics(y_test, model.predict(X_test)),
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "test_predictions": pd.DataFrame({"actual": y_test, "predicted": model.predict(X_test)}),
    }


def coefficients(model: Pipeline) -> pd.Series:
    return pd.Series(model.named_steps["regressor"].coef_, index=FEATURES)


def predict(model: Pipeline, data: pd.DataFrame | dict) -> np.ndarray:
    """Predict total length for one measurement dict or a DataFrame of rows."""
    frame = pd.DataFrame([data]) if isinstance(data, dict) else data
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return model.predict(frame[FEATURES].apply(pd.to_numeric, errors="coerce"))


if __name__ == "__main__":
    result = train_and_evaluate(load_data())
    for split in ("train", "test"):
        print(split, {k: round(v, 3) for k, v in result[split].items()})
    print(f"5-fold CV R2: {result['cv_r2_mean']:.3f} ± {result['cv_r2_std']:.3f}")
