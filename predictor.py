from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


ARTIFACT_DIR = Path("artifacts")

missing_value_handler = joblib.load(ARTIFACT_DIR / "missing_value_handler.joblib")
preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
model = joblib.load(ARTIFACT_DIR / "loan_model.joblib")
label_encoder = joblib.load(ARTIFACT_DIR / "label_encoder.joblib")


# Must match the training notebook (cat_cols + num_cols + ['Credit_History'])
CLEANED_FEATURE_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

EXPECTED_INPUT_COLS = ["Loan_ID"] + CLEANED_FEATURE_COLS


def _safe_log(series: pd.Series) -> np.ndarray:
    v = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(dtype=float)
    v = np.where(v > 0, v, 1.0)
    return np.log(v)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    df["Dependents"] = df["Dependents"].replace("3+", "3")
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")

    df["ApplicantIncome"] = pd.to_numeric(df["ApplicantIncome"], errors="coerce")
    df["CoapplicantIncome"] = pd.to_numeric(df["CoapplicantIncome"], errors="coerce")

    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_Log"] = _safe_log(df["LoanAmount"])
    df["Total_Income_Log"] = _safe_log(df["Total_Income"])

    df = df.drop(
        columns=[
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Total_Income",
            "Loan_ID",
        ],
        errors="ignore",
    )
    return df


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in EXPECTED_INPUT_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = df[EXPECTED_INPUT_COLS]

    df["ApplicantIncome"] = pd.to_numeric(df["ApplicantIncome"], errors="coerce")
    df["CoapplicantIncome"] = pd.to_numeric(df["CoapplicantIncome"], errors="coerce")
    df["LoanAmount"] = pd.to_numeric(df["LoanAmount"], errors="coerce")
    df["Loan_Amount_Term"] = pd.to_numeric(df["Loan_Amount_Term"], errors="coerce")
    df["Credit_History"] = pd.to_numeric(df["Credit_History"], errors="coerce")

    return df


def _prepare_features(raw_df: pd.DataFrame):
    raw_df = _normalize_input(raw_df)

    cleaned_arr = missing_value_handler.transform(raw_df)

    # IMPORTANT: use the SAME column order the handler was trained on
    cleaned_df = pd.DataFrame(cleaned_arr, columns=CLEANED_FEATURE_COLS, index=raw_df.index)

    # Add Loan_ID just for feature_engineering drop logic (safe)
    fe_input = pd.concat([raw_df[["Loan_ID"]], cleaned_df], axis=1)
    fe_df = feature_engineering(fe_input)

    X = preprocessor.transform(fe_df)
    return X


def predict_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame([payload])
    X = _prepare_features(df)

    pred = model.predict(X)
    proba = model.predict_proba(X)[0]
    label = label_encoder.inverse_transform(pred)[0]

    return {"Loan_Status": str(label), "confidence": float(np.max(proba))}


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    X = _prepare_features(df)

    preds = model.predict(X)
    confs = model.predict_proba(X).max(axis=1)
    labels = label_encoder.inverse_transform(preds)

    out = df.copy()
    out["Loan_Status"] = labels
    out["confidence"] = confs.astype(float)
    return out
