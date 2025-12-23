from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import joblib

from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Define feature engineering function


def feature_engineering(df):
    df = df.copy()
    df['Dependents'] = df['Dependents'].replace('3+', '3')
    df['Dependents'] = pd.to_numeric(df['Dependents'])
    if {"ApplicantIncome", "CoapplicantIncome"}.issubset(df.columns):
        df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Safe log helper: log(x) on x>0 else 0
    def safe_log(series):
        return np.log(np.where(series.astype(float) > 0, series.astype(float), 1.0))

    if "LoanAmount" in df.columns:
        df["LoanAmount_Log"] = safe_log(df["LoanAmount"])

    if "Total_Income" in df.columns:
        df["Total_Income_Log"] = safe_log(df["Total_Income"])
    cols_to_drop = ["ApplicantIncome",
                    "CoapplicantIncome", "LoanAmount", "Total_Income"]
    if "Loan_ID" in df.columns:
        cols_to_drop.append("Loan_ID")

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df


def feature_names_out(input_features_1, input_features_2):
    return np.array([
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'Property_Area', 'Loan_Amount_Term', 'Credit_History',
        'LoanAmount_Log', 'Total_Income_Log'
    ], dtype=object
    )


feature_eng = FunctionTransformer(
    feature_engineering,
    validate=False,
    feature_names_out=feature_names_out
)

# Load the preprocessor and model

data_transformation = joblib.load("loan_preprocessor.pkl")
model = joblib.load("loan_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


def predict_one(payload: Dict[str, Any]) -> Dict[str, Any]:

    input_df = pd.DataFrame([payload])
    transformed_input = data_transformation.transform(input_df)
    prediction = model.predict(transformed_input)
    pred_label = label_encoder.inverse_transform(prediction)[0]
    confidence = max(model.predict_proba(transformed_input)[0])
    return {"Loan_Status": pred_label,
            "confidence": confidence,
            "notes": "Stub prediction (replace with real model pipeline)."}


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    transformed_input = data_transformation.transform(df)
    predictions = model.predict(transformed_input)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    confidences = model.predict_proba(transformed_input).max(axis=1)
    out = df.copy()
    out["Loan_Status"] = decoded_predictions
    out["confidence"] = confidences
    out["notes"] = "Batch prediction using trained model."
    return out
