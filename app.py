from __future__ import annotations
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from predictor import predict_one, predict_batch


def run_single(
    Gender, Married, Dependents, Education, Self_Employed, Property_Area,
    Loan_Amount_Term, Credit_History,
    ApplicantIncome, CoapplicantIncome, LoanAmount
):
    payload = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "Property_Area": Property_Area,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
    }

    result = predict_one(payload)

   
    pred = result.get("Loan_Status", "")
    conf = result.get("confidence", None)
    notes = result.get("notes", "")

    if conf is None:
        return f"Prediction: {pred}\n{notes}"
    return f"Prediction: {pred}\nConfidence: {conf:.2f}\n{notes}"



def run_batch(file_obj):
    """
    file_obj is a Gradio File object.
    We read it, predict, then write out a new CSV to download.
    """
    if file_obj is None:
        raise gr.Error("Please upload a CSV file.")

    df = pd.read_csv(file_obj.name)

    pred_df = predict_batch(df)

    tmpdir = Path(tempfile.mkdtemp())
    out_path = tmpdir / "predictions.csv"
    pred_df.to_csv(out_path, index=False)

    return pred_df.head(15), str(out_path)



with gr.Blocks(title="Loan Approval Predictor") as demo:
    gr.Markdown("# Loan Approval Predictor (Gradio)")
    gr.Markdown(
        "This app supports **single prediction** (manual entry) and **batch prediction** (CSV upload). "
    )

    with gr.Tabs():
        with gr.Tab("Single Prediction"):
            with gr.Row():
                Gender = gr.Dropdown(["Female", "Male"], label="Gender", value="Male")
                Married = gr.Dropdown(["No", "Yes"], label="Married", value="No")
                Dependents = gr.Dropdown(["0", "1", "2", "3+"], label="Dependents", value="0")
                Education = gr.Dropdown(["Graduate", "Not Graduate"], label="Education", value="Graduate")
                Self_Employed = gr.Dropdown(["No", "Yes"], label="Self Employed", value="No")
                Property_Area = gr.Dropdown(["Rural", "Semiurban", "Urban"], label="Property Area", value="Urban")

            with gr.Row():
                Loan_Amount_Term = gr.Number(label="Loan Amount Term", value=360)
                Credit_History = gr.Number(label="Credit History (0 or 1)", value=1)

            with gr.Row():
                ApplicantIncome = gr.Number(label="Applicant Income", value=5000)
                CoapplicantIncome = gr.Number(label="Coapplicant Income", value=0)
                LoanAmount = gr.Number(label="Loan Amount", value=150)

            btn_single = gr.Button("Predict")
            out_single = gr.Textbox(label="Result", lines=4)

            btn_single.click(
                fn=run_single,
                inputs=[
                    Gender, Married, Dependents, Education, Self_Employed, Property_Area,
                    Loan_Amount_Term, Credit_History,
                    ApplicantIncome, CoapplicantIncome, LoanAmount,
                ],
                outputs=out_single,
            )

        with gr.Tab("Batch Prediction (CSV Upload)"):
            gr.Markdown(
                "Upload a CSV (e.g. the provided `loan_test.csv`). "
                "The app will return a predictions CSV with `Loan_Status` appended."
            )

            file_in = gr.File(label="Upload CSV", file_types=[".csv"])
            btn_batch = gr.Button("Run Batch Prediction")

            preview = gr.Dataframe(label="Preview (first 15 rows)", interactive=False)
            file_out = gr.File(label="Download predictions.csv")

            btn_batch.click(
                fn=run_batch,
                inputs=file_in,
                outputs=[preview, file_out],
            )


if __name__ == "__main__":
    demo.launch()
