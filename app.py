from __future__ import annotations

import tempfile
from pathlib import Path
import gradio as gr
import pandas as pd

from predictor import predict_one, predict_batch


def run_single(
    Loan_ID,
    Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area,
):
    payload = {
        "Loan_ID": Loan_ID,
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
    }

    r = predict_one(payload)
    return f"Prediction: {r['Loan_Status']}\nConfidence: {r['confidence']:.2f}"


def run_batch(file_obj):
    if file_obj is None:
        raise gr.Error("Upload a CSV file.")

    df = pd.read_csv(file_obj.name)
    pred_df = predict_batch(df)

    out_dir = Path(tempfile.mkdtemp())
    out_path = out_dir / "predictions.csv"
    pred_df.to_csv(out_path, index=False)

    return pred_df.head(15), str(out_path)


with gr.Blocks(title="Loan Approval Predictor") as demo:
    gr.Markdown("# Loan Approval Predictor")

    with gr.Tabs():
        with gr.Tab("Single Prediction"):
            with gr.Row():
                Loan_ID = gr.Textbox(label="Loan_ID", value="LP000000")
                Gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
                Married = gr.Dropdown(["Yes", "No"], label="Married", value="Yes")
                Dependents = gr.Dropdown(["0", "1", "2", "3+"], label="Dependents", value="0")

            with gr.Row():
                Education = gr.Dropdown(["Graduate", "Not Graduate"], label="Education", value="Graduate")
                Self_Employed = gr.Dropdown(["No", "Yes"], label="Self_Employed", value="No")
                Property_Area = gr.Dropdown(["Urban", "Semiurban", "Rural"], label="Property_Area", value="Urban")

            with gr.Row():
                ApplicantIncome = gr.Number(label="ApplicantIncome", value=5000)
                CoapplicantIncome = gr.Number(label="CoapplicantIncome", value=0)
                LoanAmount = gr.Number(label="LoanAmount", value=150)

            with gr.Row():
                Loan_Amount_Term = gr.Number(label="Loan_Amount_Term", value=360)
                Credit_History = gr.Number(label="Credit_History (0 or 1)", value=1)

            btn_single = gr.Button("Predict")
            out_single = gr.Textbox(label="Result", lines=3)

            btn_single.click(
                fn=run_single,
                inputs=[
                    Loan_ID,
                    Gender,
                    Married,
                    Dependents,
                    Education,
                    Self_Employed,
                    ApplicantIncome,
                    CoapplicantIncome,
                    LoanAmount,
                    Loan_Amount_Term,
                    Credit_History,
                    Property_Area,
                ],
                outputs=out_single,
            )

        with gr.Tab("Batch Prediction"):
            gr.Markdown("Upload a CSV like `loan_test.csv` and download predictions.")

            file_in = gr.File(label="Upload CSV", file_types=[".csv"])
            btn_batch = gr.Button("Run Batch Prediction")

            preview = gr.Dataframe(label="Preview (first 15 rows)", interactive=False)
            file_out = gr.File(label="Download predictions.csv")

            btn_batch.click(fn=run_batch, inputs=file_in, outputs=[preview, file_out])


if __name__ == "__main__":
    demo.launch()
