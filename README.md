# MACHINE LEARNIING GROUP PROJECT

# Title: Building and Deploying a Predictive Machine Learning Model on Hugging Face

 Members OF Group 3
1. **[Name]** - [Student ID]
2. **[VICTORY EZEALA]** - [25120133011]
3. **Ladipo Ipadeola**
4. **[Name]** - [Student ID]
5. **[Name]** - [Student ID]

---

## Project Overview
This project aims to automate the loan eligibility process (real-time) based on customer details provided while filling out online application forms. We developed a supervised machine learning model using **Random Forest** to predict whether a loan should be **Approved (Y)** or **Rejected (N)**.

The final model is deployed as a web application using **Gradio** on **Hugging Face Spaces**.

---

Live Deployment
 **HUGGING FACE SPACE LINK-- 



 Dataset Description
The dataset consists of two CSV files:
* loan_train.csv: Used for training and validating the model. Contains the target variable `Loan_Status`.
* loan_test.csv: Used for final predictions.

Key Features:
* `Gender`, `Married`, `Dependents`, `Education`
* `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`
* `Credit_History` (Most critical feature)
* `Property_Area`



 Technical Implementation

 1. Data Preprocessing
* Missing Values:Numerical columns filled with Median; Categorical columns filled with Mode.
* Feature Engineering: Created `Total_Income` and applied Log Transformation to normalize income and loan amount distributions.
* Encoding: Used `LabelEncoder` to convert categorical text data into numbers.

2. Model Development
* **Algorithm:** Random Forest Classifier.
* **Hyperparameter Tuning:**
    * `n_estimators=200` (Number of trees)
    * `class_weight='balanced'` (To handle imbalance between approved/rejected loans)
    * `max_depth=10` (To prevent overfitting)
* **Metrics:** Evaluated using Accuracy and Confusion Matrix.

 3. Deployment
* Built a user-friendly interface using **Gradio**.
* Hosted on **Hugging Face Spaces** using the standard SDK.

---

Installation & Usage
To run this project locally:


   ```bash
   git clone [GITHUB_REPO_LINK]