<div align="center">

# 📊 ChurnIQ
### Customer Behavior Analytics & Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green?style=flat-square)
![License](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

**Teyzix Core Internship · June Batch · Task ID: ML-INT-1**

*An end-to-end ML system that turns raw customer data into churn predictions, business insights, and automated email reports.*

</div>

---

## 🧩 What does this project do?

Companies lose revenue every time a customer leaves — but most don't know *who* is about to leave or *why*. ChurnIQ solves that.

You give it a CSV of customer data. It gives you back:

- ✅ Which customers are about to churn (Low / Medium / High risk)
- ✅ How much revenue is at risk
- ✅ What's causing the churn
- ✅ An interactive dashboard to explore everything
- ✅ A professional email report you can send to your team

---

## 📁 Project Files

```
dataAnalyze/
│
├── 📄 customerservices.csv       → Input dataset (7,043 customers)
│
├── 🐍 main.py                    → Run everything from one command
├── 🐍 data_analysis.py           → Clean & explore the data
├── 🐍 feature_engineering.py     → Create ML features + customer segments
├── 🐍 visualization.py           → Generate 7 charts
├── 🐍 models.py                  → Train & evaluate ML models
├── 🐍 insights.py                → Generate business insights report
│
├── 🐍 shap_explainability.py     
├── 🐍 weekly_pipeline.py         
├── 🐍 email_report.py            
├── 🐍 app.py                     
│
├── 📄 requirements.txt
├── 📄 README.md
│
└── 📂 outputs/
    ├── clean_customers.csv
    ├── plots/          → 7 chart images
    ├── reports/        → insights + model results
    └── models/         → saved champion model
```

---

## ⚙️ Setup — Step by Step

### Step 1 — Download & enter the folder

```bash
cd dataAnalyze
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install all libraries

```bash
pip install -r requirements.txt
```

That's it. You're ready.

---

## 🚀 How to Run

### Run the full ML pipeline

```bash
python main.py
```

This runs everything in order — data cleaning → feature engineering → visualizations → model training → business insights. All outputs go to the `outputs/` folder.

### Run with SHAP explainability

```bash
python main.py --shap
```

### Launch the interactive dashboard

```bash
streamlit run app.py
```

Opens at **http://localhost:8501** in your browser. Upload the CSV from the sidebar and the full dashboard loads instantly.

### Run weekly automation

```bash
python weekly_pipeline.py
```

Runs every Monday at 8 AM automatically. Saves predictions to `outputs/reports/`.

### Send a standalone email report

```bash
# Just save the HTML report locally (no email sent)
python email_report.py --preview

# Send to someone
python email_report.py --to manager@company.com
```

---

## 📧 Email Report Setup (Gmail)

Both the dashboard and `email_report.py` can send HTML email reports directly. Gmail needs an **App Password** — your normal Gmail password won't work here.

**Get your App Password in 3 steps:**

1. Go to → [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
2. Type any app name (e.g. `ChurnIQ`) → click **Create**
3. Copy the **16-character password** it gives you

**In the Streamlit dashboard**, fill the sidebar fields:

| Field | What to enter |
|---|---|
| Recipient Email | Who you're sending to |
| SMTP Host | `smtp.gmail.com` |
| SMTP Port | `587` |
| Sender Email | Your Gmail address |
| App Password | The 16-char password from above |

Then click **📧 Send Email Report**.

**For `email_report.py`**, set these before running:

```bash
# Windows
set SENDER_EMAIL=your@gmail.com
set SENDER_PASS=abcd efgh ijkl mnop
set RECIPIENTS=boss@company.com

# Mac / Linux
export SENDER_EMAIL=your@gmail.com
export SENDER_PASS=abcd efgh ijkl mnop
export RECIPIENTS=boss@company.com

python email_report.py
```

---

## 🤖 ML Models

Three models are trained and compared automatically:

| Model | What it does |
|---|---|
| Logistic Regression | Fast baseline, good interpretability |
| Random Forest | 200 trees, handles non-linear patterns |
| XGBoost | Best accuracy, gradient boosting |

Each model is evaluated on: **Accuracy · Precision · Recall · F1 · ROC-AUC**

The best performing model (by ROC-AUC) is automatically saved as the **champion model** and used for all churn probability predictions.

---

## 🛠️ Features Created

`feature_engineering.py` creates 13 new features from the raw data:

| Feature | What it captures |
|---|---|
| `usage_score` | How actively a customer uses the service |
| `payment_risk` | Customers paying by electronic check (highest churn risk) |
| `payment_stability` | Customers on auto-pay (most stable) |
| `support_interaction_count` | How many support services they use |
| `charges_ratio` | Total vs monthly spend ratio |
| `avg_monthly_spend` | Normalized spending over tenure |
| `spending_growth_index` | Whether spending is growing or declining |
| `tenure_group` | New / Growing / Loyal / Long-term |
| `customer_value` | High / Medium / Low value segment |
| `Contract_encoded` | Numeric contract type |
| `PaymentMethod_encoded` | Numeric payment method |
| `InternetService_encoded` | Numeric internet service type |

---

## 📊 Visualizations Generated

All 7 charts are saved to `outputs/plots/` automatically:

1. **Churn Distribution** — Overall churn rate breakdown
2. **Monthly Charges by Churn** — Do churners pay more or less?
3. **Tenure Distribution** — When do customers leave?
4. **Contract Type vs Churn** — Month-to-month vs annual churn rates
5. **Correlation Heatmap** — Which features relate to churn
6. **Revenue Trend by Contract** — Revenue across contract types
7. **Feature Importance** — What drives churn predictions most

---

## 📦 Dataset

**File:** `customerservices.csv` — 7,043 customers, 21 columns

Important columns:

| Column | Description |
|---|---|
| `tenure` | How many months the customer has been with the company |
| `MonthlyCharges` | What they pay per month |
| `TotalCharges` | Total amount paid |
| `Contract` | Month-to-month / One year / Two year |
| `PaymentMethod` | How they pay |
| `Churn` | **Target** — Yes if they left, No if they stayed |

---

## ✅ Task Deliverables

| Requirement | Done |
|---|---|
| Data Analysis (EDA, missing values, outliers) | ✅ |
| Feature Engineering (13 features) | ✅ |
| Data Visualizations (7 charts) | ✅ |
| Logistic Regression | ✅ |
| Random Forest | ✅ |
| XGBoost | ✅ |
| Accuracy / Precision / Recall / F1 / ROC-AUC | ✅ |
| Customer Segmentation (High / Medium / Low) | ✅ |
| Churn Probability + Risk Category | ✅ |
| Business Insights Report | ✅ |
| Streamlit Dashboard | ✅ Bonus |
| SHAP Explainability | ✅ Bonus |
| Weekly Prediction Pipeline | ✅ Bonus |
| Email Report Generation | ✅ Bonus |

---

## 🧰 Tech Stack

| Category | Libraries |
|---|---|
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, xgboost |
| Explainability | shap |
| Dashboard | streamlit |
| Automation | schedule |
| Email | smtplib (built-in) |

---

<div align="center">

Made for **Teyzix Core Internship · June Batch 2026**

</div>
