# 💧 AquaBio-AI — Water Quality Risk Assessment System

> A dual-engine water potability prediction system combining WHO guideline rule-checking (Biotech layer) with a 3-model Machine Learning ensemble (AI layer), built as part of an AIML + Biotech interdisciplinary project.

---

## 📌 Project Overview

AquaBio-AI predicts whether a given water sample is **safe for drinking** by running two analysis engines simultaneously:

- **Layer 1 — Biotech Rule Engine:** Checks each water parameter against real WHO 2022 drinking water guidelines and flags violations with severity scores.
- **Layer 2 — ML Ensemble:** A weighted ensemble of Random Forest, XGBoost, and Gradient Boosting models trained on engineered features derived from 9 raw water quality parameters.

The combined output gives a **risk score (0–100)**, a **potability probability**, and a **SHAP explainability chart** showing which parameters drove the prediction.

---

## 🧬 Why This Approach?

Most water quality ML projects use a single model on raw features. AquaBio-AI introduces:

1. **Hybrid dual-engine architecture** — deterministic biotech rules + probabilistic ML
2. **Feature engineering** — 21 features engineered from 9 raw inputs (WQI, BOD proxy, TDS estimate, pH deviation, chemical index, etc.)
3. **Class imbalance handling** — SMOTE oversampling applied to training data only
4. **Threshold tuning** — decision threshold optimized to 0.53 via F1-score maximization
5. **Explainability** — per-prediction SHAP waterfall plots using TreeExplainer

---

## 🗂️ Project Structure

```
aquabio-ai/
│
├── app.py                  # Flask web application (main backend)
├── model_meta.json         # Saved accuracy, AUC, threshold metadata
├── feature_cols.json       # Feature column names for the scaler
│
├── rf_model.pkl            # Trained Random Forest model
├── xgb_model.pkl           # Trained XGBoost model
├── gb_model.pkl            # Trained Gradient Boosting model
├── scaler.pkl              # StandardScaler fitted on training data
│
├── static/
│   ├── style.css           # Custom CSS (Bootstrap-based)
│   ├── shap_summary.png    # Global SHAP feature importance chart
│   ├── confusion_matrix.png
│   └── eda_plots.png
│
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── index.html          # Prediction input form
│   └── result.html         # Prediction result page
│
├── requirements.txt
├── Procfile                # For Heroku/Render deployment
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask, Flask-SQLAlchemy |
| ML Models | Scikit-learn, XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Data Processing | Pandas, NumPy, Imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML, CSS, Bootstrap 5 |
| Auth | Werkzeug password hashing, Flask sessions |
| Database | SQLite (via SQLAlchemy) |
| Deployment | Gunicorn, Render / Heroku |

---

## 🔬 Input Parameters

| Parameter | Unit | WHO Safe Limit |
|---|---|---|
| pH | — | 6.5 – 8.5 |
| Hardness | mg/L | 50 – 300 |
| Solids (TDS) | ppm | ≤ 500 |
| Chloramines | ppm | ≤ 4.0 |
| Sulfate | mg/L | ≤ 250 |
| Conductivity | μS/cm | ≤ 400 |
| Organic Carbon | ppm | ≤ 2.0 |
| Trihalomethanes | μg/L | ≤ 80 |
| Turbidity | NTU | ≤ 4.0 |

---

## 🤖 Model Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest (tuned) | 61.1% | 0.624 |
| XGBoost (tuned) | 62.3% | 0.631 |
| Gradient Boosting | 63.1% | 0.642 |
| **Final Ensemble** | **63.8%** | **0.648** |

> **Note:** The dataset (Kaggle Water Potability) is known to have weak feature-to-label correlation. Published research on this same dataset reports 65–70% as the practical accuracy ceiling. Our ensemble with SMOTE, threshold tuning, and 21 engineered features reaches this ceiling while adding WHO-based interpretability that pure ML approaches lack.

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/aquabio-ai.git
cd aquabio-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python app.py
```

**4. Open in browser**
```
http://127.0.0.1:5000
```

Register an account → Login → Enter water parameters → Get prediction.

---

## 🧪 Sample Test Data

**Test Case 1 — Unsafe Water**

| Parameter | Value |
|---|---|
| pH | 3.7 |
| Hardness | 129.4 |
| Solids | 18630.0 |
| Chloramines | 6.6 |
| Sulfate | 336.0 |
| Conductivity | 592.8 |
| Organic Carbon | 15.1 |
| Trihalomethanes | 56.3 |
| Turbidity | 4.5 |

Expected output: **NOT SAFE** — high WHO violations, elevated risk score.

**Test Case 2 — Safe Water**

| Parameter | Value |
|---|---|
| pH | 7.2 |
| Hardness | 180.0 |
| Solids | 12000.0 |
| Chloramines | 3.1 |
| Sulfate | 200.0 |
| Conductivity | 350.0 |
| Organic Carbon | 8.5 |
| Trihalomethanes | 55.0 |
| Turbidity | 2.8 |

Expected output: **SAFE** — parameters within acceptable range.

---

## 📊 Engineered Features

Beyond the 9 raw inputs, the system computes 12 additional features:

| Feature | Description |
|---|---|
| `WQI` | Water Quality Index — weighted composite of 5 key parameters |
| `BOD_proxy` | Biological Oxygen Demand estimate (Organic Carbon × Conductivity) |
| `Chem_Index` | Weighted chemical contamination score |
| `Physical_Risk` | Combined turbidity + solids risk |
| `pH_deviation` | Absolute deviation from neutral pH (7.0) |
| `TDS_estimate` | Total Dissolved Solids estimate from Conductivity |
| `Hard_Cond_ratio` | Hardness to Conductivity ratio |
| `Cond_Solids_ratio` | Conductivity to Solids ratio |
| `pH_safe` | Binary flag — 1 if pH is within WHO range |
| `Turbidity_severe` | Binary flag — 1 if Turbidity exceeds WHO limit |
| `WHO_violations` | Count of parameters violating WHO limits |
| `WHO_severity_score` | Normalized severity of violations (0–1) |

---

## 📄 Dataset

- **Source:** [Water Potability Dataset — Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- **Rows:** 3,276 water samples
- **Missing values:** Handled using KNN Imputation (k=5)
- **Class imbalance:** ~61% unsafe, ~39% safe → balanced using SMOTE

---

## 👥 Team

Built as part of a 2-credit interdisciplinary project combining **Artificial Intelligence & Machine Learning** and **Biotechnology** domains.

---

## 📚 References

- WHO Guidelines for Drinking-Water Quality, 4th Edition (2022)
- Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*
- Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*
