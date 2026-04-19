AquaBio-AI — Water Quality Risk Assessment System


A dual-engine water potability prediction system combining WHO guideline rule-checking (Biotech layer) with a 3-model Machine Learning ensemble (AI layer), built as part of an AIML + Biotech interdisciplinary project.

Project Overview
AquaBio-AI predicts whether a given water sample is safe for drinking by running two analysis engines simultaneously:
Layer 1 — Biotech Rule Engine: Checks each water parameter against real WHO 2022 drinking water guidelines and flags violations with severity scores.
Layer 2 — ML Ensemble: A weighted ensemble of Random Forest, XGBoost, and Gradient Boosting models trained on engineered features derived from 9 raw water quality parameters.

The combined output gives a risk score (0–100), a potability probability, and a SHAP explainability chart showing which parameters drove the prediction.

Why This Approach?
Most water quality ML projects use a single model on raw features. AquaBio-AI introduces:
deterministic biotech rules + probabilistic ML
decision threshold optimized to 0.53 via F1-score maximization
per-prediction SHAP waterfall plots using TreeExplainer

Training Models:
rf_model.pkl            # Trained Random Forest model
xgb_model.pkl           # Trained XGBoost model
gb_model.pkl            # Trained Gradient Boosting model
scaler.pkl              # StandardScaler fitted on training data

<img width="538" height="450" alt="image" src="https://github.com/user-attachments/assets/87c34656-db85-434d-9629-c025c9ad6706" />




<img width="591" height="193" alt="image" src="https://github.com/user-attachments/assets/53d89bc1-a4ad-4f45-b891-2a1e171278b5" />

The dataset (Kaggle Water Potability) is known to have weak feature-to-label correlation. Published research on this same dataset reports 65–70% as the practical accuracy ceiling. 


How to Run Locally
1. Clone the repository
git clone https://github.com/yourusername/aquabio-ai.git
cd aquabio-ai
2. Install dependencies
pip install -r requirements.txt
3. Run the app
python app.py
4. Open in browser
http://127.0.0.1:5000
