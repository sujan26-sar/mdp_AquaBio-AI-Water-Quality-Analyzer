from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import numpy as np
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "aquabio-secret-2024")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Database 
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

#  Load models 
rf     = joblib.load("rf_model.pkl")
xgb    = joblib.load("xgb_model.pkl")
gb     = joblib.load("gb_model.pkl")
scaler = joblib.load("scaler.pkl")
meta   = json.load(open("model_meta.json"))
feats  = json.load(open("feature_cols.json"))

BEST_THRESH = meta.get("best_threshold", 0.53)

#  WHO limits for rule engine 
WHO_LIMITS = {
    'ph':              (6.5, 8.5),
    'Chloramines':     (0,   4.0),
    'Sulfate':         (0,   250.0),
    'Conductivity':    (0,   400.0),
    'Trihalomethanes': (0,   80.0),
    'Turbidity':       (0,   4.0),
}

def who_engine(data: dict):
    violations, details = [], []
    for param, (lo, hi) in WHO_LIMITS.items():
        val = data.get(param, 0)
        if val < lo or val > hi:
            violations.append(param)
            details.append(f"{param} = {val:.2f} (limit: {lo}–{hi})")
    count = len(violations)
    if count == 0:   label = "SAFE"
    elif count <= 2: label = "MODERATE RISK"
    else:            label = "HIGH RISK"
    return count, label, details

def build_features(d: dict):
    ph   = d['ph'];   turb = d['Turbidity']
    chl  = d['Chloramines']; sulf = d['Sulfate']
    thm  = d['Trihalomethanes']; cond = d['Conductivity']
    hard = d['Hardness']; sol  = d['Solids']; oc = d['Organic_carbon']

    who_viol, who_label, _ = who_engine(d)
    who_sev = min(who_viol / 6.0, 1.0)

    chem_idx     = chl*0.3 + (sulf/250)*0.3 + (thm/80)*0.4
    phys_risk    = (turb/4.0)*0.5 + (sol/50000)*0.5
    ph_dev       = abs(ph - 7.0)
    cond_sol     = cond / (sol + 1)
    wqi          = ((ph/8.5)*20 + (1-turb/4)*20 + (1-chl/4)*20 +
                    (1-thm/80)*20 + (1-sulf/250)*20)
    bod_proxy    = oc * cond / 1000
    hard_cond    = hard / (cond + 1)
    tds          = cond * 0.65
    ph_safe      = int(6.5 <= ph <= 8.5)
    turb_severe  = int(turb > 4.0)

    return np.array([[
        ph, hard, sol, chl, sulf, cond, oc, thm, turb,
        who_viol, who_sev,
        chem_idx, phys_risk, ph_dev, cond_sol,
        wqi, bod_proxy, hard_cond, tds, ph_safe, turb_severe
    ]])

#    Routes
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session["user"])

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u = User.query.filter_by(username=request.form["username"]).first()
        if u and check_password_hash(u.password, request.form["password"]):
            session["user"] = u.username
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        hashed = generate_password_hash(request.form["password"])
        try:
            db.session.add(User(username=request.form["username"], password=hashed))
            db.session.commit()
            return redirect(url_for("login"))
        except:
            return render_template("register.html", error="Username already exists")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))
    try:
        data = {
            'ph':              float(request.form["ph"]),
            'Hardness':        float(request.form["Hardness"]),
            'Solids':          float(request.form["Solids"]),
            'Chloramines':     float(request.form["Chloramines"]),
            'Sulfate':         float(request.form["Sulfate"]),
            'Conductivity':    float(request.form["Conductivity"]),
            'Organic_carbon':  float(request.form["Organic_carbon"]),
            'Trihalomethanes': float(request.form["Trihalomethanes"]),
            'Turbidity':       float(request.form["Turbidity"]),
        }

        # Build feature vector
        feat_vec = build_features(data)

        # Check feature count matches scaler
        if feat_vec.shape[1] != len(feats):
            feat_vec = feat_vec[:, :len(feats)]

        scaled = scaler.transform(feat_vec)

        # Ensemble probability
        prob = (
            rf.predict_proba(scaled)[0][1] * 0.4 +
            xgb.predict_proba(scaled)[0][1] * 0.35 +
            gb.predict_proba(scaled)[0][1]  * 0.25
        )
        prediction = int(prob >= BEST_THRESH)

        # WHO rule engine
        who_count, who_label, who_details = who_engine(data)

        # Confidence
        confidence = round(prob * 100 if prediction == 1 else (1 - prob) * 100, 1)

        # Risk score (0–100)
        risk_score = round((1 - prob) * 100, 1)

        # Result labels
        if prediction == 1:
            result     = "SAFE"
            result_msg = "Water is Safe for Drinking"
            badge      = "success"
        else:
            result     = "NOT SAFE"
            result_msg = "Water is NOT Safe for Drinking"
            badge      = "danger"

        # SHAP waterfall for this prediction
        explainer   = shap.TreeExplainer(xgb)
        shap_vals   = explainer.shap_values(scaled)
        shap_img    = "static/shap_waterfall.png"

        shap_feat_names = feats if len(feats) == scaled.shape[1] else [f"f{i}" for i in range(scaled.shape[1])]
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_vals, scaled,
                          feature_names=shap_feat_names,
                          show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(shap_img, dpi=120, bbox_inches='tight')
        plt.close()

        return render_template("result.html",
            username    = session["user"],
            result      = result,
            result_msg  = result_msg,
            badge       = badge,
            confidence  = confidence,
            risk_score  = risk_score,
            prob        = round(prob * 100, 1),
            who_label   = who_label,
            who_count   = who_count,
            who_details = who_details,
            accuracy    = meta["final_accuracy"],
            auc         = meta["final_auc"],
            data        = data,
            shap_img    = shap_img,
        )

    except Exception as e:
        return render_template("index.html",
                               error=f"Error: {str(e)}",
                               username=session.get("user",""))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run()