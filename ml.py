import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve

st.set_page_config(layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Configuration")

DATASETS = {
    "Telco (IBM/Kaggle)": "telco.csv",
    "Bank Customer Churn": "bank_churn.csv",
}
dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
dataset_file = DATASETS[dataset_name]

penalty = st.sidebar.selectbox("Regularisation", ["l2", "l1"])
C = st.sidebar.slider("C (Regularisation Strength)", 0.01, 10.0, 1.0)

cost_fn = st.sidebar.slider("Cost FN (Missed Churn)", 10, 500, 100)
cost_fp = st.sidebar.slider("Cost FP (False Alarm)", 1, 100, 10)

# -----------------------------
# LOAD DATA
# FIX 1: target is separated BEFORE get_dummies so it is never renamed/dropped
# FIX 2: load_data returns X, y directly so no KeyError at df.drop() later
# FIX 3: encoding fallback chain kept
# -----------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(filepath):
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except Exception:
            continue

    # Drop irrelevant ID columns
    drop_cols = ["customerID", "CustomerId", "RowNumber", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Detect target column
    target_col = None
    for col in ["Churn", "Exited"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        st.error(f"Target column not found! Columns: {df.columns.tolist()}")
        st.stop()

    # Clean and encode target
    df[target_col] = (
        df[target_col].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0})
    )
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)

    # Coerce numeric-like string columns (e.g. TotalCharges)
    for col in df.columns:
        if col != target_col and df[col].dtype == object:
            temp = pd.to_numeric(df[col], errors="coerce")
            if temp.notna().mean() > 0.8:
                df[col] = temp

    df = df.dropna()

    # ---- KEY FIX: split before get_dummies ----
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X, drop_first=True)
    # -------------------------------------------

    return X, y

try:
    X, y = load_data(dataset_file)
except FileNotFoundError:
    st.error(f"**{dataset_file}** not found. Place it in the same folder as ml.py.")
    st.stop()

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------
# MODEL
# FIX: train_model is cached so it doesn't retrain on every slider change
# -----------------------------
@st.cache_resource
def train_model(penalty, C, dataset):   # dataset in signature busts cache on switch
    base = LogisticRegression(
        solver="liblinear", penalty=penalty, C=C,
        random_state=42, max_iter=1000
    )
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    calibrated.fit(X_train_scaled, y_train)
    return calibrated

model = train_model(penalty, C, dataset_file)

# FIX: extract fitted base estimators from calibrated model
# (instead of calling base_model.fit() again separately)
fitted_estimators = [cc.estimator for cc in model.calibrated_classifiers_]

# -----------------------------
# THRESHOLD OPTIMIZATION
# -----------------------------
probs = model.predict_proba(X_test_scaled)[:, 1]

best_thresh = 0.5
min_cost = float("inf")

for t in np.linspace(0, 1, 200):
    preds = (probs >= t).astype(int)
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))
    cost = fp * cost_fp + fn * cost_fn
    if cost < min_cost:
        min_cost = cost
        best_thresh = t

# -----------------------------
# METRICS
# -----------------------------
auc = roc_auc_score(y_test, probs)

final_preds = (probs >= best_thresh).astype(int)
tp  = int(np.sum((final_preds == 1) & (y_test == 1)))
fp_ = int(np.sum((final_preds == 1) & (y_test == 0)))
fn_ = int(np.sum((final_preds == 0) & (y_test == 1)))

precision_val = tp / (tp + fp_) if (tp + fp_) > 0 else 0.0
recall_val    = tp / (tp + fn_) if (tp + fn_) > 0 else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("ROC AUC",           f"{auc:.3f}")
col2.metric("Optimal Threshold", f"{best_thresh:.2f}")
col3.metric("Min Cost",          f"{int(min_cost):,}")
col4.metric("Precision",         f"{precision_val:.3f}")
col5.metric("Recall",            f"{recall_val:.3f}")

# -----------------------------
# ROC CURVE
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, probs, pos_label=1)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# -----------------------------
# FEATURE IMPORTANCE (Odds Ratio)
# FIX: uses averaged coefs from fitted CV folds, not a separate re-fit
# -----------------------------
st.header("📌 Feature Importance (Odds Ratio)")

feature_names = X.columns.tolist()
avg_coef = np.mean([est.coef_[0] for est in fitted_estimators], axis=0)
odds = np.exp(avg_coef)

importance_df = (
    pd.DataFrame({"Feature": feature_names, "OddsRatio": odds})
    .sort_values("OddsRatio", ascending=False)
    .head(15)
)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(importance_df["Feature"][::-1], importance_df["OddsRatio"][::-1])
ax2.axvline(1.0, color="red", linestyle="--", linewidth=0.8, label="Odds = 1 (no effect)")
ax2.set_xlabel("Odds Ratio")
ax2.set_title("Top Feature Importance")
ax2.legend()
st.pyplot(fig2)
plt.close(fig2)

# -----------------------------
# SHAP EXPLAINER
# FIX: uses the already-fitted estimator, not a re-trained one
# -----------------------------
@st.cache_resource
def build_explainer(dataset):           # dataset in signature busts cache on switch
    return shap.LinearExplainer(fitted_estimators[0], X_train_scaled)

explainer = build_explainer(dataset_file)

# -----------------------------
# PREDICTION UI
# -----------------------------
st.header("🔍 Predict Customer Churn")
st.caption("Adjust any feature below — values not shown default to the training mean.")

cols_per_row = 3
n_display = min(15, len(feature_names))

input_data = {}
rows = [st.columns(cols_per_row) for _ in range(-(-n_display // cols_per_row))]
for i, col_name in enumerate(feature_names[:n_display]):
    with rows[i // cols_per_row][i % cols_per_row]:
        input_data[col_name] = st.number_input(
            col_name, value=float(X[col_name].mean()), key=f"inp_{col_name}"
        )

if st.button("Predict"):
    full_input = pd.DataFrame([X.mean().to_dict()])
    full_input.update(pd.DataFrame([input_data]))
    full_input = full_input.reindex(columns=X.columns)

    input_scaled = scaler.transform(full_input)
    prob = model.predict_proba(input_scaled)[0][1]
    pred = int(prob >= best_thresh)

    shap_vals = explainer.shap_values(input_scaled)[0]
    top_idx   = np.argsort(np.abs(shap_vals))[-3:][::-1]
    top_features = [(feature_names[i], round(float(shap_vals[i]), 4)) for i in top_idx]

    st.subheader("Result")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Churn Probability", f"{prob:.3f}")
    res_col2.metric("Prediction", "🔴 Churn" if pred else "🟢 No Churn",
                    help=f"Threshold: {best_thresh:.2f}")

    st.markdown("**Top 3 risk factors (SHAP log-odds contribution):**")
    for feat, val in top_features:
        direction = "↑ increases" if val > 0 else "↓ decreases"
        st.write(f"- **{feat}**: {val:+.4f}  ({direction} churn probability)")

# -----------------------------
# HIGH RISK CUSTOMERS
# -----------------------------
st.header("🚨 High Risk Customers")

df_test = X_test.copy()
df_test["Churn_Prob"]      = probs
df_test["Predicted_Churn"] = final_preds
df_test["Actual_Churn"]    = y_test.values

high_risk = (
    df_test[df_test["Churn_Prob"] > best_thresh]
    .sort_values("Churn_Prob", ascending=False)
)

st.caption(f"{len(high_risk)} customers above threshold {best_thresh:.2f}. Showing top 20.")

display_cols = ["Churn_Prob", "Predicted_Churn", "Actual_Churn"] + feature_names[:5]
st.dataframe(
    high_risk[display_cols].head(20).style.format({"Churn_Prob": "{:.3f}"}),
    use_container_width=True,
)

# -----------------------------
# EARLY-WARNING REPORT
# -----------------------------
st.header("📋 Early-Warning Report")

report_summary = f"""CHURN EARLY-WARNING REPORT
===========================
Dataset     : {dataset_name}
Regulariser : {penalty.upper()}  (C={C})
Threshold   : {best_thresh:.2f}

PERFORMANCE
-----------
ROC AUC     : {auc:.4f}
Precision   : {precision_val:.4f}
Recall      : {recall_val:.4f}
Min Cost    : {int(min_cost):,}

HIGH-RISK CUSTOMERS: {len(high_risk)}
"""

risk_csv = high_risk[["Churn_Prob", "Predicted_Churn", "Actual_Churn"]].head(50).to_csv()

st.text(report_summary)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        label="⬇️ Download Report (.txt)",
        data=report_summary,
        file_name="churn_report.txt",
        mime="text/plain",
    )
with col_dl2:
    st.download_button(
        label="⬇️ Download High-Risk List (.csv)",
        data=risk_csv,
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )
