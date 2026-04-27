import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

st.set_page_config(layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("Configuration")

DATASETS = {
    "Telco (IBM/Kaggle)": "telco.csv",
    "Bank Customer Churn": "bank_churn.csv",
    "E-commerce Churn": "ecommerce_churn.csv",
}
dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
dataset_file = DATASETS[dataset_name]

penalty = st.sidebar.selectbox("Regularisation", ["l2", "l1"], index=0)
C = st.sidebar.slider("Regularisation strength (C)", 0.01, 10.0, 1.0, step=0.01,
                      help="Smaller = stronger regularisation")

cost_fn = st.sidebar.slider("Cost of Missing Churn (FN)", 10, 500, 100)
cost_fp = st.sidebar.slider("Cost of False Alarm (FP)", 1, 100, 10)

# -----------------------------
# LOAD & PREPROCESS DATA
# -----------------------------
TARGET_COLUMNS = {
    "telco.csv":           ("Churn",      {"Yes": 1, "No": 0}),
    "bank_churn.csv":      ("Exited",     None),
    "ecommerce_churn.csv": ("Churn",      None),
}

DROP_COLUMNS = {
    "telco.csv":           ["customerID"],
    "bank_churn.csv":      ["RowNumber", "CustomerId", "Surname"],
    "ecommerce_churn.csv": ["CustomerID"],
}

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)

    drop_cols = DROP_COLUMNS.get(filepath, [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    target_col, mapping = TARGET_COLUMNS.get(filepath, ("Churn", None))

    if df[target_col].dtype == object and mapping:
        df[target_col] = df[target_col].map(mapping)

    # Coerce any object-dtype numeric columns (e.g. TotalCharges in telco)
    for col in df.select_dtypes(include="object").columns:
        if col != target_col:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() > 0.8:
                df[col] = coerced

    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df, target_col

try:
    df, target_col = load_data(dataset_file)
except FileNotFoundError:
    st.error(
        f"**{dataset_file}** not found. Place the CSV in the same directory as this script."
    )
    st.stop()

# -----------------------------
# SPLIT + SCALE
# -----------------------------
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------
# MODEL — with selected penalty
# -----------------------------
@st.cache_resource
def train_model(penalty, C):
    base_model = LogisticRegression(
        solver="liblinear", penalty=penalty, C=C, random_state=42, max_iter=1000
    )
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    calibrated.fit(X_train_scaled, y_train)
    return calibrated

model = train_model(penalty, C)

# Extract the single fitted base estimator for SHAP / odds ratios
# CalibratedClassifierCV with cv=5 produces 5 calibrated classifiers;
# we average coefs across them for a stable feature-importance estimate.
fitted_estimators = [cc.estimator for cc in model.calibrated_classifiers_]

# -----------------------------
# THRESHOLD OPTIMISATION
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
tp = int(np.sum((final_preds == 1) & (y_test == 1)))
fp_ = int(np.sum((final_preds == 1) & (y_test == 0)))
fn_ = int(np.sum((final_preds == 0) & (y_test == 1)))
tn = int(np.sum((final_preds == 0) & (y_test == 0)))

precision_val = tp / (tp + fp_) if (tp + fp_) > 0 else 0.0
recall_val    = tp / (tp + fn_) if (tp + fn_) > 0 else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("ROC AUC",           f"{auc:.3f}")
col2.metric("Optimal Threshold", f"{best_thresh:.2f}")
col3.metric("Min Cost",          f"{int(min_cost):,}")
col4.metric("Precision",         f"{precision_val:.3f}")
col5.metric("Recall",            f"{recall_val:.3f}")

# -----------------------------
# REGULARISATION COMPARISON
# -----------------------------
with st.expander("⚖️ L1 vs L2 Regularisation Comparison"):
    st.caption(
        "Trains both penalties at the current C and compares AUC on the test set."
    )
    results = {}
    for pen in ["l1", "l2"]:
        m = LogisticRegression(
            solver="liblinear", penalty=pen, C=C, random_state=42, max_iter=1000
        )
        cm = CalibratedClassifierCV(m, method="sigmoid", cv=5)
        cm.fit(X_train_scaled, y_train)
        p = cm.predict_proba(X_test_scaled)[:, 1]
        results[pen] = round(roc_auc_score(y_test, p), 4)

    comp_df = pd.DataFrame(
        {"Penalty": list(results.keys()), "ROC AUC": list(results.values())}
    )
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

# -----------------------------
# ROC CURVE
# -----------------------------
col_roc, col_pr = st.columns(2)

with col_roc:
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

# -----------------------------
# PR CURVE
# -----------------------------
with col_pr:
    precision_c, recall_c, _ = precision_recall_curve(y_test, probs)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(recall_c, precision_c)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    st.pyplot(fig2)
    plt.close(fig2)

# -----------------------------
# FEATURE IMPORTANCE (Odds Ratio)
# FIX: average coefs across all CV folds instead of re-fitting
# -----------------------------
st.header("📌 Feature Importance (Odds Ratio)")

avg_coef = np.mean([est.coef_[0] for est in fitted_estimators], axis=0)
odds = np.exp(avg_coef)

feature_names = X.columns.tolist()
importance_df = (
    pd.DataFrame({"Feature": feature_names, "OddsRatio": odds})
    .sort_values("OddsRatio", ascending=False)
    .head(15)
)

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.barh(importance_df["Feature"][::-1], importance_df["OddsRatio"][::-1])
ax3.axvline(1.0, color="red", linestyle="--", linewidth=0.8, label="Odds = 1 (no effect)")
ax3.set_xlabel("Odds Ratio")
ax3.set_title("Top Feature Importance")
ax3.legend()
st.pyplot(fig3)
plt.close(fig3)

# -----------------------------
# SHAP EXPLAINER
# FIX: use one of the fitted base estimators, not a re-trained model
# -----------------------------
@st.cache_resource
def build_explainer():
    ref_estimator = fitted_estimators[0]
    explainer = shap.LinearExplainer(ref_estimator, X_train_scaled)
    return explainer

explainer = build_explainer()

# -----------------------------
# PREDICTION UI
# -----------------------------
st.header("🔍 Predict Customer Churn")

st.caption("Adjust any feature below — values not shown default to the training mean.")

cols_per_row = 3
input_cols = X.columns.tolist()
n_display = min(15, len(input_cols))     # show up to 15 features in the UI

input_data = {}
rows = [st.columns(cols_per_row) for _ in range(-(-n_display // cols_per_row))]
for i, col_name in enumerate(input_cols[:n_display]):
    with rows[i // cols_per_row][i % cols_per_row]:
        input_data[col_name] = st.number_input(
            col_name, value=float(X[col_name].mean()), key=f"inp_{col_name}"
        )

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    # Fill ALL columns with training means — not zero — for features hidden from the UI
    full_input = pd.DataFrame([X.mean().to_dict()])
    full_input.update(input_df)
    full_input = full_input.reindex(columns=X.columns)

    input_scaled = scaler.transform(full_input)

    prob  = model.predict_proba(input_scaled)[0][1]
    pred  = int(prob >= best_thresh)

    shap_values = explainer.shap_values(input_scaled)[0]
    top_idx      = np.argsort(np.abs(shap_values))[-3:][::-1]
    top_features = [(feature_names[i], round(float(shap_values[i]), 4)) for i in top_idx]

    st.subheader("Result")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Churn Probability", f"{prob:.3f}")
    res_col2.metric(
        "Prediction",
        "🔴 Churn" if pred else "🟢 No Churn",
        help=f"Threshold: {best_thresh:.2f}",
    )

    st.markdown("**Top 3 risk factors (SHAP log-odds contribution):**")
    for feat, val in top_features:
        direction = "↑ increases" if val > 0 else "↓ decreases"
        st.write(f"- **{feat}**: {val:+.4f}  ({direction} churn probability)")

# -----------------------------
# HIGH RISK CUSTOMERS
# -----------------------------
st.header("🚨 High Risk Customers")

df_test = X_test.copy()
df_test["Churn_Prob"]       = probs
df_test["Predicted_Churn"]  = final_preds
df_test["Actual_Churn"]     = y_test.values

high_risk = (
    df_test[df_test["Churn_Prob"] > best_thresh]
    .sort_values("Churn_Prob", ascending=False)
)

st.caption(
    f"{len(high_risk)} customers above threshold {best_thresh:.2f}. "
    f"Showing top 20."
)

display_cols = ["Churn_Prob", "Predicted_Churn", "Actual_Churn"] + feature_names[:5]
st.dataframe(
    high_risk[display_cols].head(20).style.format({"Churn_Prob": "{:.3f}"}),
    use_container_width=True,
)

# -----------------------------
# EARLY-WARNING REPORT EXPORT
# -----------------------------
st.header("📋 Early-Warning Report")

@st.cache_data
def generate_report(threshold, auc_score, min_c, precision_v, recall_v):
    summary = f"""CHURN EARLY-WARNING REPORT
===========================
Dataset     : {dataset_name}
Regulariser : {penalty.upper()}  (C={C})
Threshold   : {threshold:.2f}

PERFORMANCE
-----------
ROC AUC     : {auc_score:.4f}
Precision   : {precision_v:.4f}
Recall      : {recall_v:.4f}
Min Cost    : {int(min_c):,}

HIGH-RISK CUSTOMERS: {len(high_risk)}
"""
    top_risk_csv = high_risk[["Churn_Prob", "Predicted_Churn", "Actual_Churn"]].head(50).to_csv()
    return summary, top_risk_csv

report_summary, risk_csv = generate_report(
    best_thresh, auc, min_cost, precision_val, recall_val
)

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
