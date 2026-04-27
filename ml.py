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
# SIDEBAR
# -----------------------------
st.sidebar.header("Configuration")

DATASETS = {
    "Telco (IBM/Kaggle)": "telco.csv",
    "Bank Customer Churn": "bank_churn.csv"
}

dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
dataset_file = DATASETS[dataset_name]

penalty = st.sidebar.selectbox("Regularisation", ["l2", "l1"])
C = st.sidebar.slider("C", 0.01, 10.0, 1.0)

cost_fn = st.sidebar.slider("Cost FN", 10, 500, 100)
cost_fp = st.sidebar.slider("Cost FP", 1, 100, 10)

# -----------------------------
# LOAD DATA (FIXED)
# -----------------------------
@st.cache_data
def load_data(filepath):

    # ✅ Encoding-safe loading
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except:
        try:
            df = pd.read_csv(filepath, encoding="latin1")
        except:
            df = pd.read_csv(filepath, encoding="cp1252")

    # Drop unwanted columns
    drop_cols = ["customerID", "CustomerId", "RowNumber", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Detect target column
    possible_targets = ["Churn", "Exited"]
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        st.error(f"Target column not found! Columns: {df.columns.tolist()}")
        st.stop()

    # Convert target BEFORE encoding
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

    # Convert numeric-like columns
    for col in df.columns:
        if col != target_col and df[col].dtype == object:
            temp = pd.to_numeric(df[col], errors="coerce")
            if temp.notna().mean() > 0.8:
                df[col] = temp

    df = df.dropna()

    # Split BEFORE encoding
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode features
    X = pd.get_dummies(X, drop_first=True)

    return X, y

X, y = load_data(dataset_file)

# -----------------------------
# SPLIT + SCALE
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# MODEL
# -----------------------------
base_model = LogisticRegression(solver="liblinear", penalty=penalty, C=C)
model = CalibratedClassifierCV(base_model)
model.fit(X_train_scaled, y_train)

# -----------------------------
# THRESHOLD OPTIMIZATION
# -----------------------------
probs = model.predict_proba(X_test_scaled)[:, 1]

best_thresh = 0.5
min_cost = float("inf")

for t in np.linspace(0, 1, 100):
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

col1, col2, col3 = st.columns(3)
col1.metric("AUC", round(auc, 3))
col2.metric("Threshold", round(best_thresh, 2))
col3.metric("Min Cost", int(min_cost))

# -----------------------------
# ROC CURVE
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, probs)
fig = plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
st.pyplot(fig)

# -----------------------------
# SHAP
# -----------------------------
explainer = shap.LinearExplainer(base_model.fit(X_train_scaled, y_train), X_train_scaled)

# -----------------------------
# PREDICTION UI
# -----------------------------
st.header("🔍 Predict")

input_data = {}
for col in X.columns[:10]:
    input_data[col] = st.number_input(col, float(X[col].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    full = pd.DataFrame([X.mean()])
    full.update(input_df)

    input_scaled = scaler.transform(full)

    prob = model.predict_proba(input_scaled)[0][1]
    pred = int(prob >= best_thresh)

    shap_vals = explainer.shap_values(input_scaled)[0]
    top_idx = np.argsort(np.abs(shap_vals))[-3:]

    st.write("Probability:", round(prob, 3))
    st.write("Prediction:", "Churn" if pred else "No Churn")
    st.write("Top Factors:", [X.columns[i] for i in top_idx])
