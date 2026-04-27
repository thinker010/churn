import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import graphviz

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MedDT — Clinical Decision Tree System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.main { background-color: #0d1117; }

.stApp { background: #0d1117; }

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: #58a6ff;
    letter-spacing: -0.5px;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 8px 0;
}

.metric-card .label {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-card .value {
    font-size: 28px;
    font-weight: 700;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-card .sub {
    font-size: 12px;
    color: #8b949e;
}

.diagnosis-benign {
    background: #0d2818;
    border: 2px solid #3fb950;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}

.diagnosis-malignant {
    background: #2d1117;
    border: 2px solid #f85149;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}

.diagnosis-positive {
    background: #2d1117;
    border: 2px solid #f85149;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}

.diagnosis-negative {
    background: #0d2818;
    border: 2px solid #3fb950;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}

.diagnosis-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 32px;
    font-weight: 700;
}

.path-step {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 4px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
}

.path-step.decisive {
    border-left: 3px solid #f0883e;
    color: #f0883e;
}

.rule-block {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    white-space: pre-wrap;
    overflow-x: auto;
    color: #79c0ff;
    line-height: 1.8;
}

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

.algo-badge {
    display: inline-block;
    background: #1f6feb;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 12px;
    margin-right: 6px;
}

.warning-box {
    background: #2d1f00;
    border: 1px solid #f0883e;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    color: #f0883e;
}

.info-box {
    background: #0c2d6b;
    border: 1px solid #58a6ff;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    color: #79c0ff;
}

div[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

div[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

.stSelectbox > div, .stSlider > div {
    background: #21262d;
}

.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-bottom: 1px solid #30363d;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #8b949e;
}

.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff;
}

.header-bar {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────

@st.cache_data
def load_breast_cancer_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, list(data.feature_names), ['Malignant', 'Benign'], data

@st.cache_data
def load_pima_diabetes():
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    try:
        df = pd.read_csv(url, header=None, names=cols)
        # Replace zeros with NaN for physiologically impossible values
        for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
            df[col] = df[col].replace(0, np.nan)
        df.fillna(df.median(), inplace=True)
        feature_names = cols[:-1]
        return df, feature_names, ['No Diabetes', 'Diabetes'], None
    except:
        # Fallback synthetic data if URL unavailable
        np.random.seed(42)
        n = 768
        df = pd.DataFrame({
            'Pregnancies': np.random.randint(0, 17, n),
            'Glucose': np.random.normal(120, 32, n).clip(44, 199),
            'BloodPressure': np.random.normal(69, 19, n).clip(24, 122),
            'SkinThickness': np.random.normal(20, 16, n).clip(0, 99),
            'Insulin': np.random.normal(79, 115, n).clip(0, 846),
            'BMI': np.random.normal(32, 8, n).clip(18, 67),
            'DiabetesPedigreeFunction': np.random.exponential(0.47, n).clip(0.08, 2.42),
            'Age': np.random.randint(21, 81, n),
        })
        df['Outcome'] = ((df['Glucose'] > 140) | (df['BMI'] > 35)).astype(int)
        feature_names = list(df.columns)
        df['target'] = df['Outcome']
        return df, feature_names, ['No Diabetes', 'Diabetes'], None

@st.cache_data
def load_heart_disease():
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal','target']
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    try:
        df = pd.read_csv(url, header=None, names=cols, na_values='?')
        df.fillna(df.median(), inplace=True)
        df['target'] = (df['target'] > 0).astype(int)
        feature_names = cols[:-1]
        return df, feature_names, ['No Disease', 'Heart Disease'], None
    except:
        np.random.seed(0)
        n = 303
        df = pd.DataFrame({
            'age': np.random.randint(29, 77, n),
            'sex': np.random.randint(0, 2, n),
            'cp': np.random.randint(0, 4, n),
            'trestbps': np.random.normal(131, 18, n).clip(94, 200),
            'chol': np.random.normal(246, 52, n).clip(126, 564),
            'fbs': np.random.randint(0, 2, n),
            'restecg': np.random.randint(0, 3, n),
            'thalach': np.random.normal(149, 23, n).clip(71, 202),
            'exang': np.random.randint(0, 2, n),
            'oldpeak': np.abs(np.random.normal(1.0, 1.2, n)),
            'slope': np.random.randint(0, 3, n),
            'ca': np.random.randint(0, 4, n),
            'thal': np.random.choice([3, 6, 7], n),
        })
        df['target'] = ((df['age'] > 55) & (df['chol'] > 240)).astype(int)
        feature_names = list(df.columns[:-1])
        return df, feature_names, ['No Disease', 'Heart Disease'], None

# ─────────────────────────────────────────────
# ALGORITHM HELPERS
# ─────────────────────────────────────────────

def get_criterion_and_splitter(algo):
    """Map algorithm name to sklearn parameters."""
    mapping = {
        "CART (Gini)":      ("gini",    "best"),
        "CART (MSE)":       ("gini",    "best"),  # classification uses gini
        "ID3 (Info Gain)":  ("entropy", "best"),
        "C4.5 (Info Gain)": ("entropy", "best"),
    }
    return mapping.get(algo, ("gini", "best"))

def build_model(algo, max_depth, min_samples_leaf, ccp_alpha=0.0):
    criterion, splitter = get_criterion_and_splitter(algo)
    return DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=42
    )

def extract_if_then_rules(tree, feature_names, class_names):
    """Extract human-readable IF-THEN rules from a decision tree."""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, depth, conditions):
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], depth + 1,
                    conditions + [f"{name} ≤ {threshold:.3f}"])
            recurse(tree_.children_right[node], depth + 1,
                    conditions + [f"{name} > {threshold:.3f}"])
        else:
            class_idx = np.argmax(tree_.value[node])
            label = class_names[class_idx]
            support = int(tree_.n_node_samples[node])
            conf = tree_.value[node][0][class_idx] / support * 100
            rule = "IF " + "\n   AND ".join(conditions)
            rule += f"\nTHEN → {label}  [support={support}, confidence={conf:.1f}%]\n"
            rules.append(rule)

    recurse(0, 1, [])
    return rules

def get_decision_path_explanation(model, user_input, feature_names, class_names):
    """Detailed decision path for a single prediction."""
    node_indicator = model.decision_path(user_input)
    leaf_id = model.apply(user_input)[0]
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    node_index = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]
    steps = []
    for node_id in node_index:
        if feature[node_id] != -2:
            fname = feature_names[feature[node_id]]
            val = user_input[0, feature[node_id]]
            thr = threshold[node_id]
            direction = "≤" if val <= thr else ">"
            went = "LEFT ↙" if val <= thr else "RIGHT ↘"
            steps.append({
                "feature": fname,
                "value": val,
                "threshold": thr,
                "direction": direction,
                "went": went,
                "decisive": False
            })
    if steps:
        steps[-1]["decisive"] = True
    return steps

# ─────────────────────────────────────────────
# PLOTTING HELPERS  (dark theme)
# ─────────────────────────────────────────────

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
BLUE      = "#58a6ff"
GREEN     = "#3fb950"
RED       = "#f85149"
ORANGE    = "#f0883e"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"

def dark_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    return fig, ax

def plot_confusion_matrix(cm, class_names):
    fig, ax = dark_fig((5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, color=TEXT, fontsize=10)
    ax.set_yticklabels(class_names, color=TEXT, fontsize=10)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=11)
    ax.set_ylabel("Actual", color=MUTED, fontsize=11)
    ax.set_title("Confusion Matrix", color=TEXT, fontsize=12, pad=10)
    total = cm.sum()
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            pct = val / total * 100
            color = "white" if cm[i, j] > cm.max() / 2 else TEXT
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha='center', va='center',
                    color=color, fontsize=10, fontfamily='monospace')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig

def plot_roc_curve(model, X_test, y_test, class_names):
    fig, ax = dark_fig((5, 4))
    proba = model.predict_proba(X_test)
    n_classes = len(class_names)

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=BLUE, lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.fill_between(fpr, tpr, alpha=0.15, color=BLUE)
    else:
        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
        colors = [BLUE, GREEN, ORANGE, RED]
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f'{cls} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], color=BORDER, lw=1.5, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', color=MUTED)
    ax.set_ylabel('True Positive Rate', color=MUTED)
    ax.set_title('ROC-AUC Curve', color=TEXT, fontsize=12)
    ax.legend(loc='lower right', fontsize=9,
              framealpha=0.3, labelcolor=TEXT,
              facecolor=PANEL_BG, edgecolor=BORDER)
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, top_n=12):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    vals = importances[indices]
    names = [feature_names[i] for i in indices]

    fig, ax = dark_fig((6, max(4, top_n * 0.35)))
    colors = [BLUE if v > vals.mean() else MUTED for v in vals]
    bars = ax.barh(names, vals, color=colors, height=0.65, edgecolor=BORDER)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', color=MUTED, fontsize=9,
                fontfamily='monospace')
    ax.set_xlabel("Importance Score", color=MUTED)
    ax.set_title("Feature Importance", color=TEXT, fontsize=12)
    ax.set_xlim(0, vals.max() * 1.2)
    plt.tight_layout()
    return fig

def plot_pruning_curve(X_train, X_test, y_train, y_test, criterion):
    """Post-pruning: test different ccp_alpha values."""
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas)//20)]

    train_scores, test_scores = [], []
    for a in alphas:
        m = DecisionTreeClassifier(criterion=criterion, ccp_alpha=a, random_state=42)
        m.fit(X_train, y_train)
        train_scores.append(m.score(X_train, y_train))
        test_scores.append(m.score(X_test, y_test))

    fig, ax = dark_fig((6, 3.5))
    ax.plot(alphas, train_scores, 'o-', color=BLUE, label='Train', lw=2, ms=4)
    ax.plot(alphas, test_scores, 's-', color=GREEN, label='Test', lw=2, ms=4)
    ax.set_xlabel("CCP Alpha (Pruning Strength)", color=MUTED)
    ax.set_ylabel("Accuracy", color=MUTED)
    ax.set_title("Post-Pruning: Accuracy vs. Alpha", color=TEXT, fontsize=12)
    ax.legend(fontsize=9, framealpha=0.3, labelcolor=TEXT,
              facecolor=PANEL_BG, edgecolor=BORDER)
    plt.tight_layout()
    return fig, alphas, test_scores

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚕️ MedDT Config")
    st.markdown("---")

    dataset_choice = st.selectbox(
        "📁 Dataset",
        ["Breast Cancer Wisconsin", "Pima Indians Diabetes", "Heart Disease UCI"]
    )

    algo_choice = st.selectbox(
        "🧠 Algorithm",
        ["CART (Gini)", "ID3 (Info Gain)", "C4.5 (Info Gain)"],
        help="CART uses Gini impurity; ID3/C4.5 use Information Gain (entropy)."
    )

    st.markdown("#### Tree Hyperparameters")
    max_depth = st.slider("Max Depth", 2, 10, 4)
    min_samples_leaf = st.slider("Min Samples / Leaf", 1, 30, 5)
    test_size = st.slider("Test Split %", 10, 40, 20) / 100

    st.markdown("#### ✂️ Post-Pruning (CCP Alpha)")
    use_pruning = st.checkbox("Enable Cost-Complexity Pruning", value=False)
    ccp_alpha = 0.0
    if use_pruning:
        ccp_alpha = st.slider("CCP Alpha", 0.000, 0.050, 0.005, step=0.001,
                               format="%.3f")
    st.markdown("---")
    st.markdown("#### ℹ️ Algorithm Info")
    algo_info = {
        "CART (Gini)": "CART uses **Gini Impurity** to measure node purity. Produces binary splits at each node.",
        "ID3 (Info Gain)": "ID3 uses **Information Gain** (entropy-based). Original Quinlan 1986 algorithm.",
        "C4.5 (Info Gain)": "C4.5 extends ID3 with **Gain Ratio** normalization to reduce bias toward high-cardinality features.",
    }
    st.info(algo_info[algo_choice])

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

if dataset_choice == "Breast Cancer Wisconsin":
    df, feature_names, class_names, raw = load_breast_cancer_data()
    X = df[feature_names].values
    y = df['target'].values
elif dataset_choice == "Pima Indians Diabetes":
    df, feature_names, class_names, raw = load_pima_diabetes()
    X = df[feature_names].values
    y = df['Outcome'].values if 'Outcome' in df.columns else df['target'].values
else:
    df, feature_names, class_names, raw = load_heart_disease()
    X = df[feature_names].values
    y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

criterion, splitter = get_criterion_and_splitter(algo_choice)

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────

model = build_model(algo_choice, max_depth, min_samples_leaf, ccp_alpha)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

if len(class_names) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc_val = auc(fpr, tpr)
else:
    roc_auc_val = None

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown(f"""
<div class="header-bar">
    <div>
        <h1 style="margin:0; font-size:24px; color:#58a6ff;">
            🏥 Diagnostic Decision Support System
        </h1>
        <div style="color:#8b949e; font-size:13px; margin-top:4px;">
            <span class="algo-badge">{algo_choice}</span>
            <span style="font-family:'IBM Plex Mono',monospace;">{dataset_choice}</span>
            &nbsp;·&nbsp; {len(X_train)} train / {len(X_test)} test samples
            &nbsp;·&nbsp; {len(feature_names)} features
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PERFORMANCE METRICS ROW
# ─────────────────────────────────────────────

st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)

def metric_card(label, value, sub=""):
    return f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>"""

m1.markdown(metric_card("Accuracy", f"{acc:.1%}", f"CV: {cv_scores.mean():.1%} ±{cv_scores.std():.2%}"), unsafe_allow_html=True)
m2.markdown(metric_card("Precision", f"{prec:.1%}", "Weighted avg"), unsafe_allow_html=True)
m3.markdown(metric_card("Recall", f"{rec:.1%}", "Weighted avg"), unsafe_allow_html=True)
m4.markdown(metric_card("F1 Score", f"{f1:.1%}", "Weighted avg"), unsafe_allow_html=True)
if roc_auc_val:
    m5.markdown(metric_card("ROC-AUC", f"{roc_auc_val:.3f}", "Binary classification"), unsafe_allow_html=True)
else:
    m5.markdown(metric_card("Tree Nodes", f"{model.tree_.node_count}", f"Depth: {model.get_depth()}"), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Patient Prediction",
    "📊 Model Evaluation",
    "🌲 Decision Tree",
    "📋 Rule Extraction",
    "✂️ Pruning Analysis"
])

# ══════════════════════════════════════════════
# TAB 1: PATIENT PREDICTION
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Patient Input & Clinician Explanation</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    Enter patient measurements using the sidebar controls, or adjust below.
    The model will generate a prediction with a full step-by-step decision path.
    </div>""", unsafe_allow_html=True)

    # Dynamic patient input form
    n_features = len(feature_names)
    cols_per_row = 4
    user_vals = []

    # Use first test sample as default
    default_sample = X_test[0]

    rows = [feature_names[i:i+cols_per_row] for i in range(0, n_features, cols_per_row)]
    all_inputs = {}

    for row_feats in rows:
        cols = st.columns(len(row_feats))
        for col, fname in zip(cols, row_feats):
            idx = feature_names.index(fname)
            col_vals = X[:, idx]
            dval = float(default_sample[idx])
            with col:
                v = st.number_input(
                    fname,
                    value=round(dval, 4),
                    format="%.4f",
                    key=f"feat_{fname}"
                )
                all_inputs[fname] = v

    user_input = np.array([all_inputs[f] for f in feature_names]).reshape(1, -1)
    prediction = model.predict(user_input)[0]
    pred_proba = model.predict_proba(user_input)[0]
    pred_label = class_names[prediction]

    st.markdown("---")

    # Diagnosis result
    col_diag, col_path = st.columns([1, 2])

    with col_diag:
        st.markdown('<div class="section-header">Diagnosis Result</div>', unsafe_allow_html=True)

        is_positive = prediction == 1 if dataset_choice == "Breast Cancer Wisconsin" else prediction == 1
        if dataset_choice == "Breast Cancer Wisconsin":
            is_alarming = (pred_label == "Malignant")
        else:
            is_alarming = (pred_label in ["Diabetes", "Heart Disease"])

        css_class = "diagnosis-malignant" if is_alarming else "diagnosis-benign"
        color = RED if is_alarming else GREEN
        icon = "⚠️" if is_alarming else "✅"

        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size:40px;">{icon}</div>
            <div class="diagnosis-label" style="color:{color};">{pred_label}</div>
            <div style="color:#8b949e; font-size:13px; margin-top:8px; font-family:monospace;">
                Confidence: {pred_proba[prediction]:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("**Class Probabilities**")
        for i, (cls, prob) in enumerate(zip(class_names, pred_proba)):
            bar_color = RED if i == prediction else MUTED
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin:4px 0;">
                <div style="font-family:monospace; font-size:11px; color:#8b949e; width:140px; overflow:hidden; text-overflow:ellipsis;">{cls}</div>
                <div style="flex:1; background:#21262d; border-radius:3px; height:16px; overflow:hidden;">
                    <div style="width:{prob*100:.1f}%; background:{bar_color}; height:100%; border-radius:3px;"></div>
                </div>
                <div style="font-family:monospace; font-size:11px; color:{bar_color}; width:45px;">{prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_path:
        st.markdown('<div class="section-header">Decision Path — Clinician Report</div>', unsafe_allow_html=True)
        steps = get_decision_path_explanation(model, user_input, feature_names, class_names)

        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:16px; margin-bottom:16px;">
            <div style="font-size:12px; color:#8b949e; font-family:monospace; text-transform:uppercase; letter-spacing:1px;">Case Summary</div>
            <div style="margin-top:8px; font-size:14px; color:#e6edf3;">
                The <strong style="color:#58a6ff;">{algo_choice}</strong> model traversed 
                <strong style="color:#f0883e;">{len(steps)} decision node(s)</strong> 
                and classified this patient as 
                <strong style="color:{color};">{pred_label}</strong> 
                with <strong>{pred_proba[prediction]:.1%}</strong> confidence.
            </div>
        </div>
        """, unsafe_allow_html=True)

        for i, step in enumerate(steps):
            css_class = "path-step decisive" if step["decisive"] else "path-step"
            badge = "🔑 DECISIVE SPLIT" if step["decisive"] else f"Step {i+1}"
            st.markdown(f"""
            <div class="{css_class}">
                <span style="color:#8b949e; font-size:10px;">{badge}</span><br/>
                <strong style="color:#79c0ff;">{step['feature']}</strong> = 
                <span style="color:#f0883e;">{step['value']:.4f}</span>
                &nbsp;{step['direction']}&nbsp;
                threshold <span style="color:#e6edf3;">{step['threshold']:.4f}</span>
                &nbsp;&nbsp;→&nbsp;&nbsp;
                <span style="color:#3fb950;">{step['went']}</span>
            </div>
            """, unsafe_allow_html=True)

        if steps:
            decisive = steps[-1]
            st.markdown(f"""
            <div class="warning-box" style="margin-top:12px;">
                🔑 <strong>Key Factor:</strong> The final classification was determined by 
                <strong>{decisive['feature']}</strong> = {decisive['value']:.4f} 
                (threshold: {decisive['threshold']:.4f})
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2: MODEL EVALUATION
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2)

    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = plot_confusion_matrix(cm, class_names)
        st.pyplot(fig_cm)

    with col_roc:
        fig_roc = plot_roc_curve(model, X_test, y_test, class_names)
        st.pyplot(fig_roc)

    st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(4)

    st.markdown(f"""
    <div class="rule-block">
{classification_report(y_test, y_pred, target_names=class_names)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Cross-Validation (5-Fold)</div>', unsafe_allow_html=True)
    cv_cols = st.columns(5)
    for i, (col, score) in enumerate(zip(cv_cols, cv_scores)):
        col.markdown(metric_card(f"Fold {i+1}", f"{score:.1%}", ""), unsafe_allow_html=True)

    col_fi, _ = st.columns([2, 1])
    with col_fi:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fig_fi = plot_feature_importance(model, feature_names)
        st.pyplot(fig_fi)

# ══════════════════════════════════════════════
# TAB 3: DECISION TREE VISUALIZATION
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Visual Decision Tree</div>', unsafe_allow_html=True)

    max_viz_depth = st.slider("Visualization Depth", 1, min(max_depth, 6), min(max_depth, 4),
                               help="Limit displayed depth for readability")

    # Rebuild a shallow copy for display if needed
    viz_model = model
    if max_viz_depth < max_depth:
        viz_model = build_model(algo_choice, max_viz_depth, min_samples_leaf, ccp_alpha)
        viz_model.fit(X_train, y_train)

    dot_data = export_graphviz(
        viz_model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        impurity=True,
        proportion=False
    )

    st.graphviz_chart(dot_data, use_container_width=True)

    col_stats1, col_stats2, col_stats3 = st.columns(3)
    col_stats1.markdown(metric_card("Tree Depth", model.get_depth(), "actual model"), unsafe_allow_html=True)
    col_stats2.markdown(metric_card("Total Nodes", model.tree_.node_count, "incl. leaves"), unsafe_allow_html=True)
    col_stats3.markdown(metric_card("Leaf Nodes", model.get_n_leaves(), "terminal nodes"), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    <strong>How to read:</strong> Each node shows the split condition, 
    {'Gini impurity' if criterion == 'gini' else 'Entropy'} value, 
    sample count, and class distribution. 
    <strong>Blue nodes</strong> = {class_names[1]}, 
    <strong>Orange nodes</strong> = {class_names[0]}.
    Darker color = higher purity.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4: RULE EXTRACTION
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Extracted IF-THEN Rules</div>', unsafe_allow_html=True)

    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        rules = extract_if_then_rules(model, feature_names, class_names)
        st.markdown(f"""
        <div class="info-box">
        <strong>{len(rules)} rules</strong> extracted from the decision tree. 
        Each rule is a root-to-leaf path with support count and confidence %.
        These rules can be directly handed to a clinician for review.
        </div>
        """, unsafe_allow_html=True)

        for i, rule in enumerate(rules, 1):
            with st.expander(f"Rule {i:02d} — {rule.split('THEN → ')[1].split('[')[0].strip()}"):
                st.markdown(f'<div class="rule-block">{rule}</div>', unsafe_allow_html=True)

    with col_r2:
        st.markdown('<div class="section-header">Sklearn Rule Text</div>', unsafe_allow_html=True)
        text_rules = export_text(model, feature_names=list(feature_names))
        st.markdown(f'<div class="rule-block">{text_rules}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Export Rules</div>', unsafe_allow_html=True)
    all_rules_text = "\n" + "="*60 + "\n"
    all_rules_text += f"DECISION RULES — {dataset_choice} — {algo_choice}\n"
    all_rules_text += "="*60 + "\n\n"
    for i, rule in enumerate(rules, 1):
        all_rules_text += f"RULE {i:02d}:\n{rule}\n{'-'*40}\n"

    st.download_button(
        label="⬇️ Download IF-THEN Rules (.txt)",
        data=all_rules_text,
        file_name=f"rules_{dataset_choice.replace(' ', '_')}_{algo_choice.split()[0]}.txt",
        mime="text/plain"
    )

# ══════════════════════════════════════════════
# TAB 5: PRUNING ANALYSIS
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Pre-Pruning vs. Post-Pruning</div>', unsafe_allow_html=True)

    col_pre, col_post = st.columns(2)

    with col_pre:
        st.markdown("#### Pre-Pruning: Depth vs. Accuracy")
        depths = list(range(1, 15))
        train_accs, test_accs = [], []
        for d in depths:
            m = build_model(algo_choice, d, min_samples_leaf, 0.0)
            m.fit(X_train, y_train)
            train_accs.append(m.score(X_train, y_train))
            test_accs.append(m.score(X_test, y_test))

        fig_pre, ax_pre = dark_fig((6, 3.5))
        ax_pre.plot(depths, train_accs, 'o-', color=BLUE, label='Train', lw=2, ms=5)
        ax_pre.plot(depths, test_accs, 's-', color=GREEN, label='Test', lw=2, ms=5)
        ax_pre.axvline(x=max_depth, color=ORANGE, linestyle='--', lw=1.5,
                       label=f'Current depth={max_depth}')
        ax_pre.set_xlabel("Max Depth", color=MUTED)
        ax_pre.set_ylabel("Accuracy", color=MUTED)
        ax_pre.set_title("Pre-Pruning Analysis", color=TEXT, fontsize=12)
        ax_pre.legend(fontsize=9, framealpha=0.3, labelcolor=TEXT,
                      facecolor=PANEL_BG, edgecolor=BORDER)
        plt.tight_layout()
        st.pyplot(fig_pre)

        best_depth = depths[np.argmax(test_accs)]
        st.markdown(f"""
        <div class="info-box">
        Optimal depth by test accuracy: <strong>depth = {best_depth}</strong> 
        ({max(test_accs):.1%} accuracy). 
        Deeper trees overfit — test accuracy plateaus or drops.
        </div>
        """, unsafe_allow_html=True)

    with col_post:
        st.markdown("#### Post-Pruning: CCP Alpha vs. Accuracy")
        fig_post, alphas, post_scores = plot_pruning_curve(
            X_train, X_test, y_train, y_test, criterion
        )
        st.pyplot(fig_post)

        best_alpha = alphas[np.argmax(post_scores)]
        st.markdown(f"""
        <div class="info-box">
        Optimal CCP alpha: <strong>{best_alpha:.4f}</strong> 
        ({max(post_scores):.1%} test accuracy).
        Higher alpha = more pruning = simpler tree.
        Enable pruning in the sidebar to apply.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Algorithm Comparison on this Dataset</div>', unsafe_allow_html=True)

    algo_results = {}
    for algo_name, crit in [("CART (Gini)", "gini"), ("ID3 / C4.5 (Entropy)", "entropy")]:
        m = DecisionTreeClassifier(criterion=crit, max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf, random_state=42)
        m.fit(X_train, y_train)
        yp = m.predict(X_test)
        algo_results[algo_name] = {
            "Accuracy": accuracy_score(y_test, yp),
            "Precision": precision_score(y_test, yp, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, yp, average='weighted', zero_division=0),
            "F1": f1_score(y_test, yp, average='weighted', zero_division=0),
            "Nodes": m.tree_.node_count,
            "Depth": m.get_depth(),
        }

    comp_df = pd.DataFrame(algo_results).T
    comp_df = comp_df.round(4)

    # Color-format the table
    st.dataframe(
        comp_df.style.background_gradient(cmap='Blues', subset=['Accuracy', 'F1'])
                     .format("{:.4f}", subset=['Accuracy', 'Precision', 'Recall', 'F1'])
                     .format("{:.0f}", subset=['Nodes', 'Depth']),
        use_container_width=True
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8b949e; font-family:'IBM Plex Mono',monospace; font-size:11px; padding:16px;">
    MedDT Clinical Decision Support &nbsp;|&nbsp; 
    Algorithms: ID3 · C4.5 · CART &nbsp;|&nbsp; 
    Datasets: Breast Cancer Wisconsin · Pima Indians Diabetes · Heart Disease UCI<br/>
    <span style="color:#30363d;">⚠️ For research and educational purposes only. Not for clinical use.</span>
</div>
""", unsafe_allow_html=True)