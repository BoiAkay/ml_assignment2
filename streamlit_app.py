import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="BeanClassify AI", layout="wide", page_icon="🫘")

# Updated CSS for a Teal/Blue theme
st.markdown("""
    <style>
    .main { background-color: #F0F2F6; font-family: 'Helvetica', sans-serif; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3em; 
        background-color: #008080; color: white; font-weight: 600; 
    }
    .stButton>button:hover { background-color: #006666; }
    .metric-container { 
        background-color: white; padding: 15px; border-radius: 10px; 
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05); text-align: center; 
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. GLOBAL CONSTANTS
# ---------------------------------------------------------
BEAN_TYPES = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']
INPUT_FEATURES = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
    'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
    'Roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3',
    'ShapeFactor4'
]

# Initialize Session State
if 'test_features' not in st.session_state:
    st.session_state['test_features'] = None
if 'test_targets' not in st.session_state:
    st.session_state['test_targets'] = None

# ---------------------------------------------------------
# 3. UTILITY FUNCTIONS
# ---------------------------------------------------------
def get_trained_model(selection):
    """Maps display names to file paths and loads model."""
    model_map = {
        "Logistic Regression": "log_reg.pkl",
        "Decision Tree": "dt_clf.pkl",
        "K-Nearest Neighbors": "knn.pkl",
        "Naive Bayes (Gaussian)": "gnb.pkl",
        "Random Forest": "rf_clf.pkl",
        "XGBoost": "xgb.pkl"
    }
    
    f_path = os.path.join('model', model_map.get(selection, ""))
    if os.path.exists(f_path):
        return joblib.load(f_path)
    return None

def compute_metrics(clf, x_data, y_data):
    """Calculates evaluation metrics for the given model."""
    preds = clf.predict(x_data)
    
    # Calculate AUC if probability is supported
    auc_score = 0.0
    try:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(x_data)
            auc_score = roc_auc_score(y_data, probs, multi_class='ovr', average='weighted', labels=clf.classes_)
    except:
        pass
        
    return {
        "Accuracy": accuracy_score(y_data, preds),
        "AUC": auc_score,
        "Precision": precision_score(y_data, preds, average='weighted'),
        "Recall": recall_score(y_data, preds, average='weighted'),
        "F1": f1_score(y_data, preds, average='weighted'),
        "MCC": matthews_corrcoef(y_data, preds),
        "predictions": preds
    }

# ---------------------------------------------------------
# 4. SIDEBAR SETUP
# ---------------------------------------------------------
st.sidebar.title("🫘 Menu")
app_mode = st.sidebar.radio("Navigate:", ["Classification Tool", "Model Insights"])

if app_mode == "Classification Tool":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    selected_algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        ("Logistic Regression", "Decision Tree", "K-Nearest Neighbors", 
         "Naive Bayes (Gaussian)", "Random Forest", "XGBoost")
    )

# ---------------------------------------------------------
# 5. PAGE: CLASSIFICATION TOOL
# ---------------------------------------------------------
if app_mode == "Classification Tool":
    st.title("Dry Bean Classification System")
    st.markdown("---")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("1. Input Data")
        st.info("Upload `test_data_scaled.csv` containing feature columns.")
        data_file = st.file_uploader("Drop CSV Here", type=["csv"])

        if data_file:
            try:
                raw_df = pd.read_csv(data_file)
                
                if 'Target_Class_ID' in raw_df.columns:
                    st.session_state['test_targets'] = raw_df['Target_Class_ID'].values
                    st.session_state['test_features'] = raw_df.drop(
                        columns=['Target_Class_ID', 'Target_Class_Name'], errors='ignore'
                    ).values
                    st.success(f"✅ Loaded {len(raw_df)} samples.")
                else:
                    st.warning("⚠️ No Target column found. Metrics will be unavailable.")
                    st.session_state['test_features'] = raw_df.values
                    st.session_state['test_targets'] = None
                    
            except Exception as err:
                st.error(f"File Error: {err}")

    with right_col:
        if st.session_state['test_features'] is not None:
            st.subheader("2. Analysis Results")
            active_model = get_trained_model(selected_algorithm)
            
            if active_model:
                X_in = st.session_state['test_features']
                y_true = st.session_state['test_targets']
                
                try:
                    # Generate Predictions
                    pred_indices = active_model.predict(X_in)
                    pred_names = [BEAN_TYPES[i] for i in pred_indices]
                    
                    # Display Data Table
                    display_df = pd.DataFrame(X_in, columns=INPUT_FEATURES)
                    display_df['Predicted Label'] = pred_names
                    
                    if y_true is not None:
                        display_df['Actual Label'] = [BEAN_TYPES[i] for i in y_true]
                    
                    st.dataframe(display_df.head(8), use_container_width=True)
                    
                    # Display Metrics if targets exist
                    if y_true is not None:
                        st.markdown("### Performance Metrics")
                        scores = compute_metrics(active_model, X_in, y_true)
                        
                        # --- UPDATED SECTION: 6 Columns to include Recall ---
                        c1, c2, c3, c4, c5, c6 = st.columns(6)
                        c1.metric("Accuracy", f"{scores['Accuracy']:.3f}")
                        c2.metric("AUC", f"{scores['AUC']:.3f}")
                        c3.metric("F1-Score", f"{scores['F1']:.3f}")
                        c4.metric("Precision", f"{scores['Precision']:.3f}")
                        c5.metric("Recall", f"{scores['Recall']:.3f}")  # <--- Added Recall
                        c6.metric("MCC", f"{scores['MCC']:.3f}")
                        # -------------------------------------------------
                        
                        # Confusion Matrix
                        st.write("#### Confusion Matrix Heatmap")
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                        cm_data = confusion_matrix(y_true, scores['predictions'])
                        
                        sns.heatmap(
                            cm_data, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=BEAN_TYPES, yticklabels=BEAN_TYPES
                        )
                        plt.xlabel('Predicted Class')
                        plt.ylabel('True Class')
                        st.pyplot(fig_cm)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                st.error(f"Model file for '{selected_algorithm}' not found. Run training script first.")

# ---------------------------------------------------------
# 6. PAGE: MODEL INSIGHTS
# ---------------------------------------------------------
else:
    st.title("📊 Model Benchmark & Insights")
    
    CSV_PATH = 'bean_model_performance.csv'
    
    if os.path.exists(CSV_PATH):
        st.subheader("Algorithm Leaderboard")
        df_results = pd.read_csv(CSV_PATH)
        
        # Adjust index to start at 1
        df_results.index = df_results.index + 1

        # Format columns
        st.table(df_results.style.format({
            "Accuracy": "{:.3f}", "AUC": "{:.3f}", "Precision": "{:.3f}",
            "Recall": "{:.3f}", "F1-Score": "{:.3f}", "MCC": "{:.3f}"
        }))
        
    else:
        st.warning("⚠️ Benchmark file not found. Please run `training_pipeline.py`.")
            
    st.markdown("---")
    st.subheader("Key Findings")
    
    insights = {
        "Logistic Regression": "Serves as a solid baseline (~92%). It struggles slightly with distinguishing between Sira and Dermason due to their linear inseparability.",
        "Decision Tree": "Offers high interpretability but tends to overfit on noise, leading to lower generalization accuracy (~89%) compared to ensembles.",
        "K-Nearest Neighbors": "Performs surprisingly well given that the bean classes form distinct, dense clusters in the feature space.",
        "Naive Bayes": "Shows the weakest performance. The strong correlation between geometric features (e.g., ConvexArea vs Area) violates the algorithm's independence assumption.",
        "Random Forest": "Achieves high accuracy (~93%) by aggregating multiple decision trees, which effectively reduces variance and overfitting.",
        "XGBoost": "Currently the top performer. Gradient boosting effectively optimizes the decision boundary for hard-to-classify edge cases."
    }
    
    insight_df = pd.DataFrame([
        {"Algorithm": k, "Analysis": v} for k, v in insights.items()
    ])
    insight_df.index = insight_df.index + 1
    
    st.table(insight_df)

    st.markdown("---")
    st.subheader("Feature Correlation (Test Data)")
    
    img_path = "test_data_correlation.png"
    if os.path.exists(img_path):
        st.image(img_path, caption="Correlation: Scaled Test Features & Target", use_column_width=True)
    else:
        st.warning("Correlation heatmap missing.")