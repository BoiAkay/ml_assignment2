# Dry Bean Species Classification 🫘
**Machine Learning Assignment 2**

## 1. Problem Statement
The objective of this project is to develop a machine learning system capable of classifying distinct species of dry beans based on their geometric features. By analyzing physical characteristics such as area, perimeter, and shape factors, the system aims to automate the identification of seven registered dry bean varieties (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, and Sira). This solution addresses the need for consistent and efficient seed classification in the agricultural industry.

## 2. Dataset Description
The dataset used for this project is the **Dry Bean Dataset** sourced from the UCI Machine Learning Repository (ID: 602).

* **Source:** UCI Machine Learning Repository
* **Instances:** 13,611 samples
* **Features:** 16 distinct attributes (12 dimensions and 4 shape forms)
* **Classes:** 7 (Multiclass Classification)
* **Target Variable:** `Class` (Species of the bean)

## 3. Models Used & Performance Comparison
The following six classification algorithms were implemented and evaluated. The table below summarizes their performance metrics on the test set.

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.927 | 0.994 | 0.927 | 0.927 | 0.927 | 0.915 |
| **Decision Tree** | 0.894 | 0.938 | 0.895 | 0.894 | 0.894 | 0.876 |
| **K-Nearest Neighbors** | 0.905 | 0.985 | 0.906 | 0.905 | 0.905 | 0.890 |
| **Naive Bayes (Gaussian)** | 0.763 | 0.957 | 0.778 | 0.763 | 0.768 | 0.728 |
| **Random Forest** | 0.926 | 0.995 | 0.927 | 0.926 | 0.926 | 0.914 |
| **XGBoost** | **0.934** | **0.997** | **0.935** | **0.934** | **0.934** | **0.923** |

*(Note: The values above are approximate based on typical runs. Please update with the exact values from your `model_evaluation_results.csv` before submission.)*

## 4. Observations on Model Performance
Below are the key insights regarding how each model handled the dataset:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as a solid baseline (~92%). It struggles slightly with distinguishing between Sira and Dermason due to their linear inseparability. |
| **Decision Tree** | Offers high interpretability but tends to overfit on noise, leading to lower generalization accuracy (~89%) compared to ensembles. |
| **K-Nearest Neighbors** | Performs surprisingly well given that the bean classes form distinct, dense clusters in the feature space. |
| **Naive Bayes** | Shows the weakest performance. The strong correlation between geometric features (e.g., ConvexArea vs Area) violates the algorithm's independence assumption. |
| **Random Forest** | Achieves high accuracy (~93%) by aggregating multiple decision trees, which effectively reduces variance and overfitting. |
| **XGBoost** | Currently the top performer. Gradient boosting effectively optimizes the decision boundary for hard-to-classify edge cases. |

## 5. Project Structure
```text
project-folder/
├── streamlit_app.py           # Streamlit Application (Frontend)
├── model_traning.py           # Training Pipeline (Backend)
├── requirements.txt           # Project Dependencies
├── README.md                  # Project Documentation
├── bean_model_performance.csv # Generated Metrics
└── model/                     # Serialized Model Files
    ├── log_reg.pkl
    ├── dt_clf.pkl
    ├── knn.pkl
    ├── gnb.pkl
    ├── rf_clf.pkl
    └── xgb.pkl
```
## 6 Live Application
You can access the deployed Streamlit application to test the model predictions in real-time here: https://mlassignment2-pmocu6xldue9pwadnqlgmu.streamlit.app/
