import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from base_learners import train_lgbm, train_catboost, train_dnn, model_fusion
from data_preprocessing import handle_missing_values, standardize_features, handle_class_imbalance
from feature_engineering import first_order_crossing, second_order_crossing, feature_selection_with_mutual_information
import lightgbm as lgb
import catboost as cb
import torch
from torch.utils.data import DataLoader, TensorDataset

# Generate or load data (for example purposes, using random data)
def generate_data(n_samples=10000, n_features=50):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)
    return pd.DataFrame(X), pd.Series(y)

# Train models and evaluate using various metrics
def evaluate_model(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred)
    }
    return metrics

# Train and evaluate FinStack-Net variants (Ablation Studies)
def ablation_experiment(X_train, y_train, X_val, y_val):
    # Variant 1: FinStack-Net without Attention
    model_lgbm, _ = train_lgbm(X_train, y_train, X_val, y_val)
    model_catboost, _ = train_catboost(X_train, y_train, X_val, y_val)
    model_dnn = DNN(input_size=X_train.shape[1])
    model_dnn, _ = train_dnn(X_train, y_train, X_val, y_val, model_dnn)

    # Without Attention in DNN (remove attention layer)
    model_dnn_no_attention = DNN(input_size=X_train.shape[1], dropout=0.3)
    model_dnn_no_attention, _ = train_dnn(X_train, y_train, X_val, y_val, model_dnn_no_attention)

    # Variant 2: FinStack-Net without Residual Connections
    model_dnn_no_residual = DNN(input_size=X_train.shape[1], dropout=0.3)
    model_dnn_no_residual.layer1 = nn.Linear(X_train.shape[1], 128)
    model_dnn_no_residual.layer2 = nn.Linear(128, 64)
    model_dnn_no_residual.layer3 = nn.Linear(64, 32)
    model_dnn_no_residual, _ = train_dnn(X_train, y_train, X_val, y_val, model_dnn_no_residual)

    # Full FinStack-Net Model
    final_pred = model_fusion(model_lgbm.predict(X_val), model_catboost.predict(X_val), model_dnn.predict(X_val).numpy())
    metrics_full = evaluate_model(y_val, final_pred)
    print("FinStack-Net (Full):", metrics_full)

    # Ablation without Attention
    final_pred_no_attention = model_fusion(model_lgbm.predict(X_val), model_catboost.predict(X_val), model_dnn_no_attention.predict(X_val).numpy())
    metrics_no_attention = evaluate_model(y_val, final_pred_no_attention)
    print("FinStack-Net w/o Attention:", metrics_no_attention)

    # Ablation without Residual Connections
    final_pred_no_residual = model_fusion(model_lgbm.predict(X_val), model_catboost.predict(X_val), model_dnn_no_residual.predict(X_val).numpy())
    metrics_no_residual = evaluate_model(y_val, final_pred_no_residual)
    print("FinStack-Net w/o Residual:", metrics_no_residual)

    return metrics_full, metrics_no_attention, metrics_no_residual

# Train and evaluate baseline models
def baseline_experiment(X_train, y_train, X_val, y_val):
    # LightGBM + CatBoost Ensemble
    model_lgbm, _ = train_lgbm(X_train, y_train, X_val, y_val)
    model_catboost, _ = train_catboost(X_train, y_train, X_val, y_val)

    # DNN Baseline
    model_dnn = DNN(input_size=X_train.shape[1])
    model_dnn, _ = train_dnn(X_train, y_train, X_val, y_val, model_dnn)

    # XGBoost Baseline
    model_xgb = xgb.XGBClassifier(objective="binary:logistic")
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_val)

    # Random Forest Baseline
    from sklearn.ensemble import RandomForestClassifier
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_val)

    # Logistic Regression Baseline
    from sklearn.linear_model import LogisticRegression
    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_val)

    # SVM (RBF Kernel) Baseline
    from sklearn.svm import SVC
    model_svm = SVC(kernel='rbf', probability=True)
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_val)

    # Compare all baselines and print results
    metrics = {}
    metrics["LightGBM + CatBoost Ensemble"] = evaluate_model(y_val, model_fusion(model_lgbm.predict(X_val), model_catboost.predict(X_val), model_dnn.predict(X_val).numpy()))
    metrics["DNN"] = evaluate_model(y_val, model_dnn.predict(X_val).numpy())
    metrics["XGBoost"] = evaluate_model(y_val, y_pred_xgb)
    metrics["Random Forest"] = evaluate_model(y_val, y_pred_rf)
    metrics["Logistic Regression"] = evaluate_model(y_val, y_pred_lr)
    metrics["SVM"] = evaluate_model(y_val, y_pred_svm)

    print("Baseline Models Evaluation:")
    for model, metric in metrics.items():
        print(f"{model}: {metric}")

    return metrics

# Main function to run ablation and baseline experiments
def run_experiments():
    # Generate or load data
    X, y = generate_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data preprocessing
    X_train = handle_missing_values(X_train)
    X_val = handle_missing_values(X_val)
    X_train = standardize_features(X_train)
    X_val = standardize_features(X_val)
    X_train, y_train = handle_class_imbalance(X_train, y_train)
    X_val, y_val = handle_class_imbalance(X_val, y_val)

    # Feature engineering
    X_train = first_order_crossing(X_train)
    X_val = first_order_crossing(X_val)

    # Run Ablation Study
    ablation_results = ablation_experiment(X_train, y_train, X_val, y_val)

    # Run Baseline Models
    baseline_results = baseline_experiment(X_train, y_train, X_val, y_val)

    return ablation_results, baseline_results

if __name__ == "__main__":
    ablation_results, baseline_results = run_experiments()
