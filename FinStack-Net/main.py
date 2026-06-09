from feature_engineering import first_order_crossing, second_order_crossing, mutual_information_filter, lasso_pruning
from base_learners import train_lgbm, train_catboost, train_dnn, model_fusion
from hyperparameter_optimization import study_lgbm, study_catboost, study_dnn
from data_preprocessing import handle_missing_values, standardize_features, handle_class_imbalance


def main():
    # Load your data
    data = pd.read_csv('your_data.csv')

    # Preprocess the data
    X = data.drop(columns=['target'])
    y = data['target']
    X = handle_missing_values(X)
    X = standardize_features(X)
    X, y = handle_class_imbalance(X, y)

    # Feature engineering
    X_first_order = first_order_crossing(X)
    X_second_order = second_order_crossing(X)
    X_filtered = mutual_information_filter(X, y)
    X_pruned = lasso_pruning(X_filtered, y)

    # Train base models
    model_lgbm = train_lgbm(X_pruned, y)
    model_catboost = train_catboost(X_pruned, y)
    model_dnn = train_dnn(X_pruned, y)

    # Predict and fuse results
    lgbm_pred = model_lgbm.predict_proba(X_pruned)[:, 1]
    catboost_pred = model_catboost.predict_proba(X_pruned)[:, 1]
    dnn_pred = model_dnn(X_pruned)

    final_pred = model_fusion(lgbm_pred, catboost_pred, dnn_pred)
    print("Final Prediction:", final_pred)


if __name__ == "__main__":
    main()
