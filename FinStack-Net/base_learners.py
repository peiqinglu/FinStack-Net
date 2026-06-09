import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np


# LightGBM Model
def train_lgbm(X_train, y_train, X_val, y_val, params=None):
    """
    Train LightGBM model with early stopping.
    :param X_train: Feature matrix for training
    :param y_train: Target vector for training
    :param X_val: Feature matrix for validation
    :param y_val: Target vector for validation
    :param params: Hyperparameters for LightGBM model
    :return: Trained LightGBM model, validation score
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_error',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'early_stopping_round': 100
        }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100)

    y_pred = model.predict(X_val)
    val_score = accuracy_score(y_val, y_pred)
    return model, val_score


# CatBoost Model
def train_catboost(X_train, y_train, X_val, y_val, params=None):
    """
    Train CatBoost model with early stopping.
    :param X_train: Feature matrix for training
    :param y_train: Target vector for training
    :param X_val: Feature matrix for validation
    :param y_val: Target vector for validation
    :param params: Hyperparameters for CatBoost model
    :return: Trained CatBoost model, validation score
    """
    if params is None:
        params = {
            'iterations': 800,
            'depth': 6,
            'learning_rate': 0.03,
            'l2_leaf_reg': 10,
            'bagging_temperature': 1.0,
            'early_stopping_rounds': 100
        }
    model = cb.CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    y_pred = model.predict(X_val)
    val_score = accuracy_score(y_val, y_pred)
    return model, val_score


# DNN Model
class DNN(nn.Module):
    def __init__(self, input_size, dropout=0.3):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x))
        return x


def train_dnn(X_train, y_train, X_val, y_val, model, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train DNN model with early stopping and evaluation.
    :param X_train: Feature matrix for training
    :param y_train: Target vector for training
    :param X_val: Feature matrix for validation
    :param y_val: Target vector for validation
    :param model: DNN model
    :param epochs: Number of training epochs
    :param batch_size: Size of each batch
    :param learning_rate: Learning rate for optimizer
    :return: Trained DNN model, validation score
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_score = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_score = roc_auc_score(y_val, val_outputs.squeeze())

        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model.state_dict()  # Save the best model

    model.load_state_dict(best_model)
    return model, best_val_score


# Model Fusion (Ensemble)
def model_fusion(lgbm_pred, catboost_pred, dnn_pred, alpha=0.4, beta=0.3, gamma=0.3):
    """
    Fusion of model predictions using weighted averaging.
    :param lgbm_pred: Predictions from LightGBM
    :param catboost_pred: Predictions from CatBoost
    :param dnn_pred: Predictions from DNN
    :param alpha: Weight for LightGBM
    :param beta: Weight for CatBoost
    :param gamma: Weight for DNN
    :return: Final fused prediction
    """
    return alpha * lgbm_pred + beta * catboost_pred + gamma * dnn_pred


# Evaluation Metrics
def evaluate_model(y_true, y_pred):
    """
    Evaluate model using various metrics.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred)
    }
    return metrics

