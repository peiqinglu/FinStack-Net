import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def handle_missing_values(df):
    """
    Handle missing values with multiple strategies.
    :param df: DataFrame containing the features
    :return: DataFrame with missing values imputed
    """
    imputer = SimpleImputer(strategy='mean')  # For numerical features
    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)
    df.update(df_imputed)

    # Impute categorical features with most frequent category
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_imputed_cat = pd.DataFrame(cat_imputer.fit_transform(df.select_dtypes(include=[object])), columns=df.select_dtypes(include=[object]).columns)
    df.update(df_imputed_cat)

    return df


def remove_outliers(df, threshold=3):
    """
    Remove outliers based on Z-score threshold.
    :param df: DataFrame containing the features
    :param threshold: Z-score threshold for outlier detection
    :return: DataFrame with outliers removed
    """
    from scipy.stats import zscore
    z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
    df_no_outliers = df[(z_scores < threshold).all(axis=1)]
    return df_no_outliers


def standardize_features(df, method='standard'):
    """
    Standardize features using different methods.
    :param df: DataFrame containing the features
    :param method: Scaling method ('standard' or 'robust')
    :return: Scaled DataFrame
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Unknown scaling method")

    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return scaled_df


def handle_class_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE.
    :param X_train: Feature matrix for training
    :param y_train: Target vector for training
    :return: Resampled feature and target vectors
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def encode_categorical_features(df):
    """
    Encode categorical features using OneHotEncoder.
    :param df: DataFrame with categorical features
    :return: DataFrame with encoded categorical features
    """
    encoder = OneHotEncoder(sparse=False)
    cat_features = df.select_dtypes(include=[object]).columns
    encoded_features = encoder.fit_transform(df[cat_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_features))
    df = df.drop(columns=cat_features)
    df = pd.concat([df, encoded_df], axis=1)
    return df
