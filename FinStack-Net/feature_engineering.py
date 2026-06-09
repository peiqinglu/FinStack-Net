import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Lasso

def generate_polynomial_features(df, degree=2):
    """
    Generate polynomial features for the DataFrame.
    :param df: DataFrame containing the features
    :param degree: Degree of the polynomial features
    :return: DataFrame with polynomial features
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df)
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df.columns))
    return poly_df

def first_order_crossing(features):
    """
    Perform first-order feature crossing between all pairs of features.
    :param features: DataFrame of features
    :return: DataFrame with first-order crossed features
    """
    crossed_features = []
    for i in range(len(features.columns)):
        for j in range(i + 1, len(features.columns)):
            crossed_features.append(features.iloc[:, i] * features.iloc[:, j])
    return pd.DataFrame(crossed_features).T

def second_order_crossing(features):
    """
    Perform second-order feature crossing between all triplets of features.
    :param features: DataFrame of features
    :return: DataFrame with second-order crossed features
    """
    crossed_features = []
    for i in range(len(features.columns)):
        for j in range(i + 1, len(features.columns)):
            for k in range(j + 1, len(features.columns)):
                crossed_features.append(features.iloc[:, i] * features.iloc[:, j] * features.iloc[:, k])
    return pd.DataFrame(crossed_features).T

def feature_selection_with_mutual_information(features, target, threshold=0.01):
    """
    Select features using mutual information.
    :param features: DataFrame of features
    :param target: Target variable
    :param threshold: Threshold for mutual information
    :return: DataFrame of selected features
    """
    mi = mutual_info_classif(features, target)
    selected_columns = features.columns[mi >= threshold]
    return features[selected_columns]
