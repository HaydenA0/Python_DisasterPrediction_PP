import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_x_y(raw_data, FEATURE_COLUMNS, TARGET_COLUMN):
    X = raw_data[FEATURE_COLUMNS]
    y = raw_data[TARGET_COLUMN]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )  # 0.25 x 0.8 = 0.2

    return X_train, X_val, X_test, y_train, y_val, y_test


def pre_process(X):
    # preprocess date
    X["date"] = pd.to_datetime(X["date"])
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X = X.drop("date", axis=1)

    X = pd.get_dummies(X, columns=["country", "disaster_type"], drop_first=True)
    numerical_features = [
        "severity_index",
        "casualties",
        "response_time_hours",
        "longitude",
        "latitude",
        "aid_amount_usd",
    ]

    for feature in numerical_features:
        X[feature] = (X[feature] - X[feature].min()) / (
            X[feature].max() - X[feature].min()
        )
    return X


def preprocess_target(y):
    y["recovery_days"] = (y["recovery_days"] - y["recovery_days"].min()) / (
        y["recovery_days"].max() - y["recovery_days"].min()
    )
    y["economic_loss_usd"] = (y["economic_loss_usd"] - y["economic_loss_usd"].min()) / (
        y["economic_loss_usd"].max() - y["economic_loss_usd"].min()
    )
    # transform log scale
    y["economic_loss_usd"] = np.log1p(y["economic_loss_usd"])
    return y
