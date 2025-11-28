from data_loader import get_data
from data_process import pre_process, get_x_y, preprocess_target
from train_model import (
    train_2_linear_regression_models,
    accuracy_linear_2_models,
)

file_path = "data/raw/global_disaster_response.csv"
FEATURE_COLUMNS = [
    "country",
    "date",
    "disaster_type",
    "severity_index",
    "casualties",
    "longitude",
    "latitude",
    "aid_amount_usd",
    "response_time_hours",
]
TARGET_COLUMN = ["recovery_days", "economic_loss_usd"]


if __name__ == "__main__":
    raw_data = get_data(file_path)
    X_train, X_val, X_test, y_train, y_val, y_test = get_x_y(
        raw_data, FEATURE_COLUMNS, TARGET_COLUMN
    )
    X_train_processed = pre_process(X_train)
    y_train_processed = preprocess_target(y_train)
    X_val_processed = pre_process(X_val)
    y_val_processed = preprocess_target(y_val)
    X_test_processed = pre_process(X_test)
    y_test_processed = preprocess_target(y_test)
    model_recovery, model_economic = train_2_linear_regression_models(
        X_train_processed, y_train_processed, X_val_processed, y_val_processed
    )
    accuracy_linear_2_models(
        model_recovery, model_economic, X_test_processed, y_test_processed
    )
