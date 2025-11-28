from data_loader import get_data
from data_process import pre_process, get_x_y, preprocess_target
from tester import predict
from train_model import (
    train_2_linear_regression_models,
    accuracy_linear_2_models,
)
import pandas as pd

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
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

if __name__ == "__main__":
    raw_data = get_data(file_path)
    X_train, X_val, X_test, y_train, y_val, y_test = get_x_y(
        raw_data, FEATURE_COLUMNS, TARGET_COLUMN
    )
    y_train_org = pd.DataFrame(y_train.copy())
    y_test_org = pd.DataFrame(y_test.copy())

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

    user_id = int(
        input(
            f"{GREEN}Enter what test id do you want the see prediction of ? : {RESET}\n"
        )
    )
    predict(
        X_test_processed,
        y_train_org,
        model_recovery,
        model_economic,
        y_test_org,
        user_id,
    )
    print(f"{RED}{ pd.DataFrame(raw_data).iloc[user_id].iloc[:-2] } {RESET}")
