from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_linear_regression(X_train, y_train, X_val, y_val):

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    print(f"Training MSE: {train_mse}")
    print(f"Validation MSE: {val_mse}")

    return model


def train_2_linear_regression_models(X_train, y_train, X_val, y_val):
    model_recovery = train_linear_regression(
        X_train, y_train[["recovery_days"]], X_val, y_val[["recovery_days"]]
    )
    model_economic = train_linear_regression(
        X_train, y_train[["economic_loss_usd"]], X_val, y_val[["economic_loss_usd"]]
    )
    return model_recovery, model_economic


def accuracy_linear_2_models(model_recovery, model_economic, X_test, y_test):
    from sklearn.metrics import mean_squared_error, r2_score

    y_test_recovery_pred = model_recovery.predict(X_test)
    y_test_economic_pred = model_economic.predict(X_test)

    test_mse_recovery = mean_squared_error(
        y_test[["recovery_days"]], y_test_recovery_pred
    )
    test_r2_recovery = r2_score(y_test[["recovery_days"]], y_test_recovery_pred)

    test_mse_economic = mean_squared_error(
        y_test[["economic_loss_usd"]], y_test_economic_pred
    )
    test_r2_economic = r2_score(y_test[["economic_loss_usd"]], y_test_economic_pred)

    # AI generated snippet
    metrics = {
        "recovery_days": {
            "mse": test_mse_recovery,
            "r2": test_r2_recovery,
        },
        "economic_loss_usd": {
            "mse": test_mse_economic,
            "r2": test_r2_economic,
        },
    }

    print("-" * 50)
    print(f"{'Target Variable':<25} | {'MSE':<10} | {'R² Score':<10}")
    print("-" * 50)
    print(
        f"{'Recovery Days':<25} | {test_mse_recovery:<10.4f} | {test_r2_recovery:.4f}"
    )
    print(
        f"{'Economic Loss USD':<25} | {test_mse_economic:<10.4f} | {test_r2_economic:.4f}"
    )
    print("-" * 50)

    return metrics
