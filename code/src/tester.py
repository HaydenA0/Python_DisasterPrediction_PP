import numpy as np


def predict(
    X_test_processed,
    y_train,
    model_recovery,
    model_economic,
    y_test,
    test_id,
):
    X_test_sample = X_test_processed.iloc[[test_id]]
    recovery_days_pred = model_recovery.predict(X_test_sample)
    economic_loss_pred = model_economic.predict(X_test_sample)
    # make the predictions into a readable format
    print(f"For test sample id {test_id} :")
    print(f"Predicted Recovery Days (normalized) : {recovery_days_pred[0]}")
    print(f"Predicted Economic Loss USD (log normalized) : {economic_loss_pred[0]}")
    # denormalize it and un-log it for economic loss
    # denormalize recovery days
    recovery_days_denorm = (
        recovery_days_pred[0]
        * (y_train["recovery_days"].max() - y_train["recovery_days"].min())
        + y_train["recovery_days"].min()
    )
    # un-log and denormalize economic loss
    economic_loss_unlog = np.expm1(economic_loss_pred[0])
    economic_loss_denorm = (
        economic_loss_unlog
        * (y_train["economic_loss_usd"].max() - y_train["economic_loss_usd"].min())
        + y_train["economic_loss_usd"].min()
    )
    print(f"Denormalized Predicted Recovery Days : {recovery_days_denorm}")
    print(f"Denormalized Predicted Economic Loss USD : {economic_loss_denorm}")
