from datetime import datetime
from typing import List, Optional

import numpy as np
from sklearn.metrics import classification_report
from tinygrad import Tensor

from tinydas.losses import mse
from tinydas.selections import select_model
from tinydas.utils import custom_flatten, get_config, load_das_file, reshape_back

"""
1. Measure error between the input and the reconstruction for each sample. gather these in a vector.
2. Set a threshold value above which a sample is considered an anomaly. 99th percentage often works
3. Run model on test data and calculate the error for each sample.
4. If the error is above the threshold, mark the sample as an anomaly.
5. Plot the error values and the threshold to visualize the anomalies.
"""


def anomaly_classification_report(
    y_true, y_pred, target_names: Optional[List[str]] = None
):
    """
    Compute classification report for anomaly detection.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    target_names : array-like of shape (n_classes,), default=None
        Optional display names matching the labels.

    Returns
    -------
    report : str
        Classification report.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    target_names = target_names or ["normal", "anomaly"]
    return classification_report(y_true, y_pred, target_names=target_names)


def find_anomalies(y_true: Tensor, y_pred: Tensor, threshold: float = 0.99):
    difference = y_pred.sub(y_true).abs()

    # return the indices of the anomalies
    return difference.numpy() > threshold


def find_first_anomaly_index(anomalies) -> Optional[int]:
    """Find the row where the first anomaly occurs"""
    row_mask = np.any(anomalies, axis=1)
    return int(np.argmax(row_mask)) if np.any(row_mask) else None


def get_datetime_of_first_anomaly(times, row: int) -> datetime:
    return datetime.fromtimestamp(times[row])


def alert_anomaly(dt: datetime):
    print(f"Anomaly found at {dt}")


def predict_file(filename: str, model_name: str, devices: List[str] = None, **config):
    data, times = load_das_file(filename)

    model = select_model(model_name, devices, **config)

    reconstructed = model.predict(data)
    
    print(reconstructed.shape)

    return
    # reconstruction_loss = mse(data, reconstructed)

    anomalies = find_anomalies(data, reconstructed)

    first_row = find_first_anomaly_index(anomalies)

    if first_row is None:
        return

    dt = get_datetime_of_first_anomaly(times, first_row)

    # plot anomalies and alert of times
    alert_anomaly(dt)


def stream_predict():
    """
    1. find folder of das files
    2. stream them one by one according to model
    3. When an anomaly is found, get the timestamp and
    """
    raise NotImplemented()


def send_email(user_mail: str):
    pass
