from typing import List, Optional

import numpy as np
from sklearn.metrics import classification_report

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


if __name__ == "__main__":
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    print(anomaly_classification_report(y_true, y_pred))
