from datetime import datetime
from typing import List, Optional

import numpy as np
from sklearn.metrics import classification_report
from tinygrad import Tensor

from tinydas.losses import mse
from tinydas.selections import select_model
from tinydas.utils import custom_flatten, get_config, load_das_file, reshape_back

from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

def compute_metrics(true_anomalies, reconstruction_errors):
    precisions, recalls, thresholds = precision_recall_curve(true_anomalies, reconstruction_errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    return precisions, recalls, f1_scores, thresholds


def plot_pr_curve(recalls, precisions, model_name: str, show: bool = False, save_path: Optional[str] = None):
    plt.figure(figsize=(10,7))
    plt.plot(recalls, precisions, marker='.')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if show: plt.show()
    if save_path is not None: plt.savefig(save_path)


def plot_recontruction_distribution(errors, threshold: float, model_name: str, show: bool = False, save_path: Optional[str] = None):
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50)
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
    plt.title('Distribution of Reconstruction Errors')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    if show: plt.show()
    if save_path is not None: plt.savefig(save_path)


def plot_roc_curve(trues, errors, model_name: str, show: bool = False, save_path: Optional[str] = None):
    fpr, tpr, thresholds = roc_curve(trues, errors)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10,7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if show: plt.show()
    if save_path is not None: plt.savefig(save_path)


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
