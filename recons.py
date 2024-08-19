#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from main import show_imgs, find_recons
from tinydas.utils import get_gpus, load_das_file_no_time, get_true_anomalies, get_config, load_model, minmax
from tinydas.losses import mse, mae
from tinydas.constants import *
from tinydas.anomalies import *
from tinydas.selections import select_model
from tinydas.dataset import Dataset
from tinydas.dataloader import DataLoader
from tinydas.enums import Normalization
from tinygrad import dtypes
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt


# ## Tinygrad

# In[35]:


class A:
    def __init__(self):
        self.model = "ae"

args = A()


# In[36]:


devices = get_gpus(1)


# In[37]:


#files = [f"./infer/{f}" for f in sorted(os.listdir("./infer"))]
errors = []
config = get_config(args.model)


# In[38]:


dataset = Dataset(
    path="infer/",
    normalize=Normalization.MINMAX,
    dtype=dtypes.float16 
)


# In[39]:


dl = DataLoader(
    dataset, 
    batch_size=1,
    devices=devices, 
)


# In[40]:


model  = select_model(args.model, devices, **config)
load_model(model)


# In[41]:


for data in dl:
    out = model.predict(data)
        
    rec = mse(data, out).item()
    errors.append(rec)


# In[42]:


errors = np.array(errors)
th = np.percentile(errors, 95)

predicted_anomalies = errors > th


# In[43]:


true_anomalies = get_true_anomalies()


# ### Conf Mat

# In[23]:


cm = confusion_matrix(true_anomalies, predicted_anomalies)


# In[24]:


plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'{args.model} Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[44]:


print(classification_report(true_anomalies, predicted_anomalies))


# In[45]:


def compute_metrics(true_anomalies, reconstruction_errors):
    precisions, recalls, thresholds = precision_recall_curve(true_anomalies, reconstruction_errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    return precisions, recalls, f1_scores, thresholds

precisions, recalls, f1_scores, thresholds = compute_metrics(true_anomalies, errors)

# Find the threshold that gives the best F1 score
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_threshold}")

# Use this threshold for your final predictions
final_predictions = errors > best_threshold

# Print the final classification report
print(classification_report(true_anomalies, final_predictions))


# 
# ### Pr Curve

# In[46]:


plt.figure(figsize=(10,7))
plt.plot(recalls, precisions, marker='.')
plt.title(f'{args.model} Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# ### ROC Curve

# In[47]:


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(true_anomalies, errors)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{args.model} Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

