import os
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import *

import torch

def load_data(data_dir: str, data_type: str="train"):

    if data_type == "train":
        x_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))

        return torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)

    elif data_type == "test":
        x_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))

        return torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)

    else:
        raise NotImplementedError()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def calc_metrics_classifier(label: np.ndarray, pred: np.ndarray):
    assert label.shape == pred.shape
    acc = np.array((label == pred), dtype=float).mean()
    m = confusion_matrix(y_true=label, y_pred=pred)

    return acc, m

def calc_metrics_certify(label: np.ndarray, pred: np.ndarray, radius_feat: np.ndarray) -> np.ndarray:
    assert label.shape == pred.shape
    assert pred.shape == radius_feat.shape
    
    return np.mean(np.array((label == pred), dtype=float) * radius_feat)