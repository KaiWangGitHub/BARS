import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import torch

def load_data(data_dir: str, data_type: str="train"):

    if data_type == "cluster":
        x_cluster = np.load(os.path.join(data_dir, "X_cluster.npy"))

        return torch.tensor(x_cluster, dtype=torch.float)

    elif data_type == "train":
        x_train = np.load(os.path.join(data_dir, "X_train.npy"))

        return torch.tensor(x_train, dtype=torch.float)

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
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val: float):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def calc_metrics_classifier(label: np.ndarray, pred: np.ndarray) -> (float, float, float, float):
    assert label.shape == pred.shape
    acc = np.array((label == pred), dtype=float).mean()
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    f1 = f1_score(label, pred)

    return acc, p, r, f1

def calc_metrics_certify(label: np.ndarray, pred: np.ndarray, radius_feat: np.ndarray) -> np.ndarray:
    assert label.shape == pred.shape
    assert pred.shape == radius_feat.shape
    
    return np.mean(np.array((label == pred), dtype=float) * radius_feat)