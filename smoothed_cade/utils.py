import os, collections
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score, confusion_matrix
from typing import *

import torch

def load_data(data_dir: str, data_type: str="train"):

    if data_type == "train":
        x_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train_class = np.load(os.path.join(data_dir, "y_train.npy"))

        class_map, num_classes_train = get_class_map(data_dir)
        y_train_class_new = np.zeros_like(y_train_class, dtype=np.long)
        for k, v in class_map.items():
            y_train_class_new[y_train_class == k] = v

        return torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train_class_new, dtype=torch.long), num_classes_train, class_map

    elif data_type == "test":
        x_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test_class = np.load(os.path.join(data_dir, "y_test.npy"))

        class_map, num_classes_test = get_class_map(data_dir)
        y_test_class_new = np.zeros_like(y_test_class, dtype=np.long)
        for k, v in class_map.items():
            y_test_class_new[y_test_class == k] = v

        y_train_class = np.load(os.path.join(data_dir, "y_train.npy"))
        train_class_set = np.unique(y_train_class).tolist()
        y_test_drift = np.ones_like(y_test_class, dtype=np.long)
        for i in range(len(train_class_set)):
            y_test_drift[y_test_class == train_class_set[i]] = 0

        return torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test_class_new, dtype=torch.long), num_classes_test, class_map, torch.tensor(y_test_drift, dtype=torch.long)

    else:
        raise NotImplementedError()

def get_class_map(data_dir: str):
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    class_train = np.unique(y_train).tolist()
    class_train.sort()
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    class_test = np.unique(y_test).tolist()
    class_test.sort()
    class_map = collections.OrderedDict()

    for i in range(len(class_train)):
        class_map.update({class_train[i]: len(class_map)})
    for i in range(len(class_test)):
        if class_test[i] not in class_train:
            class_map.update({class_test[i]: len(class_map)})
        
    return class_map, len(class_train)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def calc_metrics_classifier_class(label: np.ndarray, pred: np.ndarray):
    assert label.shape == pred.shape
    acc = np.array((label == pred), dtype=float).mean()
    m = confusion_matrix(y_true=label, y_pred=pred)

    return acc, m

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