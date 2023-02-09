import os
import numpy as np
from typing import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from smoothing import Smooth2

num_samples_shape = 10000
num_samples_scale = 1000

def optimizing_noise(
    x_train: torch.tensor, # input data for optimization
    pred_train: int, # classifier prediction of input data
    classifier: nn.Module, # classifier model
    noise_generator: nn.Module, # noise generator model
    criterion_shape: nn.Module, # loss function for optimizing noise shape
    learning_rate_shape: float, # learning rate for optimizing noise shape
    nt_shape: int, # number of noised samples for optimizing noise shape
    num_epochs_shape: int, # epochs for optimizing noise shape
    d: int, # number of feature dimensions
    num_classes: int, # number of certified classes
    n0: int, # number of noised samples for identify cA
    n: int, # number of noised samples for estimate pA
    alpha: float, # failure probability
    init_step_size_scale: float, # initial update step size of t for optimzing noise scale
    init_ptb_t_scale: float, # initial perturbation of t for optimzing noise scale
    decay_factor_scale: float, # decay factor for optimzing noise scale
    max_decay_scale: int, # maximum decay times for optimzing noise scale
    max_iter_scale: int, # maximum iteration times for optimzing noise scale
    batch_size_iteration: int, # batch size of certified samples for robustness certification
    batch_size_memory: int, # batch size of noised samples for robustness certification
    print_step: int, # step size for showing certification progress
    save_dir: str # saving directory for robustness certification
    ):

    assert classifier.device == noise_generator.device

    print("***** Optimize noise shape *****")

    idx = torch.arange(x_train.shape[0])
    idx = idx[torch.randperm(idx.size(0))]
    idx = torch.sort(idx[:min(num_samples_shape, idx.shape[0])])[0]
    x_train = x_train[idx, :]

    dataset = TensorDataset(x_train)
    data_loader = DataLoader(dataset, batch_size=batch_size_iteration, shuffle=True)

    opt = optim.Adam(noise_generator.distribution_transformer.parameters(), lr=learning_rate_shape)

    noise_norm = noise_generator.sample_norm(nt_shape).repeat(1, batch_size_iteration).view(-1, noise_generator.d)

    loss_record = AverageMeter()
    classifier.train()
    for epoch in range(0, num_epochs_shape + 1):
        for i, (X,) in zip(range(1, len(data_loader) + 1), data_loader):
            X = X.to(classifier.device)
            if X.size(0) < batch_size_iteration:
                noise_idx = torch.cat([torch.arange(X.size(0)) + i * batch_size_iteration \
                    for i in range(nt_shape)], 0).to(X.device)
            else:
                noise_idx = torch.arange(noise_norm.size(0)).to(X.device)
            noise_feat = noise_generator.norm_to_feat(noise_norm[noise_idx, :])

            score = classifier.score(X.repeat((nt_shape, 1)).to(X.device), noise_feat)
            loss = criterion_shape(score, noise_generator.get_weight())

            if epoch > 0:
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()
 
            loss_record.update(loss.item())
            if (i % print_step == 0) or ((print_step > len(data_loader)) and (i == len(data_loader))):
                print("Batch: [%d/%d][%d/%d] | Loss: %.6f" % (epoch, num_epochs_shape, i, len(data_loader), loss_record.val))

        print("Epoch: [%d/%d] | Loss (Avg): %.6f" % (epoch, num_epochs_shape, loss_record.avg))
        loss_record.reset()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(noise_generator.distribution_transformer, os.path.join(save_dir, "checkpoint-distribution-transformer"))

    print("***** Optimize noise scale *****")

    idx = torch.arange(x_train.shape[0])
    idx = idx[torch.randperm(idx.size(0))]
    idx = torch.sort(idx[:min(num_samples_scale, idx.shape[0])])[0]
    x_train = x_train[idx, :]

    dataset = TensorDataset(x_train)
    data_loader = DataLoader(dataset, batch_size=batch_size_iteration, shuffle=True)

    classifier.eval()
    smoothed_classifier = Smooth2(classifier, d, num_classes, noise_generator, classifier.device)
    ptb_t_scale = init_ptb_t_scale
    step_size_scale = init_step_size_scale
    decay_scale = 0
    iter_scale = 0
    t = 0.0
    grad_sign_last = 1
    torch.set_grad_enabled(False)
    while (iter_scale < max_iter_scale) and (decay_scale < max_decay_scale):
        cA_record = np.array([])
        robust_radius_record = np.array([])
        for (X,) in data_loader:
            X = X.to(classifier.device)

            cA, _, robust_radius = smoothed_classifier.bars_certify(X, n0, n, t, alpha, batch_size_memory)
            
            cA_record = np.concatenate([cA_record, cA], 0)
            robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
        
        mean_robust_radius = calc_mean_robust_radius(pred_train, cA_record, robust_radius_record)

        cA_record = np.array([])
        robust_radius_record = np.array([])
        for (X,) in data_loader:
            X = X.to(classifier.device)

            cA, _, robust_radius = smoothed_classifier.bars_certify(X, n0, n, t + ptb_t_scale, alpha, batch_size_memory)
            
            cA_record = np.concatenate([cA_record, cA], 0)
            robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
        
        mean_robust_radius_ptb = calc_mean_robust_radius(pred_train, cA_record, robust_radius_record)

        grad_t = (mean_robust_radius_ptb - mean_robust_radius) / ptb_t_scale
        if np.sign(grad_t) != grad_sign_last:
            ptb_t_scale *= decay_factor_scale
            step_size_scale *= decay_factor_scale
            grad_sign_last = np.sign(grad_t)
            decay_scale += 1
        t = t + step_size_scale * np.sign(grad_t)

        cA_record = np.array([])
        robust_radius_record = np.array([])
        for (X,) in data_loader:
            X = X.to(classifier.device)

            cA, _, robust_radius = smoothed_classifier.bars_certify(X, n0, n, t, alpha, batch_size_memory)
            
            cA_record = np.concatenate([cA_record, cA], 0)
            robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

        mean_robust_radius_last = calc_mean_robust_radius(pred_train, cA_record, robust_radius_record)

        iter_scale += 1
        print("Iter: [%d] | t: %.6e | Robust radius: %.6e | Step size: %.6e | Grad sign: %d" % \
            (iter_scale, t, mean_robust_radius_last, step_size_scale, grad_sign_last))

    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    w = open(os.path.join(save_dir, "t"), "w")
    w.write("%.6e" % (t))
    w.close()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

def calc_mean_robust_radius(pred: int, cA: np.ndarray, radius_feat: np.ndarray) -> np.ndarray:
    assert cA.shape == radius_feat.shape

    return np.mean(np.array((pred == cA), dtype=float) * radius_feat)