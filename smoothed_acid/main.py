import os, random
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../bars/')

import matplotlib.pyplot as plt
### 解决报错 Could not connect to any X display.
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
sns.set_style("darkgrid")

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from classifier.subnet import AdaptiveClustering, Loss_AdaptiveClustering
from classifier.classifier import AdaptiveClusteringClassifier
from utils import load_data, AverageMeter, calc_metrics_classifier, calc_metrics_certify

from smoothing import Smooth2, Noise
from distribution_transformer import distribution_transformers, loss_functions
from optimizing_noise import optimizing_noise

# General
no_gpu = False # Avoid using CUDA when available
cuda_ids = "0" # GPU No
n_gpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() and not no_gpu else "cpu")
data_dir = "data" # Data directory
dataset_name = "ids18" # The name of data set
if dataset_name not in os.listdir(data_dir):
    raise ValueError("Dataset not found: %s" % (dataset_name))
data_dir = os.path.join(data_dir, dataset_name)

# Building classifier
x_train, y_train = load_data(data_dir, "train")
input_size_classifier = x_train.size(1) # Number of feature dimensions
num_classes_classifier = torch.unique(y_train).size(0) # Number of classes for classifier
learning_rate_classifier = 1e-4 # Learning rate for training classifier
num_epochs_classifier = 30 # Epoch number for training classifier
batch_size_train = 512 # Batch size for training classifier
batch_size_eval = 512 # Batch size for evaluating classifier
save_dir_train = "save/%s/train" % (dataset_name) # Saving directory for classifier training
if not os.path.exists(save_dir_train):
    os.makedirs(save_dir_train)
print_step_classifier = 100 # Step size for showing training progress

# Certifying classifier
certify_class = 0 # The certified class. e.g., Benign(0), FTP-Bruteforce(1), DDoS-HOIC(2), Botnet-Zeus&Ares(3)
feature_noise_distribution = "isru" # Feature noise distribution. e.g., "gaussian" "isru" "isru_gaussian" "isru_gaussian_arctan"
learning_rate_shape = 1e-2 # Learning rate for optimzing noise shape
nt_shape = 1000 # Number of noised samples for optimzing noise shape
lambda_shape = 1e-2 # Regularizer weight
num_epochs_shape = 10 # Number of epochs for optimzing noise shape
d = input_size_classifier # Number of feature dimensions
num_classes_certify = num_classes_classifier # Number of certified classes
n0 = 100 # Number of noised samples for identify cA
n = 10000 # Number of noised samples for estimate pA
alpha = 1e-3 # Failure probability
init_step_size_scale = 5e-2 # Initial update step size of t for optimzing noise scale
init_ptb_t_scale = 1e-2 # Initial perturbation of t for optimzing noise scale
decay_factor_scale = 0.5 # Decay factor for optimzing noise scale
max_decay_scale = 6 # Maximum decay times for optimzing noise scale
max_iter_scale = 100 # Maximum iteration times for optimzing noise scale
batch_size_iteration_certify = 128 # Batch size of certified samples for robustness certification
batch_size_memory_certify = 1000000 # Batch size of noised samples for robustness certification
print_step_certify = 20 # Step size for showing certification progress
save_dir_certify = "save/%s/certify" % (dataset_name) # Saving directory for robustness certification
if not os.path.exists(save_dir_certify):
    os.makedirs(save_dir_certify)
num_samples_certify = 10000 # Number of certified samples sampled from dataset

seed = 42 # Random seed

def set_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def create_ac():
    return AdaptiveClustering(input_size_classifier, num_classes_classifier, device)

def create_classifier(encoder_list: nn.ModuleList, kernel_net_list: nn.ModuleList):
    return AdaptiveClusteringClassifier(encoder_list, kernel_net_list, device)

def create_noise_generator():
    dist_trans = distribution_transformers[feature_noise_distribution](d).to(device)
    return Noise(dist_trans, d, device)

def train():
    print("\n***** Run training *****")
    print("Number of classes for classifier:", num_classes_classifier)

    ac = create_ac()
    ac.to(device)

    opt = optim.Adam(ac.parameters(), lr=learning_rate_classifier)

    criterion = Loss_AdaptiveClustering(num_classes_classifier)

    x_train, y_train = load_data(data_dir, "train")

    data_dataset = TensorDataset(x_train, y_train)
    data_loader = DataLoader(data_dataset, batch_size=batch_size_train, shuffle=True)

    loss_record = AverageMeter()
    loss_classification_record = AverageMeter()
    loss_clustering_close_record = AverageMeter()
    loss_clustering_dist_record = AverageMeter()
    ac.to(ac.device)
    ac.train()
    for epoch in range(num_epochs_classifier + 1):
        for i, (X, y) in zip(range(1, len(data_loader) + 1), data_loader):
            X, y = X.to(ac.device), y.to(ac.device)

            z, kernel_weight, o = ac(X)
            loss, loss_classification, loss_clustering_close, loss_clustering_dist = criterion(z, kernel_weight, o, y)

            if epoch > 0:
                opt.zero_grad()
                loss.backward()
                opt.step()

            loss_record.update(loss.item())
            loss_classification_record.update(loss_classification.item())
            loss_clustering_close_record.update(loss_clustering_close.item())
            loss_clustering_dist_record.update(loss_clustering_dist.item())
            if i % print_step_classifier == 0:
                print(("Batch: [%d/%d][%d/%d] | Loss: %.6f | " + \
                    "Loss classification: %.6f | " + \
                    "Loss clustering_close: %.6f | " + \
                    "Loss clustering_dist: %.6f") % ( \
                    epoch, num_epochs_classifier, i, len(data_loader), loss_record.val, \
                    loss_classification_record.val, \
                    loss_clustering_close_record.val, \
                    loss_clustering_dist_record.val))

        print(('Epoch: [%d/%d] | Loss (Avg): %.6f | ' + \
            'Loss classification (Avg): %.6f | ' + \
            'Loss clustering_close (Avg): %.6f | ' + \
            'Loss clustering_dist (Avg): %.6f') % ( \
            epoch, num_epochs_classifier, loss_record.avg, \
            loss_classification_record.avg, \
            loss_clustering_close_record.avg, \
            loss_clustering_dist_record.avg))

        loss_record.reset()
        loss_classification_record.reset()
        loss_clustering_close_record.reset()
        loss_clustering_dist_record.reset()

    torch.save(ac.encoder_list, os.path.join(save_dir_train, "checkpoint-encoder_list"))
    torch.save(ac.kernel_weight, os.path.join(save_dir_train, "checkpoint-kernel_weight"))
    torch.save(ac.kernel_net_list, os.path.join(save_dir_train, "checkpoint-kernel_net_list"))

def evaluate():
    print("\n***** Run evaluating *****")
    print("Number of classes for classifier:", num_classes_classifier)

    encoder_list = torch.load(os.path.join(save_dir_train, "checkpoint-encoder_list"))
    encoder_list = nn.ModuleList([e.to(device) for e in encoder_list])
    kernel_net_list = torch.load(os.path.join(save_dir_train, "checkpoint-kernel_net_list"))
    kernel_net_list = nn.ModuleList([n.to(device) for n in kernel_net_list])
    classifier = create_classifier(encoder_list, kernel_net_list)

    x_test, y_test = load_data(data_dir, "test")

    data_dataset = TensorDataset(x_test, y_test)
    data_loader = DataLoader(data_dataset, batch_size=batch_size_eval, shuffle=False)

    label_record = np.array([], dtype=np.long)
    pred_record = np.array([], dtype=np.long)
    classifier.eval()
    torch.set_grad_enabled(False)
    for X, y in tqdm(data_loader, desc="Evaluate"):
        X, y = X.to(classifier.device), y.to(classifier.device)
        pred = classifier(X)

        pred_record = np.concatenate([pred_record, pred.detach().cpu().numpy()], 0)
        label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()

    acc, m = calc_metrics_classifier(label_record, pred_record)
    print(('Accuracy: %.4f') % (acc))
    print("Confusion matrix (Row: ground truth class, Col: prediction class):")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if j < m.shape[1] - 1:
                print("%d " % (m[i, j]), end="")
            else:
                print("%d" % (m[i, j]))

def certify():
    print("\n***** Optimize noise *****")

    encoder_list = torch.load(os.path.join(save_dir_train, "checkpoint-encoder_list"))
    encoder_list = nn.ModuleList([e.to(device) for e in encoder_list])
    kernel_net_list = torch.load(os.path.join(save_dir_train, "checkpoint-kernel_net_list"))
    kernel_net_list = nn.ModuleList([n.to(device) for n in kernel_net_list])
    classifier = create_classifier(encoder_list, kernel_net_list)

    noise_generator = create_noise_generator()

    criterion_shape = loss_functions["acid"](lambda_shape, certify_class)

    x_train, _ = load_data(data_dir, "train")

    dataset = TensorDataset(x_train)
    data_loader = DataLoader(dataset, batch_size=batch_size_iteration_certify, shuffle=True)
    x_train_c = torch.tensor([], dtype=torch.float32)
    for (X,) in tqdm(data_loader, desc="Select class %d" % (certify_class)):
        X = X.to(device)

        pred = classifier(X)

        x_train_c = torch.cat([x_train_c, X[(pred == certify_class), :].detach().cpu()], 0)

    optimizing_noise(
        x_train_c,
        certify_class,
        classifier,
        noise_generator,
        criterion_shape,
        learning_rate_shape,
        nt_shape,
        num_epochs_shape,
        d,
        num_classes_certify,
        n0,
        n,
        alpha,
        init_step_size_scale,
        init_ptb_t_scale,
        decay_factor_scale,
        max_decay_scale,
        max_iter_scale,
        batch_size_iteration_certify,
        batch_size_memory_certify,
        print_step_certify,
        save_dir_certify)

    noise_generator.distribution_transformer = torch.load(os.path.join(save_dir_certify, "checkpoint-distribution-transformer")).to(device)

    r = open(os.path.join(save_dir_certify, "t"), "r")
    t = float(r.readline())
    r.close()

    print("\n***** Certify robustness *****")

    x_test, y_test = load_data(data_dir, "test")

    dataset = TensorDataset(x_test, y_test)
    data_loader = DataLoader(dataset, batch_size=batch_size_iteration_certify, shuffle=False)
    x_test_c = torch.tensor([], dtype=torch.float32)
    y_test_c = torch.tensor([], dtype=torch.long)
    for (X, y) in tqdm(data_loader, desc="Select class %d" % (certify_class)):
        X = X.to(device)

        pred = classifier(X)

        x_test_c = torch.cat([x_test_c, X[(pred == certify_class), :].detach().cpu()], 0)
        y_test_c = torch.cat([y_test_c, y[(pred == certify_class)].detach().cpu()], 0)

    idx = torch.arange(x_test_c.shape[0])
    idx = idx[torch.randperm(idx.size(0))]
    idx = torch.sort(idx[:min(num_samples_certify, idx.shape[0])])[0]
    x_test_c, y_test_c = x_test_c[idx, :], y_test_c[idx]

    dataset = TensorDataset(x_test_c, y_test_c)
    data_loader = DataLoader(dataset, batch_size=batch_size_iteration_certify, shuffle=False)

    classifier.eval()
    smoothed_classifier = Smooth2(classifier, d, num_classes_certify, noise_generator, device)
    cA_record = np.array([], dtype=np.long)
    robust_radius_record = np.array([], dtype=np.float32)
    label_record = np.array([], dtype=np.long)
    torch.set_grad_enabled(False)
    for X, y in tqdm(data_loader, desc="Certify"):
        X = X.to(device)

        cA, _, robust_radius = smoothed_classifier.bars_certify(X, n0, n, t, alpha, batch_size_memory_certify)

        cA_record = np.concatenate([cA_record, cA], 0)
        robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

        label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()

    mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
    print("Mean Robustness Radius: %.6e" % (mean_robust_radius))

    w = open(os.path.join(save_dir_certify, "certification_results"), "w")
    w.write("label\tcA\trobust_radius\n")
    for i in range(label_record.shape[0]):
        w.write("%d\t%d\t%.6e\n" % (label_record[i], cA_record[i], robust_radius_record[i]))
    w.close()

    max_robust_radius = {"ids18-0": 0.8, "ids18-1": 0.5, "ids18-2": 0.25, "ids18-3": 0.8}
    robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset_name, certify_class)], max_robust_radius["%s-%d" % (dataset_name, certify_class)] * 1e-3)
    certified_accuracy_plot = np.array([], dtype=np.float32)
    for r in robust_radius_plot:
        certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
    plt.figure(1)
    plt.plot(robust_radius_plot, certified_accuracy_plot)
    plt.ylim((0, 1))
    plt.xlim((0, max_robust_radius["%s-%d" % (dataset_name, certify_class)]))
    plt.tick_params(labelsize=14)
    plt.xlabel("Robustness Radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
    plt.title("ACID %s" % (class_name["%s-%d" % (dataset_name, certify_class)]), fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_certify, "certified_accuracy_robustness_radius_curve.png"))
    plt.close()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_ids

    set_seed() # For reproductibility

    # Training
    train()

    # Evaluating
    evaluate()

    # Certifying
    certify()

if __name__ == "__main__":
    main()