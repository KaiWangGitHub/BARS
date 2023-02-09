import os, random
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../bars/')

import matplotlib.pyplot as plt
# For "Could not connect to any X display."
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

from classifier.autoencoder import Autoeconder, Loss_Contrastive
from classifier.detect import calc_centroid_dis_median_mad, Detector
from utils import load_data, AverageMeter, calc_metrics_classifier, calc_metrics_classifier_class, calc_metrics_certify

from smoothing import Smooth2, Noise
from distribution_transformer import distribution_transformers, loss_functions
from optimizing_noise import optimizing_noise

# General
no_gpu = False # Avoid using CUDA when available
cuda_ids = "0" # GPU No
n_gpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() and not no_gpu else "cpu")
data_dir = "data" # Data directory
dataset_name = "newinfiltration" # The name of data set. e.g., "newinfiltration" "newhulk"
if dataset_name not in os.listdir(data_dir):
    raise ValueError("Dataset not found: %s" % (dataset_name))
data_dir = os.path.join(data_dir, dataset_name)

# Building classifier
learning_rate_classifier = 1e-4 # Learning rate for training classifier
margin = 10 # Distance margin for sample pairs from different classes
lambda_classifier = 1e-1 # Weight for contrastive loss term
mad_threshold = 3.5 # MAD threshold for detecting drift
num_epochs_classifier = 30 # Epoch number for training classifier
batch_size_train = 512 # Batch size for training classifier
batch_size_eval = 512 # Batch size for evaluating classifier
save_dir_train = "save/%s/train" % (dataset_name) # Saving directory for classifier training
if not os.path.exists(save_dir_train):
    os.makedirs(save_dir_train)
print_step_classifier = 50 # Step size for showing training progress

# Certifying classifier
certify_class = 0 # The certified class. e.g., newinfiltration: Benign(0), SSH-Bruteforce(1), DoS-Hulk(2); newhulk: Infiltration(2)
feature_noise_distribution = "isru" # Feature noise distribution. e.g., "gaussian" "isru" "isru_gaussian" "isru_gaussian_arctan"
learning_rate_shape = 2e-4 # Learning rate for optimzing noise shape
nt_shape = 1000 # Number of noised samples for optimzing noise shape
lambda_shape = 1e-1 # Regularizer weight
num_epochs_shape = 5 # Number of epochs for optimzing noise shape
x_train, _, _, _ = load_data(data_dir, "train")
d = x_train.size(1) # Number of feature dimensions
num_classes_certify = 2 # Number of certified classes. i.e., Known class(0), Drift class(1)
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

def create_ae():
    _, _, num_classes_train, _ = load_data(data_dir, "train")
    return Autoeconder(d, num_classes_train, device)

def create_detector(encoder: nn.Module, centroid: torch.tensor, dis_median: torch.tensor, mad: torch.tensor):
    return Detector(encoder, centroid, dis_median, mad, mad_threshold, device)

def create_noise_generator():
    dist_trans = distribution_transformers[feature_noise_distribution](d).to(device)
    return Noise(dist_trans, d, device)

def train():
    print("\n***** Run training *****")
    
    ae = create_ae()
    ae.to(device)

    opt = optim.Adam(ae.parameters(), lr=learning_rate_classifier)

    criterion = Loss_Contrastive(margin, lambda_classifier)

    x_train, y_train_class, num_classes_train, class_map = load_data(data_dir, "train")
    print("Number of classes for training:", num_classes_train)
    print("Class Map from original class to new class (First %d classes are for training): %s" % (num_classes_train, str(class_map)))

    data_dataset = TensorDataset(x_train, y_train_class)
    data_loader = DataLoader(data_dataset, batch_size=batch_size_train, shuffle=True)

    loss_record = AverageMeter()
    loss_reconstructed_record = AverageMeter()
    loss_contrastive_record = AverageMeter()
    z_record = torch.tensor([], dtype=torch.float32)
    label_class_record = torch.tensor([], dtype=torch.long)
    ae.train()
    for epoch in range(num_epochs_classifier + 1):
        for i, (X, y_class) in zip(range(1, len(data_loader) + 1), data_loader):
            X, y_class = X.to(device), y_class.to(device)

            z, x_reconstructed = ae(X)
            loss, loss_reconstructed, loss_contrastive = criterion(X, z, x_reconstructed, y_class)

            if epoch > 0:
                opt.zero_grad()
                loss.backward()
                opt.step()

            loss_record.update(loss.item())
            loss_reconstructed_record.update(loss_reconstructed.item())
            loss_contrastive_record.update(loss_contrastive.item())
            if i % print_step_classifier == 0:
                print(("Batch: [%d/%d][%d/%d] | Loss: %.6f | " + \
                    "Loss reconstructed: %.6f | " + \
                    "Loss contrastive: %.6f") % ( \
                    epoch, num_epochs_classifier, i, len(data_loader), loss_record.val, \
                    loss_reconstructed_record.val, \
                    loss_contrastive_record.val))

            if epoch == num_epochs_classifier:
                z, x_reconstructed = ae(X)

                z_record = torch.cat([z_record, z.detach().cpu()], 0)
                label_class_record = torch.cat([label_class_record, y_class.detach().cpu()], 0)

        print(("Epoch: [%d/%d] | Loss (Avg): %.6f | " + \
            "Loss reconstructed (Avg): %.6f | " + \
            "Loss contrastive (Avg): %.6f") % ( \
            epoch, num_epochs_classifier, loss_record.avg, \
            loss_reconstructed_record.avg, \
            loss_contrastive_record.avg))

        loss_record.reset()
        loss_reconstructed_record.reset()
        loss_contrastive_record.reset()

    torch.save(ae.encoder, os.path.join(save_dir_train, "checkpoint-encoder"))
    torch.save(ae.decoder, os.path.join(save_dir_train, "checkpoint-decoder"))

    centroid, dis_median, mad = calc_centroid_dis_median_mad(z_record, label_class_record)
    torch.save({
        "centroid": centroid,
        "dis_median": dis_median,
        "mad": mad
    }, os.path.join(save_dir_train, "checkpoint-param"))

def evaluate():
    print("\n***** Run evaluating *****")

    param = torch.load(os.path.join(save_dir_train, "checkpoint-param"))
    encoder = torch.load(os.path.join(save_dir_train, "checkpoint-encoder"))
    detector = create_detector(encoder.to(device), param["centroid"].to(device), param["dis_median"].to(device), \
        param["mad"].to(device))

    x_test, y_test_class, num_classes_train, class_map, y_test_drift = load_data(data_dir, "test")
    print("Number of classes for training:", num_classes_train)
    print("Class Map from original class to new class (First %d classes are for training): %s" % (num_classes_train, str(class_map)))

    dataset = TensorDataset(x_test, y_test_class, y_test_drift)
    data_loader = DataLoader(dataset, batch_size=batch_size_eval, shuffle=False)

    closest_class_record = np.array([], dtype=np.long)
    drift_record = np.array([], dtype=np.long)
    label_class_record = np.array([], dtype=np.long)
    label_drift_record = np.array([], dtype=np.long)
    detector.eval()
    torch.set_grad_enabled(False)
    for X, y_class, y_drift in tqdm(data_loader, desc="Evaluate"):
        X, y_class, y_drift = X.to(detector.device), y_class.to(detector.device), y_drift.to(detector.device)
        
        closest_class = detector.closest_class(X)
        drift = detector(X)

        closest_class_record = np.concatenate([closest_class_record, closest_class.detach().cpu().numpy()], 0)
        drift_record = np.concatenate([drift_record, drift.detach().cpu().numpy()], 0)
        label_class_record = np.concatenate([label_class_record, y_class.detach().cpu().numpy()], 0)
        label_drift_record = np.concatenate([label_drift_record, y_drift.detach().cpu().numpy()], 0)

    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()

    _, m_class = calc_metrics_classifier_class(label_class_record, closest_class_record)
    save_idx = np.array([], dtype=np.long)
    for c in range(num_classes_train):
        save_idx = np.concatenate([save_idx, np.where(label_class_record == c)[0]], 0)
    acc_class, _ = calc_metrics_classifier_class(closest_class_record[save_idx], label_class_record[save_idx])

    print(('Accuracy (Class for training): %.4f') % (acc_class))
    print("Confusion matrix (Row: ground truth class, Col: prediction class):")
    for i in range(m_class.shape[0]):
        for j in range(m_class.shape[1]):
            if j < m_class.shape[1] - 1:
                print("%d " % (m_class[i, j]), end="")
            else:
                print("%d" % (m_class[i, j]))

    acc, p, r, f1 = calc_metrics_classifier(label_drift_record, drift_record)
    _, m_drift = calc_metrics_classifier_class(label_class_record, drift_record)
    m_drift = m_drift[:, :2]

    print("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1 Score: %.4f" % (acc, p, r, f1))
    print("Confusion matrix (Row: ground truth class, Col: drift result): ")
    for i in range(m_drift.shape[0]):
        for j in range(m_drift.shape[1]):
            if j < m_drift.shape[1] - 1:
                print("%d " % (m_drift[i, j]), end="")
            else:
                print("%d" % (m_drift[i, j]))

def certify():
    print("\n***** Optimize noise *****")

    param = torch.load(os.path.join(save_dir_train, "checkpoint-param"))
    encoder = torch.load(os.path.join(save_dir_train, "checkpoint-encoder"))
    detector = create_detector(encoder.to(device), param["centroid"].to(device), param["dis_median"].to(device), \
        param["mad"].to(device))

    noise_generator = create_noise_generator()

    criterion_shape = loss_functions["cade"](lambda_shape, mad_threshold)

    x_train, _, num_classes_train, class_map = load_data(data_dir, "train")
    print("Number of classes for training:", num_classes_train)
    print("Class Map from original class to new class (First %d classes are for training): %s" % (num_classes_train, str(class_map)))

    dataset = TensorDataset(x_train)
    data_loader = DataLoader(dataset, batch_size=batch_size_iteration_certify, shuffle=False)
    x_train_c = torch.tensor([], dtype=torch.float32)
    for (X,) in tqdm(data_loader, desc="Select class %d" % (certify_class)):
        X = X.to(device)
        closest_class = detector.closest_class(X)
        x_train_c = torch.cat([x_train_c, X[(closest_class == certify_class), :].detach().cpu()], 0)

    optimizing_noise(
        x_train_c,
        0, # Certifying known class
        detector,
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

    x_test, _, num_classes_train, class_map, y_test_drift = load_data(data_dir, "test")
    print("Number of classes for training:", num_classes_train)
    print("Class Map from original class to new class (First %d classes are for training): %s" % (num_classes_train, str(class_map)))

    dataset = TensorDataset(x_test, y_test_drift)
    data_loader = DataLoader(dataset, batch_size=batch_size_iteration_certify, shuffle=False)
    x_test_c = torch.tensor([], dtype=torch.float32)
    y_test_drift_c = torch.tensor([], dtype=torch.long)
    for (X, y) in tqdm(data_loader, desc="Select class %d" % (certify_class)):
        X = X.to(device)

        closest_class = detector.closest_class(X)

        x_test_c = torch.cat([x_test_c, X[(closest_class == certify_class), :].detach().cpu()], 0)
        y_test_drift_c = torch.cat([y_test_drift_c, y[(closest_class == certify_class)].detach().cpu()], 0)

    idx = torch.arange(x_test_c.shape[0])
    idx = idx[torch.randperm(idx.size(0))]
    idx = torch.sort(idx[:min(num_samples_certify, idx.shape[0])])[0]
    x_test_c, y_test_drift_c = x_test_c[idx, :], y_test_drift_c[idx]

    dataset = TensorDataset(x_test_c, y_test_drift_c)
    data_loader = DataLoader(dataset, batch_size=batch_size_iteration_certify, shuffle=False)

    detector.eval()
    smoothed_classifier = Smooth2(detector, d, num_classes_certify, noise_generator, device)
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

    max_robust_radius = {"newinfiltration-0": 0.04, "newinfiltration-1": 0.04, "newinfiltration-2": 0.015, "newhulk-2": 0.4}
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
    class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration"}
    plt.title("CADE %s" % (class_name["%s-%d" % (dataset_name, certify_class)]), fontsize=20)
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