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
from torch.utils.data import DataLoader, TensorDataset

from classifier.autoencoder import Autoencoder
from classifier.detect import Detector
from classifier.preprocess import Normalizer, Feature_Map
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
dataset_name = "ssdp" # The name of data set. e.g., "mirai" "ssdp"
if dataset_name not in os.listdir(data_dir):
    raise ValueError("Dataset not found: %s" % (dataset_name))
data_dir = os.path.join(data_dir, dataset_name)

# Building classifier
learning_rate_classifier = 1e-3 # Learning rate for training classifier
ad_threshold_dict = {"mirai": 0.1, "ssdp": 0.3}
ad_threshold = ad_threshold_dict[dataset_name] # Threshold for anomaly detection
num_epochs_classifier = 30 # Epoch number for training classifier
batch_size_train = 512 # Batch size for training classifier
batch_size_eval = 2048 # Batch size for evaluating classifier
save_dir_train = "save/%s/train" % (dataset_name) # Saving directory for classifier training
if not os.path.exists(save_dir_train):
    os.makedirs(save_dir_train)
print_step_classifier = 10 # Step size for showing training progress

# Certifying classifier
certify_class = 0 # The certified class (benign class)
feature_noise_distribution = "isru" # Feature noise distribution. e.g., "gaussian" "isru" "isru_gaussian" "isru_gaussian_arctan"
learning_rate_shape = 1e-2 # Learning rate for optimzing noise shape
nt_shape = 1000 # Number of noised samples for optimzing noise shape
lambda_shape = 1e-3 # Regularizer weight
num_epochs_shape = 5 # Number of epochs for optimzing noise shape
x_train = load_data(data_dir, "train")
d = x_train.size(1) # Number of feature dimensions
num_classes_certify = 2 # Number of certified classes. i.e., Benign(0), Anomaly(1)
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
num_samples_certify = 1000 # Number of certified samples sampled from dataset

seed = 42 # Random seed

def set_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def create_ae(feature_size: int) -> nn.Module:
    return Autoencoder(feature_size, device)

def create_normalizer():
    return Normalizer(device)

def create_fm():
    return Feature_Map(device)

def create_detector(normalizer: nn.Module, fm: nn.Module, ae: nn.Module) -> nn.Module:
    return Detector(normalizer, fm, ae, ad_threshold, device)

def create_noise_generator():
    dist_trans = distribution_transformers[feature_noise_distribution](d).to(device)
    return Noise(dist_trans, d, device)

def train():
    print("\n***** Run clustering *****")

    x_cluster = load_data(data_dir, "cluster")
    fm = create_fm()
    fm.init(x_cluster)

    torch.save(fm.mp, os.path.join(save_dir_train, "checkpoint-fm"))

    criterion = nn.MSELoss()

    x_train = load_data(data_dir, "train")
    dataset = TensorDataset(x_train)
    data_loader = DataLoader(dataset, batch_size=batch_size_train, shuffle=False)

    normalizer = create_normalizer()
    normalizer.update(x_train) 
    torch.save({"norm_max": normalizer.norm_max, "norm_min": normalizer.norm_min
        }, os.path.join(save_dir_train, "checkpoint-norm"))

    for i in range(fm.get_num_clusters()):
        print("\n***** Run training AE %d *****" % (i))

        ae = create_ae(len(fm.mp[i]))
        ae.to(device)

        opt = optim.Adam(ae.parameters(), lr=learning_rate_classifier)

        ae.train()
        for epoch in range(num_epochs_classifier + 1):
            loss_record = AverageMeter()
            for (X,) in data_loader:
                X = X.to(ae.device)

                X = normalizer(X)
                X = X[:, fm.mp[i]]
                x_reconstructed = ae(X)
                loss = criterion(X, x_reconstructed)

                if epoch > 0:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                loss_record.update(loss.item())

            if epoch % print_step_classifier == 0:
                print(('Epoch: [%d/%d] | MSE Loss (Avg): %.6f') % ( \
                    epoch, num_epochs_classifier, loss_record.avg))

        torch.save(ae.ae, os.path.join(save_dir_train, "checkpoint-ae-%d" % (i)))

def evaluate():
    print("\n***** Run evaluating *****")

    normalizer = create_normalizer()
    norm_param = torch.load(os.path.join(save_dir_train, "checkpoint-norm"))
    normalizer.norm_max = norm_param["norm_max"].to(device)
    normalizer.norm_min = norm_param["norm_min"].to(device)    
    fm = create_fm()
    fm.mp = torch.load(os.path.join(save_dir_train, "checkpoint-fm"))
    detector = create_detector(normalizer, fm, nn.ModuleList([torch.load(os.path.join(save_dir_train, "checkpoint-ae-%d" % (i)
        )).to(device) for i in range(fm.get_num_clusters())]))

    x_test, y_test = load_data(data_dir, "test")

    dataset = TensorDataset(x_test, y_test)
    data_loader = DataLoader(dataset, batch_size=batch_size_eval, shuffle=False)

    rmse_record = np.array([], dtype=np.float32)
    pred_record = np.array([], dtype=np.long)
    label_record = np.array([], dtype=np.long)
    detector.eval()
    torch.set_grad_enabled(False)
    for X, y in tqdm(data_loader, desc="Evaluate"):
        X, y = X.to(detector.device), y.to(detector.device)

        pred = detector(X)
        rmse = detector.score(X)

        rmse_record = np.concatenate([rmse_record, rmse.detach().cpu().numpy()], 0)
        pred_record = np.concatenate([pred_record, pred.detach().cpu().numpy()], 0)

        label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()

    acc, p, r, f1 = calc_metrics_classifier(pred_record, label_record)

    print("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1 Score: %.4f" % (acc, p, r, f1))

def certify():
    print("\n***** Optimize noise *****")

    normalizer = create_normalizer()
    norm_param = torch.load(os.path.join(save_dir_train, "checkpoint-norm"))
    normalizer.norm_max = norm_param["norm_max"].to(device)
    normalizer.norm_min = norm_param["norm_min"].to(device)    
    fm = create_fm()
    fm.mp = torch.load(os.path.join(save_dir_train, "checkpoint-fm"))
    detector = create_detector(normalizer, fm, nn.ModuleList([torch.load(os.path.join(save_dir_train, "checkpoint-ae-%d" % (i)
        )).to(device) for i in range(fm.get_num_clusters())]))

    noise_generator = create_noise_generator()

    criterion_shape = loss_functions["kitsune"](lambda_shape, ad_threshold)

    x_train = load_data(data_dir, "train")

    dataset = TensorDataset(x_train)
    data_loader = DataLoader(dataset, batch_size=batch_size_iteration_certify, shuffle=False)
    x_train_c = torch.tensor([], dtype=torch.float32)
    for (X,) in tqdm(data_loader, desc="Select class %d" % (certify_class)):
        X = X.to(device)

        anomaly = detector(X)

        x_train_c = torch.cat([x_train_c, X[(anomaly == certify_class), :].detach().cpu()], 0)

    optimizing_noise(
        x_train_c,
        certify_class,
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

    x_test, y_test = load_data(data_dir, "test")

    dataset = TensorDataset(x_test, y_test)
    data_loader = DataLoader(dataset, batch_size=batch_size_iteration_certify, shuffle=False)
    x_test_c = torch.tensor([], dtype=torch.float32)
    y_test_c = torch.tensor([], dtype=torch.long)
    for (X, y) in tqdm(data_loader, desc="Select class %d" % (certify_class)):
        X = X.to(device)

        anomaly = detector(X)

        x_test_c = torch.cat([x_test_c, X[(anomaly == certify_class), :].detach().cpu()], 0)
        y_test_c = torch.cat([y_test_c, y[(anomaly == certify_class)].detach().cpu()], 0)

    idx = torch.arange(x_test_c.shape[0])
    idx = idx[torch.randperm(idx.size(0))]
    idx = torch.sort(idx[:min(num_samples_certify, idx.shape[0])])[0]
    x_test_c, y_test_c = x_test_c[idx, :], y_test_c[idx]

    dataset = TensorDataset(x_test_c, y_test_c)
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

    max_robust_radius = {"mirai": 0.6, "ssdp": 1.5}
    robust_radius_plot = np.arange(0, max_robust_radius[dataset_name], max_robust_radius[dataset_name] * 1e-3)
    certified_accuracy_plot = np.array([], dtype=np.float32)
    for r in robust_radius_plot:
        certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
    plt.figure(1)
    plt.plot(robust_radius_plot, certified_accuracy_plot)
    plt.ylim((0, 1))
    plt.xlim((0, max_robust_radius[dataset_name]))
    plt.tick_params(labelsize=14)
    plt.xlabel("Robustness Radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.title("Kitsune %s" % (dataset_name), fontsize=20)
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