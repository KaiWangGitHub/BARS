import torch
import torch.nn as nn

def Encoder(input_size: int, output_size: int):

    dims = [input_size, 64, 32, 16, output_size]

    return nn.Sequential(
        nn.Linear(dims[0], dims[1]),
        nn.ReLU(inplace=True),
        nn.Linear(dims[1], dims[2]),
        nn.ReLU(inplace=True),
        nn.Linear(dims[2], dims[3]),
        nn.ReLU(inplace=True),
        nn.Linear(dims[3], dims[4])
    )

def Decoder(input_size: int, output_size: int):

    dims = [input_size, 16, 32, 64, output_size]

    return nn.Sequential(
        nn.Linear(dims[0], dims[1]),
        nn.ReLU(inplace=True),
        nn.Linear(dims[1], dims[2]),
        nn.ReLU(inplace=True),
        nn.Linear(dims[2], dims[3]),
        nn.ReLU(inplace=True),
        nn.Linear(dims[3], dims[4])
    )

class Autoeconder(nn.Module):
    def __init__(self, input_size: int, num_classes: int, device: object):
        super().__init__()

        self.encoder = Encoder(input_size, num_classes)
        self.decoder = Decoder(num_classes, input_size)

        self.device = device

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return z, x_reconstructed

class Loss_Contrastive(nn.Module):
    def __init__(self, margin: float, lambda_: float):
        super().__init__()

        self.criterion_reconstructed = nn.MSELoss(reduction="mean")

        self.margin = margin
        self.lambda_ = lambda_

        self.eps = 1e-10

    def forward(self, x: torch.tensor, z: torch.tensor, x_reconstructed: torch.tensor, y: torch.tensor):
        loss_reconstructed = self.criterion_reconstructed(x, x_reconstructed)

        left_idx = torch.arange(0, int(x.size(0) / 2)).to(x.device)
        right_idx = torch.arange(int(x.size(0) / 2), int(x.size(0) / 2) * 2).to(x.device)
        dist = torch.sqrt(torch.sum(torch.pow(z[left_idx] - z[right_idx], 2), 1) + self.eps)
        in_same_class = torch.zeros_like(dist).to(x.device)
        in_same_class[y[left_idx] == y[right_idx]] = 1
        in_diff_class = torch.zeros_like(dist).to(x.device)
        in_diff_class[y[left_idx] != y[right_idx]] = 1
        loss_contrastive = torch.mean(in_same_class * dist + in_diff_class * nn.ReLU(inplace=True)(self.margin - dist))

        return loss_reconstructed + self.lambda_ * loss_contrastive, loss_reconstructed, loss_contrastive