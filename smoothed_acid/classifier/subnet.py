import numpy as np

import torch
import torch.nn as nn

class Sin_Weighted(nn.Module):
    def __init__(self, input_size: int):
        super(Sin_Weighted, self).__init__()

        self.wa = nn.Parameter(torch.zeros((input_size), requires_grad=True))
        self.wf = nn.Parameter(torch.zeros((1), requires_grad=True))

        self.weight_init()

    def weight_init(self):
        self.wa.data.normal_(0, 1)
        self.wf.data.normal_(0, 1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.wf * torch.sin(2 * np.pi * x * self.wa)

class Encoder(nn.Module):
    def __init__(self, input_size: int):
        super(Encoder, self).__init__()

        layers_dims = [500, 200, 50]

        kernel_size = 10

        self.encoder = nn.Sequential(
            nn.Linear(input_size, layers_dims[0]),
            Sin_Weighted(layers_dims[0]),
            nn.Linear(layers_dims[0], layers_dims[1]),
            Sin_Weighted(layers_dims[1]),
            nn.Linear(layers_dims[1], layers_dims[2]),
            Sin_Weighted(layers_dims[2]),
            nn.Linear(layers_dims[2], kernel_size)
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.encoder(x)

class Kernel_Net(nn.Module):
    def __init__(self, input_size: int):
        super(Kernel_Net, self).__init__()

        layers_dims = [100, 50, 30]

        self.kernel_net = nn.Sequential(
            nn.Linear(input_size, layers_dims[0]),
            nn.Tanh(),
            nn.Linear(layers_dims[0], layers_dims[1]),
            nn.Tanh(),
            nn.Linear(layers_dims[1], layers_dims[2]),
            nn.Tanh(),
            nn.Linear(layers_dims[2], 1)
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.kernel_net(x)

class AdaptiveClustering(nn.Module):
    def __init__(self, input_size: int, num_kernels: int, device: object):
        super(AdaptiveClustering, self).__init__()

        self.num_kernels = num_kernels

        kernel_size = 10

        self.encoder_list = nn.ModuleList([
            Encoder(input_size) for i in range(num_kernels)
        ])

        self.kernel_weight = nn.Parameter(torch.zeros((num_kernels, kernel_size), requires_grad=True))

        self.kernel_net_list = nn.ModuleList([
            Kernel_Net(kernel_size) for i in range(num_kernels)
        ])
        self.out_act = nn.Sigmoid()
        
        self.device = device

    def weight_init(self):
        self.kernel_weight.data.normal_(0, 1)

    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor, torch.tensor):
        z = torch.tensor([], device=self.device)
        o = torch.tensor([], device=self.device)
        for i in range(self.num_kernels):
            zi = self.encoder_list[i](x)
            z = torch.cat([z, zi.unsqueeze(1)], 1)
            o = torch.cat([o, self.out_act(self.kernel_net_list[i](zi))], 1)

        return z, self.kernel_weight, o

class Loss_AdaptiveClustering(nn.Module):
    def __init__(self, num_classes: int):
        super(Loss_AdaptiveClustering, self).__init__()

        self.num_classes = num_classes
        self.margin = 1

        self.criterion = nn.MSELoss()

    def forward(self, z: torch.tensor, kernel_weight: torch.tensor, o: torch.tensor, y: torch.tensor) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        assert self.num_classes == kernel_weight.size(0)

        y_one_hot = torch.zeros_like(o, dtype=torch.float).to(o.device)
        y_one_hot[torch.arange(o.size(0)).to(o.device), y] = 1
        loss_classification = self.criterion(o, y_one_hot)

        loss_clustering_close = torch.tensor(0.0, dtype=torch.float).to(z.device)
        for c in range(self.num_classes):
            loss_clustering_close += self.criterion(
                z[y == c, c, :], 
                kernel_weight[c, :].repeat((torch.sum(y == c), 1)))

        loss_clustering_dist = torch.tensor(0.0, dtype=torch.float).to(z.device)
        for c in range(self.num_classes):
            loss_clustering_dist += torch.clamp(self.margin - self.criterion(
                kernel_weight[torch.arange(self.num_classes).to(kernel_weight.device) != c, :],
                kernel_weight[c, :].clone().detach().repeat((torch.sum(torch.arange(self.num_classes).to(kernel_weight.device) != c), 1))),
                min=0)
        loss_clustering_dist /= self.num_classes

        return loss_classification + loss_clustering_close + loss_clustering_dist, \
            loss_classification, loss_clustering_close, loss_clustering_dist
