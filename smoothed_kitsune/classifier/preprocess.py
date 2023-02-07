from tqdm import trange
from typing import *

import torch
import torch.nn as nn

from classifier.corClust import corClust

class Normalizer(nn.Module):
    def __init__(self, device: object):
        super(Normalizer, self).__init__()
        
        self.norm_max = None
        self.norm_min = None

        self.device = device
        self.eps = 1e-16

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.norm_max == None or self.norm_min == None:
            self.norm_max = torch.max(x, 0)[0].to(self.device)
            self.norm_min = torch.min(x, 0)[0].to(self.device)

        return (x - self.norm_min) / (self.norm_max - self.norm_min + self.eps)

    def update(self, x: torch.tensor):
        if self.norm_max == None or self.norm_min == None:
            self.norm_max = torch.max(x, 0)[0].to(self.device)
            self.norm_min = torch.min(x, 0)[0].to(self.device)
        else:
            self.norm_max = torch.max(torch.cat([x, self.norm_max.unsqueeze(0)], 0), 0)[0]
            self.norm_min = torch.min(torch.cat([x, self.norm_min.unsqueeze(0)], 0), 0)[0]

class Feature_Map(nn.Module):
    def __init__(self, device: object):
        super(Feature_Map, self).__init__()

        self.mp = None

        self.device = device

    def init(self, x: torch.tensor, maxClust: int=10):
        c = corClust(x.size(1))
        for i in trange(x.size(0), desc="Feat Clust"):
            c.update(x[i, :].detach().cpu().numpy())
        self.mp = c.cluster(maxClust)
    
    def forward(self, x: torch.tensor) -> List:
        if self.mp == None:
            self.init(x)

        x_mapped = []
        for m in self.mp:
            x_mapped.append(x[:, m])

        return x_mapped

    def get_num_clusters(self) -> int:
        return len(self.mp)