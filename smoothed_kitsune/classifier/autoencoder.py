import numpy as np

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, feature_size: float, device: object):
        super(Autoencoder, self).__init__()

        self.ae = nn.Sequential(
            nn.Linear(feature_size, int(np.ceil(feature_size / 2))),
            nn.ReLU(inplace=True),
            nn.Linear(int(np.ceil(feature_size / 2)), int(np.ceil(feature_size / 4))),
            nn.Linear(int(np.ceil(feature_size / 4)), int(np.ceil(feature_size / 2))),
            nn.ReLU(inplace=True),
            nn.Linear(int(np.ceil(feature_size / 2)), feature_size))

        self.device = device

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.ae(x)