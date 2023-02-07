import torch
import torch.nn as nn

class Detector(nn.Module):
    def __init__(self, normalizer: nn.Module, fm: nn.Module, ae: nn.Module, ad_threshold: float, device: object):
        super(Detector, self).__init__()

        self.normalizer = normalizer
        self.fm = fm
        self.ae = ae

        self.ad_threshold = ad_threshold

        self.device = device
        self.eps = 1e-16

    def forward(self, x: torch.tensor, noise: torch.tensor=0) -> torch.tensor:
        _, anomaly = self._predict_rmse_anomaly(x, noise)

        return anomaly
    
    def score(self, x: torch.tensor, noise: torch.tensor=0) -> torch.tensor:
        rmse, _ = self._predict_rmse_anomaly(x, noise)

        return rmse

    # We simplify Kitsune by removing the output layer autoencoder for ease of illustration.
    def _predict_rmse_anomaly(self, x: torch.tensor, noise: torch.tensor) -> (torch.tensor, torch.tensor):
        rmse = torch.zeros((x.size(0)), dtype=torch.float).to(self.device)
        x = self.normalizer(x)
        x += noise
        x = self.fm(x)
        for i in range(self.fm.get_num_clusters()):
            x_reconstructed = self.ae[i](x[i])
            rmse += self._rmse(x[i], x_reconstructed)
        rmse /= self.fm.get_num_clusters()

        anomaly = torch.zeros_like(rmse, dtype=torch.long)
        anomaly[rmse > self.ad_threshold] = 1

        return rmse, anomaly

    def _rmse(self, x: torch.tensor, x_reconstruct: torch.tensor) -> torch.tensor:
        return torch.sqrt(torch.mean(torch.pow((x - x_reconstruct), 2), dim=1) + self.eps)