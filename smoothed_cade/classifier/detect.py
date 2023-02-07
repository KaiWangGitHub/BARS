import torch
import torch.nn as nn

class Detector(nn.Module):
    def __init__(self, encoder, centroid, dis_median, mad, mad_threshold, device):
        super(Detector, self).__init__()

        self.encoder = encoder

        self.centroid = centroid
        self.dis_median = dis_median
        self.mad = mad
        self.mad_threshold = mad_threshold

        self.device = device

    def forward(self, x: torch.tensor, noise: torch.tensor=0) -> torch.tensor:
        centroid_class = self.closest_class(x)
        _, drift = self._score_drift_under_centroid_class(x + noise, centroid_class)

        return drift

    def score(self, x: torch.tensor, noise: torch.tensor=0) -> torch.tensor:
        centroid_class = self.closest_class(x)
        score, _ = self._score_drift_under_centroid_class(x + noise, centroid_class)

        return score

    def closest_class(self, x: torch.tensor) -> torch.tensor:
        z = self.encoder(x)

        dis = torch.norm(z.unsqueeze(1).repeat((1, self.centroid.size(0), 1)) - \
            self.centroid.unsqueeze(0).repeat((z.size(0), 1, 1)), p=2, dim=2)
        closest_class = dis.argmin(1)

        return closest_class

    # We adjust CADE by detecting drift based on the closest classes of the original samples for ease of illustration.
    def _score_drift_under_centroid_class(self, x: torch.tensor, centroid_class: torch.tensor) -> (torch.tensor, torch.tensor):
        assert x.size(0) == centroid_class.size(0)

        z = self.encoder(x)
        dis = torch.norm(z - self.centroid[centroid_class, :], p=2, dim=1)
        score = torch.abs(dis - self.dis_median[centroid_class]) / self.mad[centroid_class]
        drift = torch.zeros_like(score, dtype=torch.long)
        drift[score > self.mad_threshold] = 1

        return score, drift

def calc_centroid_dis_median_mad(z_train: torch.tensor, y_train: torch.tensor):
    class_list = torch.unique(y_train).tolist()

    centroid_record = torch.tensor([], device=z_train.device)
    for i in range(len(class_list)):
        centroid_record = torch.cat([centroid_record, z_train[y_train == class_list[i], :].mean(0).unsqueeze(0)], 0)

    dis_median_record = torch.tensor([], device=z_train.device)
    mad_record = torch.tensor([], device=z_train.device)
    for i in range(len(class_list)):
        dis = torch.norm(z_train[y_train == class_list[i], :] - centroid_record[class_list[i], :], p=2, dim=1)
        dis_median = torch.median(dis)
        mad = 1.4826 * torch.median(torch.abs(dis - dis_median))
        dis_median_record = torch.cat([dis_median_record, dis_median.unsqueeze(0)], 0)
        mad_record = torch.cat([mad_record, mad.unsqueeze(0)], 0)

    return centroid_record, dis_median_record, mad_record