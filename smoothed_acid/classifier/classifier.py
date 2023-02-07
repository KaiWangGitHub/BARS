import torch
import torch.nn as nn

class AdaptiveClusteringClassifier(nn.Module):
    def __init__(self, encoder_list, kernel_net_list, device):
        super(AdaptiveClusteringClassifier, self).__init__()

        assert len(encoder_list) == len(kernel_net_list)

        self.encoder_list = encoder_list
        self.kernel_net_list = kernel_net_list
        self.num_kernels = len(encoder_list)

        self.device = device

    def forward(self, x: torch.tensor, noise: torch.tensor=0) -> torch.tensor:
        o = torch.tensor([], device=self.device)
        for i in range(self.num_kernels):
            o = torch.cat([o, self.kernel_net_list[i](self.encoder_list[i](x + noise))], 1)

        return o.argmax(1)
    
    def score(self, x: torch.tensor, noise: torch.tensor=0) -> torch.tensor:
        o = torch.tensor([], device=self.device)
        for i in range(self.num_kernels):
            o = torch.cat([o, self.kernel_net_list[i](self.encoder_list[i](x + noise))], 1)

        return o