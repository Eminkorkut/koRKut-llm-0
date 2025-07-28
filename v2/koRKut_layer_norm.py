import torch
import torch.nn as nn

class koRKutLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps=1e-5, device: str="cpu"):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim, device=device))
        self.device = device

    def forward(self, x):
        if not x.is_floating_point():
            x = x.float()
        mean = x.mean(dim= -1, keepdim= True)
        variance = x.var(dim= -1, keepdim= True, unbiased = False)
        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)

        return self.weight * normalized_x