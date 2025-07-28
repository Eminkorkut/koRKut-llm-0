import torch
import torch.nn as nn

class koRKutMultiLayerPerceptron(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, device: str="cpu"):
        super().__init__()

        self.gate_proj = nn.Linear(in_features=embedding_dim, out_features=hidden_dim, device=device, bias=False)
        self.up_proj = nn.Linear(in_features=embedding_dim, out_features=hidden_dim, device=device, bias=False)
        self.down_proj = nn.Linear(in_features=hidden_dim, out_features=embedding_dim, device=device, bias=True)
        self.silu = nn.SiLU().to(device=device)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = self.silu(gate)
        up = self.up_proj(x)
        fuse = gate * up
        output = self.down_proj(fuse)
        
        return output