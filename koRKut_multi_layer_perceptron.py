import torch
import torch.nn as nn

class koRKutLMP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, device="cpu"):
        super().__init__()
        self.gate_proj = nn.Linear(embedding_dim, hidden_dim, device=device)
        self.up_proj = nn.Linear(embedding_dim, hidden_dim, device=device)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim, device=device)
        self.gelu = nn.GELU()

    def forward(self, x):
        gate = self.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
