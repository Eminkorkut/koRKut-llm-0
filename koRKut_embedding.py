import torch
import torch.nn as nn

def rotary_position_encoding(seq_len, dim, base=10000, device="cpu"):
    assert dim % 2 == 0, "Dimension must be even for rotary encoding"

    half_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)

    angles = positions * inv_freq.unsqueeze(0)  # [seq_len, half_dim]
    sin = torch.sin(angles).unsqueeze(0)        # [1, seq_len, half_dim]
    cos = torch.cos(angles).unsqueeze(0)        # [1, seq_len, half_dim]

    return sin, cos

def apply_rotary(tensor, sin, cos):
    bsz, seq_len, dim = tensor.shape
    half_dim = dim // 2

    t1, t2 = tensor[..., :half_dim], tensor[..., half_dim:]
    t_rotated_1 = t1 * cos - t2 * sin
    t_rotated_2 = t1 * sin + t2 * cos

    return torch.cat([t_rotated_1, t_rotated_2], dim=-1)

class KoRKutEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int, base: int = 10000, device: str = "cpu"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.base = base
        self.device = device

        # Positional encodings are precomputed and cached
        sin, cos = rotary_position_encoding(context_length, embedding_dim, base, device)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, ids):
        if not isinstance(ids, torch.Tensor):
            raise TypeError("Input data must be in torch.Tensor format.")

        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        embeddings = self.embedding(ids)

        seq_len = embeddings.size(1)
        sin = self.sin[:, :seq_len]
        cos = self.cos[:, :seq_len]

        encoded = apply_rotary(embeddings, sin, cos)
        return encoded

