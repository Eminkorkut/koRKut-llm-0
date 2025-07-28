import torch.nn as nn
import torch

def get_rotary_position_encoding(input: torch.Tensor, base=10000, device: str="cpu"):
    batch_size, context_length, dimension = input.shape

    assert dimension % 2 == 0, "dimension must be even"

    half_dimension = dimension // 2

    freqs_indices = torch.arange(0, half_dimension, device=device, dtype=torch.float32)

    freqs = 1.0 / (base ** (freqs_indices / dimension))

    positions = torch.arange(0, context_length, device=device, dtype=torch.float32).unsqueeze(1)

    angles = positions * freqs

    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    input_even = input[:, :, :half_dimension]
    input_odd = input[:, :, half_dimension:]

    input_even_rotated = input_even * cos_angles - input_odd * sin_angles
    input_odd_rotated = input_even * sin_angles + input_odd * cos_angles

    input_rotated = torch.empty_like(input, device=device)

    input_rotated[:, :, :half_dimension] = input_even_rotated
    input_rotated[:, :, half_dimension:] = input_odd_rotated

    return input_rotated
    


class koRKutEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int, device: str="cpu"):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, device=device)
        self.get_pos = get_rotary_position_encoding
        self.device = device

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        x = x.to(self.embedding.weight.device)
        x = self.embedding(x)
        x = self.get_pos(x, device=self.device)

        return x
