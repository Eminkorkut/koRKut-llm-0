import torch
import torch.nn as nn
from koRUkut_layer_norm import koRKutLayerNorm
from koRKut_multi_layer_perceptron import koRKutLMP
from koRKut_multi_head_attention import koRKutMultiHeadAttention

class koRKutDecoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int, device: str):
        super().__init__()
        self.self_attention = koRKutMultiHeadAttention(
            embedding_dim=embedding_dim,
            output_dim=embedding_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout_rate=0.5,
            device=device
        )
        self.norm1 = koRKutLayerNorm(embedding_dim, device=device)
        self.mlp = koRKutLMP(embedding_dim, embedding_dim, device=device)
        self.norm2 = koRKutLayerNorm(embedding_dim, device=device)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, seq_len, embedding_dim)
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """

        # Self-Attention block with pre-norm and residual connection
        x_norm = self.norm1(x)
        attn_out = self.self_attention(x_norm)
        attn_out = self.dropout1(attn_out)
        x = x + attn_out

        # MLP block with pre-norm and residual connection
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        mlp_out = self.dropout2(mlp_out)
        x = x + mlp_out

        return x
