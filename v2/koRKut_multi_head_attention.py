import torch.nn as nn
import torch

class koRKutMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, context_length: int, num_heads: int, dropout_rate: float=0.1, device: str="cpu"):
        super().__init__()

        self.context_length = context_length

        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout_rate, device=device, bias=False)
        self.projection = nn.Linear(in_features=embedding_dim, out_features=output_dim, device=device, bias=False)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool().to(device=device))

    def forward(self, x):
        number_of_tokens = x.shape[0]
        x = x[:self.context_length]
        attention_mask = self.mask[:number_of_tokens, :number_of_tokens]
        out, _ = self.multi_head_attention(x, x, x, attn_mask = attention_mask)
        out = self.projection(out)

        return out
