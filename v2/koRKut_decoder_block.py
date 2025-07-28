import torch
import torch.nn as nn
from koRKut_layer_norm import koRKutLayerNorm
from koRKut_multi_head_attention import koRKutMultiHeadAttention
from koRKut_multi_layer_perceptron import koRKutMultiLayerPerceptron

class koRKutDecoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int, device: str="cpu"):
        super().__init__()

        self.self_attention = koRKutMultiHeadAttention(
            embedding_dim=embedding_dim,
            output_dim=embedding_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout_rate=0.5,
            device = device
        )

        self.norm_layer1 = koRKutLayerNorm(embedding_dim=embedding_dim,device=device)

        self.multiLayerPerceptron = koRKutMultiLayerPerceptron(embedding_dim=embedding_dim, hidden_dim=embedding_dim, device=device)

        self.norm_layer2 = koRKutLayerNorm(embedding_dim=embedding_dim, device=device)

    def forward(self, x):
        res = self.norm_layer1(x)
        x = self.self_attention(x)
        x = self.norm_layer1(x)

        x = x + res

        res = self.norm_layer2(x)
        x = self.multiLayerPerceptron(x)
        x = self.norm_layer2(x)

        x = x + res

        return x