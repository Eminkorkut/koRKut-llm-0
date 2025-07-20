import torch
import torch.nn as nn

class koRKutMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        context_length: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        device: str = "cpu",
        use_causal_mask: bool = True
    ):
        super().__init__()
        self.context_length = context_length
        self.device = device
        self.use_causal_mask = use_causal_mask

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
            device=device
        )
        self.projection = nn.Linear(embedding_dim, output_dim, device=device)

        if use_causal_mask:
            mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
            self.register_buffer("mask", mask.to(device))
        else:
            self.register_buffer("mask", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: Output tensor shape (batch_size, seq_len, output_dim)
        """

        batch_size, seq_len, _ = x.shape

        if seq_len > self.context_length:
            x = x[:, :self.context_length, :]
            seq_len = self.context_length

        attn_mask = self.mask[:seq_len, :seq_len] if self.use_causal_mask else None

        # nn.MultiheadAttention expects input shape (batch, seq, embed) if batch_first=True
        attn_output, _ = self.multi_head_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask
        )

        out = self.projection(attn_output)

        return out
