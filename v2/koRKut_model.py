import torch
import torch.nn as nn
from koRKut_embedding import koRKutEmbedding
from koRKut_decoder_block import koRKutDecoderBlock

class koRKutModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, context_length: int, num_layers: int, device: str="cpu"):
        super().__init__()

        self.embedding = koRKutEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim, context_length=context_length, device=device)
        self.layers = nn.Sequential(
            *[koRKutDecoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, context_length=context_length, device=device) for _ in range(num_layers)]
        )

        self.lm_head = nn.Linear(in_features=embedding_dim, out_features=vocab_size, device=device)
        self.device = device

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        x = self.lm_head(x)

        return x
    
    def top_p_filtering(self, logits, top_p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        sorted_logits[sorted_indices_to_remove] = -float('inf')
        filtered_logits = sorted_logits.clone()
        filtered_logits.scatter_(0, sorted_indices, sorted_logits)
        return filtered_logits
    
    def generate(self, 
            x: torch.Tensor,
            max_new_tokens: int=3,
            temperature: float = 1.0,
            top_k: int = 64,
            top_p: float = 1.0
            ): # top_k, top_p, temperature
        tokens = x.tolist()

        for _ in range(max_new_tokens):
            x = x.unsqueeze(0).to(self.device)
            out = self.forward(x)
            out = out.squeeze(0)
            logits = out[-1]
            if top_k > 0:
                values, indexes = torch.topk(logits, k=top_k)
                logits = torch.full_like(logits, -float('inf'))
                logits.scatter_(0, indexes, values)

            if top_p > 0 and top_p < 1:
                logits = self.top_p_filtering(logits, top_p)

            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature
            
            probs = torch.softmax(values, dim=-1)
            # _, max_index = torch.max(probs, dim=-1)
            sample = torch.multinomial(probs, 1)
            max_index = indexes[sample]
            tokens.append(max_index.item())
            
            x = torch.tensor(tokens)

        return tokens
