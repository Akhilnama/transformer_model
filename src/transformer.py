import torch
import torch.nn as nn
from src.attention import SelfAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        head_size = d_model // num_heads
        self.heads = nn.ModuleList(
            [SelfAttention(d_model, head_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):

        outs = [h(x)[0] for h in self.heads]
        out = torch.cat(outs, dim=-1)
        return self.proj(out)


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.sa = MultiHeadAttention(d_model, num_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x