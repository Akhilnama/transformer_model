import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):

    def __init__(self, d_model, head_size):
        super().__init__()

        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(1024, 1024))
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei / math.sqrt(C)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)

        v = self.value(x)

        out = wei @ v
        return out, wei