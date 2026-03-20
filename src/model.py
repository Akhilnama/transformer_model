import torch
import torch.nn as nn
from src.transformer import TransformerBlock

class TransformerLM(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.block_size = 128
        self.d_model = 256

        self.token_emb = nn.Embedding(vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.block_size, self.d_model)

        self.blocks = nn.Sequential(
            *[TransformerBlock(self.d_model, 4) for _ in range(4)]
        )

        self.ln = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T))

        x = tok + pos

        x = self.blocks(x)
        x = self.ln(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):

        for _ in range(max_new_tokens):

            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx