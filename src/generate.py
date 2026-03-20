import torch
from src.tokenizer import CharTokenizer
from src.model import TransformerLM
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "movie_dialogues.txt")

text = open(DATA_PATH, encoding="utf-8").read()
tok = CharTokenizer(text)

model = TransformerLM(tok.vocab_size)
model.load_state_dict(torch.load("checkpoints/model.pt"))

context = torch.zeros((1,1), dtype=torch.long)

out = model.generate(context, 500, temperature=0.8)

print(tok.decode(out[0].tolist()))