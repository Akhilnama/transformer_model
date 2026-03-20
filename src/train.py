import torch
import os
from tqdm import tqdm

from src.tokenizer import CharTokenizer
from src.dataset import TextDataset
from src.model import TransformerLM


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "movie_dialogues.txt")

text = open(DATA_PATH, encoding="utf-8").read()

tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text))

train_data = data[:int(0.9*len(data))]
dataset = TextDataset(train_data, 128)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerLM(tok.vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for step in tqdm(range(1000)):

    xb, yb = dataset.get_batch(32)
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print("loss:", loss.item())
        torch.save(model.state_dict(), "checkpoints/model.pt")

torch.save(model.state_dict(), "checkpoints/model_final.pt")

context = torch.zeros((1,1), dtype=torch.long).to(device)
out = model.generate(context, 400, temperature=0.8)

print(tok.decode(out[0].tolist()))