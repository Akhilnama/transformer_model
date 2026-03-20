import torch
from src.tokenizer import CharTokenizer
from src.dataset import TextDataset
from src.model import TransformerLM
from tqdm import tqdm

text = open("data/shakespeare.txt").read()

tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text))

train_data = data[:int(0.9*len(data))]
dataset = TextDataset(train_data, 128)

model = TransformerLM(tok.vocab_size)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for step in tqdm(range(5000)):

    xb, yb = dataset.get_batch(32)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print("loss:", loss.item())
        torch.save(model.state_dict(), "checkpoints/model.pt")