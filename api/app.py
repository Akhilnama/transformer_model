import torch
from fastapi import FastAPI
from pydantic import BaseModel

from src.model import TransformerLM
from src.tokenizer import CharTokenizer

text = open("data/shakespeare.txt").read()
tok = CharTokenizer(text)

model = TransformerLM(tok.vocab_size)
model.load_state_dict(torch.load("checkpoints/model.pt"))
model.eval()

app = FastAPI()

class Prompt(BaseModel):
    text: str
    temperature: float = 0.8
    max_tokens: int = 200


@app.post("/generate")
def generate(p: Prompt):

    idx = torch.tensor([tok.encode(p.text)])

    out = model.generate(
        idx,
        max_new_tokens=p.max_tokens,
        temperature=p.temperature
    )

    return {
        "response": tok.decode(out[0].tolist())
    }