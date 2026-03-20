import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

inp = os.path.join(BASE_DIR, "data", "movie_lines.txt")
out = os.path.join(BASE_DIR, "data", "movie_dialogues.txt")

sentences = []

with open(inp, encoding="iso-8859-1") as f:
    for line in f:
        parts = line.split(" +++$+++ ")
        if len(parts) == 5:
            sentences.append(parts[-1].strip())

text = "\n".join(sentences)

with open(out, "w", encoding="utf-8") as f:
    f.write(text)

print("Dataset ready:", len(sentences), "lines")