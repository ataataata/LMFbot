# build_index.py
import json, numpy as np, ollama

oll = ollama.Client()
qa   = json.load(open("qa.json"))

def embed(t: str):
    return np.array(
        oll.embeddings(model="nomic-embed-text", prompt=t)["embedding"],
        dtype=np.float32
    )

vecs = np.vstack([embed(row["q"]) for row in qa])
np.save("vecs.npy", vecs)
print("Saved", vecs.shape, "→ vecs.npy")
