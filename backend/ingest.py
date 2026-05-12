import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FILE = "profile_data.txt"
INDEX_FILE = "vector.index"
CHUNKS_FILE = "chunks.pkl"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=120):
    words = text.split()

    if not words:
        return []

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Create it in the backend folder.")

with open(DATA_FILE, "r", encoding="utf-8") as file:
    text = file.read().strip()

if not text:
    raise ValueError("profile_data.txt is empty. Add your resume/project/profile information first.")

chunks = chunk_text(text)

if not chunks:
    raise ValueError("No chunks were created. Check profile_data.txt content.")

print(f"Total chunks created: {len(chunks)}")

embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

if len(embeddings.shape) != 2:
    raise ValueError(f"Embedding shape is wrong: {embeddings.shape}")

embeddings = embeddings.astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)

with open(CHUNKS_FILE, "wb") as file:
    pickle.dump(chunks, file)

print("Vector database created successfully.")
print(f"Saved: {INDEX_FILE}")
print(f"Saved: {CHUNKS_FILE}")