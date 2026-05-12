import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env")

if not PINECONE_INDEX_NAME:
    raise ValueError("Missing PINECONE_INDEX_NAME in .env")

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

DATA_FILE = "profile_data.txt"


def chunk_text(text, chunk_size=120):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size]).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found.")

with open(DATA_FILE, "r", encoding="utf-8") as file:
    text = file.read().strip()

if not text:
    raise ValueError("profile_data.txt is empty.")

chunks = chunk_text(text)

vectors = []

for i, chunk in enumerate(chunks):
    embedding = embedding_model.encode(chunk).tolist()

    vectors.append({
        "id": f"chunk-{i}",
        "values": embedding,
        "metadata": {
            "text": chunk
        }
    })

print("Uploading vectors to Pinecone...")

index.upsert(vectors=vectors)

print("Upload complete.")
print(f"Total chunks uploaded: {len(vectors)}")