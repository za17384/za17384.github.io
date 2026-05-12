import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")

if not PINECONE_INDEX_NAME:
    raise ValueError("Missing PINECONE_INDEX_NAME")

# Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Your PDF file
PDF_FILE = "resume.pdf"


def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=120):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size]).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


if not os.path.exists(PDF_FILE):
    raise FileNotFoundError(f"{PDF_FILE} not found.")

print("Extracting PDF text...")

pdf_text = extract_pdf_text(PDF_FILE)

if not pdf_text.strip():
    raise ValueError("No text extracted from PDF.")

chunks = chunk_text(pdf_text)

print(f"Total chunks created: {len(chunks)}")

vectors = []

for i, chunk in enumerate(chunks):

    embedding = embedding_model.encode(chunk).tolist()

    vectors.append({
        "id": f"pdf-chunk-{i}",
        "values": embedding,
        "metadata": {
            "text": chunk
        }
    })

print("Uploading to Pinecone...")

index.upsert(vectors=vectors)

print("Upload complete.")
print(f"Total vectors uploaded: {len(vectors)}")