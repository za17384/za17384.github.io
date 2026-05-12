import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env")

if not PINECONE_INDEX_NAME:
    raise ValueError("Missing PINECONE_INDEX_NAME in .env")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

# Groq LLM client
groq_client = Groq(api_key=GROQ_API_KEY)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://za17384.github.io",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


def search_context(question, top_k=1):
    question_embedding = embedding_model.encode(question).tolist()

    results = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True
    )

    contexts = []

    for match in results.matches:
        if match.metadata and "text" in match.metadata:
            contexts.append(match.metadata["text"])

    return "\n\n".join(contexts)


@app.get("/")
def home():
    return {
        "message": "Pinecone + Groq chatbot backend is running."
    }


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        question = request.question.strip()

        if not question:
            return {"answer": "Please ask a question."}

        context = search_context(question)

        if not context:
            return {"answer": "I don't have that information yet."}

        prompt = f"""
You are Zeeshan Ali's personal AI chatbot for his portfolio website.

Use ONLY the information below to answer.
If the answer is not available, say: "I don't have that information yet."

Information:
{context}

Question:
{question}
"""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=300
        )

        answer = response.choices[0].message.content

        return {"answer": answer}

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "answer": f"Backend error: {str(e)}"
        }