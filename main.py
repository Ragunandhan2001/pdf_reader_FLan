from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM model (better than GPT2)
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

VECTOR_DB_PATH = "vector_store/index.faiss"
CHUNK_FILE = "chunks.txt"


# -------------------------
# Extract text from PDF
# -------------------------
def extract_text(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


# -------------------------
# Split text into chunks
# -------------------------
def chunk_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks


# -------------------------
# Upload PDF
# -------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    text = extract_text(path)

    chunks = chunk_text(text)

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    faiss.write_index(index, VECTOR_DB_PATH)

    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c + "\n")

    return {"message": "PDF processed successfully"}


# -------------------------
# Question Request Model
# -------------------------
class Question(BaseModel):
    question: str


# -------------------------
# Ask Question
# -------------------------
@app.post("/ask")
def ask_question(data: Question):

    question = data.question

    index = faiss.read_index(VECTOR_DB_PATH)

    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = f.readlines()

    query_vector = embedding_model.encode([question])

    D, I = index.search(np.array(query_vector), k=3)

    context = " ".join([chunks[i] for i in I[0]])

    prompt = f"""
        Answer the question based only on the context.

        Context:
        {context}

        Question:
        {question}
        """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(**inputs, max_new_tokens=120)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer}