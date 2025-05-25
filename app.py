from fastapi import  Fast
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

# Load FAISS index and doc mapping
index = faiss.read_index("faiss_index.idx")
with open("doc_mapping.pkl", "rb") as f:
    doc_ids = pickle.load(f)

# Load documents
from knowledge_base import documents

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

class Query(BaseModel):
    question: str

def generate_answer(question, context_docs, max_length=150):
    context = " ".join(context_docs)
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

@app.post("/chat/")
async def chat(query: Query):
    q_embedding = embed_model.encode([query.question], convert_to_numpy=True)
    D, I = index.search(q_embedding, k=3)  # top 3 relevant docs
    relevant_docs = [documents[doc_ids[i]] for i in I[0]]
    answer = generate_answer(query.question, relevant_docs)
    return {"answer": answer}