from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from knowledge_base import documents

model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for documents
texts = list(documents.values())
embeddings = model.encode(texts, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(embeddings)

# Save index and mapping
faiss.write_index(index, "faiss_index.idx")
import pickle
with open("doc_mapping.pkl", "wb") as f:
    pickle.dump(list(documents.keys()), f)