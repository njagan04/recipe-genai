import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import FAISS_INDEX_PATH, FAISS_META_PATH



model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

# Important for IVF
index.nprobe = 10

print("Loading metadata...")
with open(FAISS_META_PATH, "r") as f:
    recipes = json.load(f)


# ---------- NORMALIZATION ----------
def normalize_query(query: str):
    query = query.lower().strip()

    # remove commas and extra spaces
    query = query.replace(",", " ")
    query = " ".join(query.split())

    return query


# ---------- SEARCH ----------
def search(query, top_k=10):
    query = normalize_query(query)

    query_vec = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vec, top_k * 2)  
    # fetch more → then filter

    seen_titles = set()
    results = []

    for i in indices[0]:
        if i >= len(recipes):
            continue

        recipe = recipes[i]
        title = recipe["title"]

        if title not in seen_titles:
            results.append(recipe)
            seen_titles.add(title)

        if len(results) >= top_k:
            break

    return results