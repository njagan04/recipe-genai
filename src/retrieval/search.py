import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import FAISS_INDEX_PATH, FAISS_META_PATH



print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

# Important for IVF
index.nprobe = 32

if index.d == 768:
    EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
elif index.d == 384:
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
else:
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Loading metadata...")
with open(FAISS_META_PATH, "r") as f:
    recipes = json.load(f)

print(f"Search embedding model: {EMBEDDING_MODEL_NAME}")


# ---------- NORMALIZATION ----------
def normalize_query(query: str):
    query = query.lower().strip()

    # remove commas and extra spaces
    query = query.replace(",", " ")
    query = " ".join(query.split())

    return query


def normalize_token(text: str):
    token = str(text).lower().strip()
    token = " ".join(token.split())

    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 3 and token.endswith("es") and not token.endswith(("ses", "xes", "zes", "ches", "shes")):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]

    return token


def _tokenize_query(query: str):
    tokens = []
    for part in query.split():
        token = normalize_token(part)
        if token:
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def _ingredient_set(recipe: dict):
    values = []
    for ingredient in recipe.get("ingredients", []):
        normalized = normalize_token(ingredient)
        if normalized:
            values.append(normalized)
    return set(values)


def _title_tokens(recipe: dict):
    title = normalize_query(recipe.get("title", ""))
    values = []
    for part in title.split():
        token = normalize_token(part)
        if token:
            values.append(token)
    return set(values)


# ---------- SEARCH ----------
def search(query, top_k=10):
    query = normalize_query(query)
    query_tokens = _tokenize_query(query)
    query_set = set(query_tokens)

    query_vec = model.encode([query]).astype("float32")

    candidate_k = max(top_k * 20, 200)
    distances, indices = index.search(query_vec, candidate_k)

    seen_titles = set()
    scored_results = []

    for rank, i in enumerate(indices[0]):
        if i >= len(recipes):
            continue

        recipe = recipes[i]
        title = recipe["title"]

        if title not in seen_titles:
            ingredient_set = _ingredient_set(recipe)
            title_set = _title_tokens(recipe)

            ingredient_overlap = len(query_set & ingredient_set)
            title_overlap = len(query_set & title_set)

            lexical_score = (ingredient_overlap * 1.0) + (title_overlap * 1.25)
            semantic_rank_score = 1 / (1 + rank)
            final_score = (semantic_rank_score * 0.35) + (lexical_score * 0.65)

            scored_results.append((final_score, rank, recipe))
            seen_titles.add(title)

    scored_results.sort(key=lambda item: (-item[0], item[1]))
    return [recipe for _, _, recipe in scored_results[:top_k]]