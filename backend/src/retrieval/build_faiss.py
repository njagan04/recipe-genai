import json
import os
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.config import DATA_PATH, FAISS_INDEX_PATH, FAISS_META_PATH


# Model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
MAX_RECIPES = int(os.getenv("MAX_RECIPES", "120000"))
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def normalize_text(text):
    return str(text).lower().strip()


def create_text(recipe):
    ingredients = " ".join(recipe["ingredients"])
    title = recipe["title"]
    return normalize_text(f"{title} | {ingredients}")


def main():
    print("Loading data...")
    with open(DATA_PATH, "r") as f:
        recipes = json.load(f)

    if MAX_RECIPES > 0:
        recipes = recipes[:MAX_RECIPES]
        print(f"Using first {len(recipes)} recipes (MAX_RECIPES={MAX_RECIPES})")

    texts = [create_text(r) for r in recipes]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]

    print("Building IVF index...")

    nlist = 256  # more clusters for stronger recall on larger corpus

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    print("Training index...")
    index.train(embeddings)

    print("Adding vectors...")
    index.add(embeddings)

    # Set search parameter
    index.nprobe = 32  # controls recall vs speed

    print("Saving index...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("Saving metadata...")
    with open(FAISS_META_PATH, "w") as f:
        json.dump(recipes, f)

    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")

    print("Done.")


if __name__ == "__main__":
    main()