import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.config import DATA_PATH, FAISS_INDEX_PATH, FAISS_META_PATH


# Model
model = SentenceTransformer("all-MiniLM-L6-v2")


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

    # OPTIONAL: reduce size for now
    recipes = recipes[:50000]

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

    nlist = 100  # clusters (tune later)

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    print("Training index...")
    index.train(embeddings)

    print("Adding vectors...")
    index.add(embeddings)

    # Set search parameter
    index.nprobe = 10  # controls recall vs speed

    print("Saving index...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("Saving metadata...")
    with open(FAISS_META_PATH, "w") as f:
        json.dump(recipes, f)

    print("Done.")


if __name__ == "__main__":
    main()