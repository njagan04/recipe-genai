import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data/processed/recipes_cleaned.json")

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index/recipes_ivf.index")
FAISS_META_PATH = os.path.join(BASE_DIR, "faiss_index/metadata.json")