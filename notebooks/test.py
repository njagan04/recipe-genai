import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.search import search

results = search("milk sugar butter", top_k=5)

for r in results:
    print(r["title"])