import numpy as np
import nltk
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download("punkt", quiet=True)

# -------------------------------------------------------------
# Load data and models
# -------------------------------------------------------------
CATALOG_PATH = "data/catalog_clean.csv"
BM25_PATH = "data/bm25.pkl"
EMBEDDINGS_PATH = "data/embeddings.npy"

catalog = pd.read_csv(CATALOG_PATH)

# Load precomputed embeddings for catalog
embeddings = np.load(EMBEDDINGS_PATH)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load BM25
with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)


# -------------------------------------------------------------
# Utility: Hybrid search (semantic + lexical)
# -------------------------------------------------------------
def hybrid_search(query, top_k=10, w_dense=0.55, w_sparse=0.45):
    """
    Performs hybrid search over catalog using semantic + lexical scores.
    Returns top_k URLs with highest combined score.
    """

    # ---------- Sparse (BM25) ----------
    q_tokens = nltk.word_tokenize(query.lower())
    sparse_scores = np.array(bm25.get_scores(q_tokens))

    # ---------- Dense (semantic) ----------
    query_emb = model.encode([query], convert_to_numpy=True)
    # Normalize embeddings for cosine similarity
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    dense_scores = cosine_similarity(query_norm, embeddings_norm)[0]

    # ---------- Combine scores ----------
    final_scores = w_dense * dense_scores + w_sparse * sparse_scores
    top_ids = np.argsort(final_scores)[::-1][:top_k]

    return catalog.iloc[top_ids][["name", "url"]].to_dict(orient="records")


# -------------------------------------------------------------
# API helper (called by FastAPI endpoint)
# -------------------------------------------------------------
def recommend(query: str, top_k: int = 10):
    """
    Returns a list of recommended assessments (dicts with name + url)
    for a given natural-language query.
    """
    if not query or len(query.strip()) == 0:
        return []

    try:
        results = hybrid_search(query, top_k=top_k)
        return results
    except Exception as e:
        print(f"[ERROR] in recommend(): {e}")
        return []
