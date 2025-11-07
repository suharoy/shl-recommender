# backend/recommender.py (enhanced with intent balancing + duration filtering)
import os
import re
import faiss
import pickle
import nltk
import numpy as np
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer

# Ensure NLTK tokenizer is available
nltk.download('punkt', quiet=True)

DATA_DIR = os.getenv("SHL_DATA_DIR", "data")

# ----------------------------
# Configurable weights (fallbacks)
# ----------------------------
W_DENSE = 0.55
W_SPARSE = 0.45
try:
    from backend import config as _cfg
    W_DENSE = getattr(_cfg, "W_DENSE", W_DENSE)
    W_SPARSE = getattr(_cfg, "W_SPARSE", W_SPARSE)
except Exception:
    pass

# ----------------------------
# Load artifacts once
# ----------------------------
META_PATH = os.path.join(DATA_DIR, "meta_catalog.json")
DF = pd.read_json(META_PATH, lines=True)
for col, default in [("test_type", "Unknown"), ("duration_minutes", np.nan), ("description", "")]:
    if col not in DF.columns:
        DF[col] = default

INDEX = faiss.read_index(os.path.join(DATA_DIR, "embeddings.faiss"))
with open(os.path.join(DATA_DIR, "bm25.pkl"), "rb") as f:
    BM25 = pickle.load(f)

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Intent & duration utilities
# ----------------------------
TECH_WORDS = [
    "python","java","sql","excel","developer","engineering","programming","code",
    ".net","c#","react","node","apis","cloud","kotlin","swift","powerbi","tableau","spark","ml","dl"
]
BEHAV_WORDS = [
    "team","communication","stakeholder","collaboration","leadership","people","interpersonal",
    "behavior","personality","conflict","cooperate","empathy"
]
EXPANSIONS = {
    "writer": ["english", "verbal", "grammar", "communication"],
    "content": ["english", "writing", "verbal", "communication"],
    "seo": ["english", "communication", "analysis"],
}

def detect_intent(query: str) -> str:
    q = query.lower()
    t_hits = sum(w in q for w in TECH_WORDS)
    b_hits = sum(w in q for w in BEHAV_WORDS)
    if t_hits and b_hits:
        return "mixed"
    if t_hits:
        return "technical"
    if b_hits:
        return "behavioral"
    return "unknown"

def extract_duration_minutes(query: str):
    q = query.lower()
    m = re.search(r"(\d+)\s*(minute|min|hour|hr|hrs)", q)
    if not m:
        return None
    val = int(m.group(1))
    unit = m.group(2)
    return val * 60 if unit.startswith("hour") or unit.startswith("hr") else val

# ----------------------------
# Core retrieval (hybrid)
# ----------------------------

def _dense_scores(query: str) -> np.ndarray:
    q_emb = MODEL.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, _ = INDEX.search(q_emb, len(DF))
    return scores.flatten()

def _sparse_scores(query: str) -> np.ndarray:
    tokens = nltk.word_tokenize(query.lower())
    return np.array(BM25.get_scores(tokens))

def hybrid_search(query: str, top_k: int = 10, w_dense: float = W_DENSE, w_sparse: float = W_SPARSE) -> pd.DataFrame:
    dense = _dense_scores(query)
    sparse = _sparse_scores(query)
    final = w_dense * dense + w_sparse * sparse
    top_ids = np.argsort(final)[::-1][:top_k]
    out = DF.iloc[top_ids].copy()
    out["_score"] = final[top_ids]
    return out

# ----------------------------
# Balanced recommendation wrapper
# ----------------------------

def _expand_query_if_thin(query: str) -> str:
    q = query
    ql = query.lower()
    for key, vals in EXPANSIONS.items():
        if key in ql:
            q += " " + " ".join(vals)
    return q

def balanced_recommend(query: str, top_k: int = 10) -> List[dict]:
    intent = detect_intent(query)
    target_dur = extract_duration_minutes(query)

    # Start with broader pool
    pool_q = _expand_query_if_thin(query)
    pool = hybrid_search(pool_q, top_k=max(50, top_k))

    # Duration filter if specified
    if target_dur is not None and "duration_minutes" in pool.columns:
        low, high = target_dur - 10, target_dur + 10
        pool = pool[(pool["duration_minutes"].fillna(target_dur) >= low) & (pool["duration_minutes"].fillna(target_dur) <= high)]
        if pool.empty:  # fall back if too strict
            pool = hybrid_search(pool_q, top_k=max(50, top_k))

    if "test_type" not in pool.columns:
        pool["test_type"] = "Unknown"

    def pick(df, n):
        return df.head(n)

    if intent == "mixed":
        k_part = pick(pool[pool["test_type"].str.upper() == "K"], 6)
        p_part = pick(pool[pool["test_type"].str.upper() == "P"], 4)
        final = pd.concat([k_part, p_part])
        if len(final) < top_k:
            final = pd.concat([final, pool.drop(final.index).head(top_k - len(final))])
    elif intent == "technical":
        final = pick(pool[pool["test_type"].str.upper() == "K"], top_k)
        if len(final) < top_k:
            final = pd.concat([final, pool.drop(final.index).head(top_k - len(final))])
    elif intent == "behavioral":
        final = pick(pool[pool["test_type"].str.upper() == "P"], top_k)
        if len(final) < top_k:
            final = pd.concat([final, pool.drop(final.index).head(top_k - len(final))])
    else:
        final = pool.head(top_k)

    final = final.drop_duplicates(subset=["url"]).head(top_k)
    return final[["name", "url"]].to_dict(orient="records")

# ----------------------------
# FastAPI entrypoint wrapper
# ----------------------------

def recommend_assessments(query: str) -> List[dict]:
    return balanced_recommend(query, top_k=10)
