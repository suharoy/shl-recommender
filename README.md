# SHL Assessment Recommender

An intelligent recommendation system that maps natural-language job descriptions to relevant SHL assessments.

## Structure
- `backend/` – FastAPI microservice with `/health` and `/recommend`
- `frontend/` – Streamlit client for quick testing
- `data/` – Catalog, embeddings, metadata
- `notebooks/` – Crawling, embeddings, and evaluation
- `scripts/` – Utilities for submission CSV

## Run Locally
`pip install -r requirements.txt`

`uvicorn backend.app:app --reload`

`streamlit run frontend/app.py`

### Demo Video 
[Click here to watch the full local deployment demo](https://drive.google.com/drive/folders/1UqNxaKICr2Jn9xlvTv1-t5YSleyqSMDI)

> The system runs fully locally using FAISS for vector retrieval and BM25 for lexical matching. 
> A working demo is provided showing backend (FastAPI), frontend (Streamlit), and evaluation pipeline in action.

