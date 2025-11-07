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
