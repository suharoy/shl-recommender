# backend/recommender.py
import random

def recommend_assessments(query: str):
    """
    Placeholder recommender function.
    Later: replace with hybrid FAISS + BM25 + rerank logic.
    """
    mock_catalog = [
        {"name": "Core Java - Advanced Level", "url": "https://www.shl.com/.../core-java-advanced/"},
        {"name": "Interpersonal Skills Assessment", "url": "https://www.shl.com/.../interpersonal-communications/"},
        {"name": "Python Programming Test", "url": "https://www.shl.com/.../python-programming/"},
        {"name": "Team Collaboration Skills", "url": "https://www.shl.com/.../collaboration/"},
        {"name": "Cognitive Ability Assessment", "url": "https://www.shl.com/.../cognitive-assessment/"}
    ]
    return random.sample(mock_catalog, 3)
