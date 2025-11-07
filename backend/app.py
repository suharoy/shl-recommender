# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

from backend.recommender import recommend_assessments

app = FastAPI(title="SHL Assessment Recommender")

# ----------------------------
# Pydantic Schemas
# ----------------------------
class QueryRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    name: str
    url: str

class RecommendationResponse(BaseModel):
    recommendations: List[Assessment]

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: QueryRequest):
    try:
        results = recommend_assessments(req.query)
        return {"recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
