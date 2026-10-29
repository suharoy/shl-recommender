# SHL Assessment Recommender

### Hybrid Retrieval and Ranking for Job-to-Assessment Recommendation

A local recommendation system that maps **natural-language job descriptions and hiring requirements** to relevant SHL assessments using a combination of semantic search, lexical retrieval, intent-aware ranking, and practical constraints.

The system combines **FAISS-based dense retrieval** with **BM25 lexical matching**, then applies lightweight decision logic to improve the relevance and balance of the final recommendation set.

---

## Overview

Selecting appropriate pre-employment assessments from a large catalogue is fundamentally a **retrieval and ranking problem**.

A job description may contain:

* technical skills;
* behavioural requirements;
* role-specific terminology;
* communication expectations;
* time constraints;
* mixed competency requirements.

Simple keyword search can miss semantic relationships, while pure embedding search can overlook exact skill terminology.

This project therefore uses a **hybrid recommendation architecture**.

```text
Job description / hiring query
             │
             ▼
       Query processing
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
Dense retrieval    Sparse retrieval
 MiniLM + FAISS         BM25
     │                │
     └───────┬────────┘
             ▼
      Weighted ranking
             │
             ▼
      Intent detection
             │
   ┌─────────┼─────────┐
   │         │         │
Technical   Mixed   Behavioural
   │         │         │
   └─────────┼─────────┘
             ▼
     Duration filtering
             │
             ▼
 Balanced recommendation
             │
             ▼
      Top assessments
```

---

## Core Idea

The recommender combines two complementary retrieval signals.

### Dense semantic retrieval

Assessment descriptions are represented using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting embeddings are indexed using **FAISS**.

This allows the system to retrieve assessments that are semantically related to a job description even when the exact wording differs.

### Sparse lexical retrieval

A **BM25** index captures direct term overlap.

This is useful for requirements containing specific terminology such as:

```text
Python
Java
SQL
Excel
communication
leadership
engineering
```

### Hybrid ranking

The current implementation combines the two signals using weighted retrieval:

```text
Dense semantic score
        +
BM25 lexical score
        ↓
Hybrid ranking
```

The default implementation gives both retrieval mechanisms substantial influence rather than relying on either one in isolation.

---

## Intent-Aware Recommendation

The system applies lightweight intent analysis after retrieval.

Queries are classified approximately into:

```text
technical
behavioural
mixed
unknown
```

This helps prevent a mixed job description from returning only one category of assessment.

For example, a role requiring both:

```text
Python / SQL / engineering skills
```

and:

```text
stakeholder communication / teamwork
```

can receive a more balanced recommendation set.

The implementation distinguishes assessment categories during final ranking and attempts to preserve appropriate coverage when the query contains mixed requirements.

---

## Duration-Aware Filtering

The recommender also recognizes simple duration constraints expressed in natural language.

For example:

```text
Recommend assessments for a software engineer
that can be completed in about 40 minutes.
```

The system extracts the requested duration and prefers assessments within an approximate time range.

If the filter becomes too restrictive, retrieval falls back to the broader candidate pool instead of returning no recommendations.

---

## Query Expansion

Some short job queries contain limited information.

The implementation therefore includes lightweight expansion rules for selected terms.

For example:

```text
writer
```

can introduce related signals such as:

```text
English
verbal
grammar
communication
```

This helps improve retrieval when the original query is underspecified.

---

## System Architecture

### Retrieval layer

```text
backend/recommender.py
```

Responsible for:

* loading the assessment catalogue;
* loading the FAISS index;
* loading the BM25 index;
* encoding user queries;
* hybrid retrieval;
* intent detection;
* duration extraction;
* recommendation balancing;
* duplicate removal.

### API layer

```text
backend/app.py
```

Exposes the recommender through FastAPI.

Available endpoints include:

```text
GET  /health
POST /recommend
```

### Interface layer

```text
frontend/app.py
```

Provides a lightweight Streamlit interface for submitting job descriptions and inspecting returned assessment recommendations.

---

## Repository Structure

```text
.
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── models.py
│   └── recommender.py
│
├── frontend/
│   └── app.py
│
├── notebooks/
│   ├── 01_build_catalog.ipynb
│   ├── 02_build_index.ipynb
│   └── 03_eval_recall.ipynb
│
├── scripts/
│   └── make_submission_csv.py
│
├── data/
│   ├── catalog.csv
│   ├── catalog_clean.csv
│   ├── meta_catalog.json
│   ├── embeddings.faiss
│   ├── bm25.pkl
│   ├── recall_tuning.png
│   ├── sample_results.csv
│   └── submission.csv
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Development Workflow

The repository captures the recommendation pipeline as three main experimental stages.

### 1. Catalogue construction

```text
notebooks/01_build_catalog.ipynb
```

Prepares the assessment catalogue and associated metadata.

### 2. Retrieval index construction

```text
notebooks/02_build_index.ipynb
```

Builds the artifacts required by the recommendation system, including semantic representations and retrieval indexes.

### 3. Evaluation

```text
notebooks/03_eval_recall.ipynb
```

Contains retrieval evaluation and tuning experiments.

The repository also contains:

```text
data/recall_tuning.png
```

for visual inspection of the tuning process.

---

## Technology Stack

| Component           | Technology                |
| ------------------- | ------------------------- |
| Language            | Python                    |
| Semantic embeddings | Sentence Transformers     |
| Embedding model     | MiniLM-L6-v2              |
| Dense retrieval     | FAISS                     |
| Sparse retrieval    | BM25                      |
| Ranking             | Weighted hybrid retrieval |
| API                 | FastAPI                   |
| Frontend            | Streamlit                 |
| Data manipulation   | Pandas / NumPy            |
| NLP utilities       | NLTK                      |
| Evaluation          | Scikit-learn / notebooks  |

---

## Local Setup

Clone the repository:

```powershell
git clone https://github.com/suharoy/shl-recommender.git
cd shl-recommender
```

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run the API

Start the FastAPI backend:

```powershell
uvicorn backend.app:app --reload
```

The API runs locally at:

```text
http://localhost:8000
```

Health check:

```text
GET /health
```

Recommendation endpoint:

```text
POST /recommend
```

Example request:

```json
{
  "query": "Software engineer with Python, SQL and strong stakeholder communication skills"
}
```

The API returns a ranked list containing assessment names and their catalogue URLs.

---

## Run the Streamlit Interface

Keep the FastAPI service running and open a second terminal.

Run:

```powershell
streamlit run frontend/app.py
```

The Streamlit application sends recommendation requests to:

```text
http://localhost:8000/recommend
```

---

## Submission Generation

The repository contains:

```text
scripts/make_submission_csv.py
```

which runs test queries against the local recommendation API and writes results to a submission CSV.

With the API running:

```powershell
python scripts/make_submission_csv.py
```

---

## Demo

A local deployment demonstration covering the backend, frontend, and recommendation workflow is available here:

[**SHL Assessment Recommender — Demo Video**](https://drive.google.com/drive/folders/1UqNxaKICr2Jn9xlvTv1-t5YSleyqSMDI)

---

## Why Hybrid Retrieval?

Neither lexical nor semantic retrieval is sufficient for every hiring query.

BM25 performs well when exact terminology matters:

```text
Java
Python
SQL
Excel
```

Dense embeddings are more useful when the relationship is conceptual rather than lexical:

```text
ability to work across teams
```

versus:

```text
stakeholder collaboration
```

The hybrid system therefore combines:

```text
exact terminology
      +
semantic similarity
      +
intent-aware balancing
      +
practical constraints
```

to create the final recommendation set.

---

## Limitations

This repository represents a recommendation-system prototype rather than a production hiring platform.

Important limitations include:

* intent detection is rule-based;
* query expansion uses manually defined mappings;
* hybrid ranking weights are manually configured;
* duration extraction supports relatively simple natural-language patterns;
* recommendation quality depends on the completeness and accuracy of the assessment catalogue;
* the system does not learn from recruiter feedback or historical hiring outcomes;
* semantic retrieval uses a general-purpose sentence embedding model rather than a recruitment-specific encoder;
* evaluation is limited to the available development and test data.

The recommendations should therefore be interpreted as **ranking suggestions**, not automated hiring decisions.

---

## Future Work

Potential extensions include:

* learned ranking models;
* cross-encoder reranking;
* recruitment-domain embedding models;
* richer competency extraction;
* structured skill ontologies;
* improved duration and constraint parsing;
* feedback-based ranking;
* explainable recommendation scores;
* offline retrieval benchmarks;
* model-based query expansion;
* multi-objective ranking across relevance, duration and assessment diversity.

---

## Responsible Use

This system recommends **assessment products**, not candidates.

It should not be used to infer protected characteristics, automatically determine candidate suitability, or make employment decisions.

Human review should remain part of any real assessment-selection workflow.

---

## License

The original source code and documentation in this repository are released under the **MIT License**.

See [`LICENSE`](LICENSE).

Assessment names, catalogue information, source datasets, trademarks, and third-party content remain subject to the rights and terms of their respective owners.

This repository is an independent technical project and is not presented as an official SHL product.

---

## Author

**Suha Roy**

M.Sc. Computer Science — Big Data Analytics
Central University of Rajasthan

**GitHub:** https://github.com/suharoy
**LinkedIn:** https://linkedin.com/in/suharoy
