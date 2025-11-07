from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm

# Resolve absolute paths safely
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Gen_AI Dataset.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "submission.csv"
API_URL = "http://localhost:8000/recommend"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load test queries
test_df = pd.read_excel(DATA_PATH, sheet_name="Test-Set")
test_df.columns = [c.strip() for c in test_df.columns]

rows = []
for query in tqdm(test_df["Query"], desc="Generating predictions"):
    try:
        resp = requests.post(API_URL, json={"query": query})
        if resp.status_code == 200:
            data = resp.json().get("recommendations", [])
            for item in data:
                rows.append({"Query": query, "Assessment_url": item["url"]})
        else:
            print(f"Error {resp.status_code} for query: {query[:40]}...")
    except Exception as e:
        print(f"Request failed for query: {query[:40]}... -> {e}")

sub_df = pd.DataFrame(rows, columns=["Query", "Assessment_url"])
sub_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n Submission CSV saved to {OUTPUT_PATH} with {len(sub_df)} rows.")
