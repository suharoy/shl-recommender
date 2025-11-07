# frontend/app.py
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SHL Recommender", layout="centered")

st.title("🧠 SHL Assessment Recommender")
st.write("Enter a job description or query to get top SHL assessments.")

query = st.text_area("Job Description / Query", height=200)
if st.button("Recommend"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Fetching recommendations..."):
            resp = requests.post("http://localhost:8000/recommend", json={"query": query})
            if resp.status_code == 200:
                data = resp.json()["recommendations"]
                st.success(f"Found {len(data)} recommendations:")
                st.dataframe(pd.DataFrame(data))
            else:
                st.error(f"API error: {resp.text}")
