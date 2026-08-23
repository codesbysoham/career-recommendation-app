import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from CareerClassifier import _download_postings

POSTINGS_PATH = "data/postings.csv"


def _prepare_postings():
    _download_postings()
    postings = pd.read_csv(POSTINGS_PATH, low_memory=False)

    defaults = {
        "title": "",
        "company_name": "Unknown",
        "location": "Unknown",
        "description": "",
        "skills_desc": "",
    }
    for col, default in defaults.items():
        if col not in postings.columns:
            postings[col] = default

    for col in ["max_salary", "med_salary", "min_salary", "normalized_salary", "views", "applies"]:
        if col not in postings.columns:
            postings[col] = np.nan
        postings[col] = pd.to_numeric(postings[col], errors="coerce")

    for col in ["title", "company_name", "location", "description", "skills_desc"]:
        postings[col] = postings[col].fillna("").astype(str)

    if "remote_allowed" in postings.columns:
        postings["remote_label"] = postings["remote_allowed"].map({1: "Remote", 0: "Not Remote"}).fillna("Unknown")
    else:
        postings["remote_label"] = "Unknown"

    return postings


@st.cache_data(show_spinner="Loading job-market data...")
def load_postings():
    """Load the 123k job postings without building the TF-IDF model."""
    return _prepare_postings()


@st.cache_resource(show_spinner="Loading job market and ML model...")
def load_job_engine():
    """Load postings plus TF-IDF only for recommendation features."""
    postings = _prepare_postings()
    model_text = postings["title"] + " " + postings["description"]
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf_matrix = tfidf.fit_transform(model_text)
    return postings, tfidf, tfidf_matrix
