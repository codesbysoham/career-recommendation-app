import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from CareerClassifier import _download_postings

POSTINGS_PATH = "data/postings.csv"

# Only columns actually used by the dashboard, matcher, gap analyzer and
# explorer. Reading the entire LinkedIn export wastes a lot of RAM on deploy.
JOB_COLUMNS = [
    "title", "company_name", "location", "description", "skills_desc",
    "formatted_work_type", "formatted_experience_level", "remote_allowed",
    "normalized_salary", "max_salary", "med_salary", "min_salary",
    "views", "applies", "job_posting_url",
]


@st.cache_data(show_spinner=False)
def load_postings():
    """Load only the fields used by the application; do not build TF-IDF."""
    _download_postings()

    # Some dataset versions may not contain every optional field, so inspect
    # the header first and request only the intersection.
    header = pd.read_csv(POSTINGS_PATH, nrows=0)
    available = [c for c in JOB_COLUMNS if c in header.columns]
    postings = pd.read_csv(
        POSTINGS_PATH,
        usecols=available,
        low_memory=False,
    )

    defaults = {
        "title": "",
        "company_name": "Unknown",
        "location": "Unknown",
        "description": "",
        "skills_desc": "",
        "formatted_work_type": "Unknown",
        "formatted_experience_level": "Unknown",
        "job_posting_url": "",
    }
    for col, default in defaults.items():
        if col not in postings.columns:
            postings[col] = default

    for col in [
        "max_salary", "med_salary", "min_salary",
        "normalized_salary", "views", "applies"
    ]:
        if col not in postings.columns:
            postings[col] = np.nan
        postings[col] = pd.to_numeric(postings[col], errors="coerce")

    for col in [
        "title", "company_name", "location", "description", "skills_desc",
        "formatted_work_type", "formatted_experience_level", "job_posting_url"
    ]:
        postings[col] = postings[col].fillna("").astype(str)

    if "remote_allowed" in postings.columns:
        postings["remote_label"] = postings["remote_allowed"].map(
            {1: "Remote", 0: "Not Remote"}
        ).fillna("Unknown")
    else:
        postings["remote_label"] = "Unknown"

    return postings


@st.cache_resource(show_spinner=False)
def load_job_engine():
    """Build TF-IDF only when Career Matcher actually needs it."""
    postings = load_postings()
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
