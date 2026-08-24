import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from CareerClassifier import _download_postings

POSTINGS_PATH = "data/postings.csv"

# Keep the normal dashboard/explorer table small. Large text columns are loaded
# only by the feature that actually needs them.
BASE_COLUMNS = [
    "title", "company_name", "location",
    "formatted_work_type", "formatted_experience_level", "remote_allowed",
    "normalized_salary", "max_salary", "med_salary", "min_salary",
    "views", "applies", "job_posting_url",
]


@st.cache_resource(show_spinner=False)
def _read_header():
    _download_postings()
    return list(pd.read_csv(POSTINGS_PATH, nrows=0).columns)


def _read_columns(columns):
    available = [c for c in columns if c in _read_header()]
    return pd.read_csv(POSTINGS_PATH, usecols=available, low_memory=False)


@st.cache_resource(show_spinner=False)
def load_postings():
    """Load only compact job fields used by dashboard/explorer/recommendations."""
    postings = _read_columns(BASE_COLUMNS)

    defaults = {
        "title": "", "company_name": "Unknown", "location": "Unknown",
        "formatted_work_type": "Unknown",
        "formatted_experience_level": "Unknown",
        "job_posting_url": "",
    }
    for col, default in defaults.items():
        if col not in postings.columns:
            postings[col] = default

    for col in ["max_salary", "med_salary", "min_salary", "normalized_salary", "views", "applies"]:
        if col not in postings.columns:
            postings[col] = np.nan
        postings[col] = pd.to_numeric(postings[col], errors="coerce")

    for col in ["title", "company_name", "location", "formatted_work_type", "formatted_experience_level", "job_posting_url"]:
        postings[col] = postings[col].fillna("").astype(str)

    if "remote_allowed" in postings.columns:
        postings["remote_label"] = postings["remote_allowed"].map({1: "Remote", 0: "Not Remote"}).fillna("Unknown")
    else:
        postings["remote_label"] = "Unknown"

    return postings


@st.cache_resource(show_spinner=False)
def load_skill_postings():
    """Load only title + skills text for market skill-frequency analysis."""
    postings = _read_columns(["title", "skills_desc"])
    for col in ["title", "skills_desc"]:
        if col not in postings.columns:
            postings[col] = ""
        postings[col] = postings[col].fillna("").astype(str)
    return postings


@st.cache_resource(show_spinner=False)
def load_job_engine():
    """Load title/description text and build TF-IDF only for Career Matcher."""
    postings = load_postings()
    text = _read_columns(["title", "description"])
    for col in ["title", "description"]:
        if col not in text.columns:
            text[col] = ""
        text[col] = text[col].fillna("").astype(str)

    model_text = text["title"] + " " + text["description"]
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
