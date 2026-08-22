# Career Recommendation backend for Streamlit
import os
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


POSTINGS_PATH = "data/postings.csv"
KAGGLE_DATASET = "soham1510/career-postings-dataset"


def _download_postings():
    """Download postings.csv from Kaggle if it is not already present."""
    if os.path.exists(POSTINGS_PATH):
        return

    os.makedirs("data", exist_ok=True)

    os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle_username"]
    os.environ["KAGGLE_KEY"] = st.secrets["kaggle_key"]

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_file(
        KAGGLE_DATASET,
        "postings.csv",
        path="data",
        force=True,
    )

    zip_path = "data/postings.csv.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("data")
        os.remove(zip_path)


@st.cache_resource(show_spinner="Loading career recommendation engine...")
def load_pipeline():
    """
    Load postings and build the TF-IDF recommendation model once.

    Everything is kept inside one cached resource so Streamlit does not
    repeatedly create large DataFrame/TF-IDF objects on reruns.
    """
    _download_postings()

    postings_df = pd.read_csv(POSTINGS_PATH, low_memory=False)

    # Detect the relevant columns without depending on exact capitalization.
    title_col = next(
        (c for c in postings_df.columns if "title" in c.lower()),
        None,
    )
    desc_col = next(
        (c for c in postings_df.columns if "descrip" in c.lower()),
        None,
    )
    sal_col = next(
        (
            c for c in postings_df.columns
            if "max_sal" in c.lower() or "salary" in c.lower()
        ),
        None,
    )
    loc_col = next(
        (c for c in postings_df.columns if "location" in c.lower()),
        None,
    )

    rename = {}
    if title_col:
        rename[title_col] = "job_title"
    if desc_col:
        rename[desc_col] = "description"
    if sal_col:
        rename[sal_col] = "max_salary"
    if loc_col:
        rename[loc_col] = "location"

    postings_df.rename(columns=rename, inplace=True)

    if "job_title" not in postings_df.columns:
        postings_df["job_title"] = "Unknown"

    if "description" not in postings_df.columns:
        postings_df["description"] = ""

    postings_df["job_title"] = postings_df["job_title"].fillna("Unknown")
    postings_df["description"] = postings_df["description"].fillna("")

    # Lightweight TF-IDF suitable for Streamlit Cloud.
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 1),
        stop_words="english",
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
        dtype=np.float32,
    )

    tfidf_matrix = tfidf.fit_transform(postings_df["description"])

    return postings_df, tfidf, tfidf_matrix


def recommend_jobs(
    user_skills,
    tfidf,
    tfidf_matrix,
    postings_df,
    top_n=5,
):
    """Return the top job postings matching the selected skills."""
    user_text = " ".join(user_skills)

    user_vector = tfidf.transform([user_text])
    similarities = cosine_similarity(user_vector, tfidf_matrix)[0]

    # Avoid modifying the cached DataFrame.
    result = postings_df[["job_title"]].copy()
    result["similarity"] = similarities

    return (
        result
        .sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
