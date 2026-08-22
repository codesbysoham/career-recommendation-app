# Career recommendation backend
import os
import zipfile
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

DATA_PATHS = {
    "skills": "data/Skills.xlsx",
    "knowledge": "data/Knowledge.xlsx",
    "occupations": "data/Occupation Data.xlsx",
}

POSTINGS_PATH = "data/postings.csv"
KAGGLE_DATASET = "soham1510/career-postings-dataset"


def download_postings():
    """Download postings.csv from Kaggle only when it is not already present."""
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
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("data")
        os.remove(zip_path)


@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    """Load the Excel files from GitHub and postings.csv from Kaggle."""
    download_postings()

    skills_df = pd.read_excel(DATA_PATHS["skills"])
    knowledge_df = pd.read_excel(DATA_PATHS["knowledge"])
    occupations_df = pd.read_excel(DATA_PATHS["occupations"])
    postings_df = pd.read_csv(POSTINGS_PATH, low_memory=False)

    return skills_df, knowledge_df, occupations_df, postings_df


@st.cache_resource(show_spinner="Building recommendation model...")
def build_model(postings_df):
    """Build the TF-IDF model once and cache it."""
    postings_df = postings_df.copy()

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

    if "max_salary" not in postings_df.columns:
        postings_df["max_salary"] = np.nan

    postings_df["job_title"] = postings_df["job_title"].fillna("Unknown")
    postings_df["description"] = postings_df["description"].fillna("")

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
    )

    tfidf_matrix = tfidf.fit_transform(postings_df["description"])

    return postings_df, tfidf, tfidf_matrix


def load_pipeline():
    """Load all data and build the cached recommendation model."""
    skills_df, knowledge_df, occupations_df, raw_postings = load_data()

    postings_df, tfidf, tfidf_matrix = build_model(raw_postings)

    return {
        "skills_df": skills_df,
        "knowledge_df": knowledge_df,
        "occupations_df": occupations_df,
        "postings_df": postings_df,
        "tfidf": tfidf,
        "tfidf_matrix": tfidf_matrix,
    }


def recommend_jobs(user_skills, tfidf, tfidf_matrix, postings_df, top_n=5):
    """Return the top job postings matching the selected skills."""
    user_text = " ".join(user_skills)

    user_vec = tfidf.transform([user_text])
    similarities = cosine_similarity(user_vec, tfidf_matrix)[0]

    result = postings_df.copy()
    result["similarity"] = similarities

    return (
        result.sort_values("similarity", ascending=False)
        .head(top_n)[["job_title", "similarity"]]
        .reset_index(drop=True)
    )
