# Career Recommendation & Job Market Analytics backend
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

SKILLS_PATH = "data/Skills.xlsx"
KNOWLEDGE_PATH = "data/Knowledge.xlsx"
OCCUPATIONS_PATH = "data/Occupation Data.xlsx"


def _download_postings():
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
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("data")
        os.remove(zip_path)


@st.cache_data(show_spinner="Loading O*NET datasets...")
def load_onet_data():
    skills = pd.read_excel(SKILLS_PATH)
    knowledge = pd.read_excel(KNOWLEDGE_PATH)
    occupations = pd.read_excel(OCCUPATIONS_PATH)

    # Keep only the columns needed by the dashboard.
    skills = skills[
        [
            "O*NET-SOC Code",
            "Title",
            "Element Name",
            "Scale ID",
            "Data Value",
        ]
    ].copy()

    knowledge = knowledge[
        [
            "O*NET-SOC Code",
            "Title",
            "Element Name",
            "Scale ID",
            "Data Value",
        ]
    ].copy()

    return skills, knowledge, occupations


@st.cache_resource(show_spinner="Loading job market data and building model...")
def load_pipeline():
    """Load postings, O*NET data and build a lightweight TF-IDF model."""
    _download_postings()

    postings = pd.read_csv(POSTINGS_PATH, low_memory=False)

    # Standardize a few fields used throughout the app.
    for col in ["title", "company_name", "location", "description"]:
        if col not in postings.columns:
            postings[col] = ""

    for col in ["max_salary", "med_salary", "min_salary",
                "normalized_salary", "views", "applies"]:
        if col not in postings.columns:
            postings[col] = np.nan

    postings["title"] = postings["title"].fillna("Unknown").astype(str)
    postings["company_name"] = postings["company_name"].fillna("Unknown").astype(str)
    postings["location"] = postings["location"].fillna("Unknown").astype(str)
    postings["description"] = postings["description"].fillna("").astype(str)

    # Numeric fields.
    for col in ["max_salary", "med_salary", "min_salary",
                "normalized_salary", "views", "applies"]:
        postings[col] = pd.to_numeric(postings[col], errors="coerce")

    if "remote_allowed" in postings.columns:
        postings["remote_label"] = postings["remote_allowed"].map(
            {1: "Remote", 0: "Not Remote"}
        ).fillna("Unknown")
    else:
        postings["remote_label"] = "Unknown"

    if "listed_time" in postings.columns:
        postings["listed_date"] = pd.to_datetime(
            postings["listed_time"], errors="coerce", unit="ms"
        )
        # If the source is already datetime-like, retry without a unit.
        if postings["listed_date"].notna().sum() == 0:
            postings["listed_date"] = pd.to_datetime(
                postings["listed_time"], errors="coerce"
            )
    else:
        postings["listed_date"] = pd.NaT

    # Lightweight TF-IDF for career/job matching.
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 1),
        stop_words="english",
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
        dtype=np.float32,
    )

    tfidf_matrix = tfidf.fit_transform(postings["description"])

    skills, knowledge, occupations = load_onet_data()

    return postings, tfidf, tfidf_matrix, skills, knowledge, occupations


def get_skill_options(skills_df):
    """Return the real O*NET skill names from Skills.xlsx."""
    return sorted(
        skills_df["Element Name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


@st.cache_data
def get_skill_profile(skills_df):
    """Build occupation × skill importance/level tables from O*NET Skills."""
    skills = skills_df.copy()

    importance = (
        skills[skills["Scale ID"] == "IM"]
        .pivot_table(
            index=["O*NET-SOC Code", "Title"],
            columns="Element Name",
            values="Data Value",
            aggfunc="mean",
        )
        .fillna(0)
    )

    level = (
        skills[skills["Scale ID"] == "LV"]
        .pivot_table(
            index=["O*NET-SOC Code", "Title"],
            columns="Element Name",
            values="Data Value",
            aggfunc="mean",
        )
        .fillna(0)
    )

    return importance, level


def career_match(selected_skills, skills_df, top_n=10):
    """Rank occupations using the O*NET importance values of selected skills."""
    importance, _ = get_skill_profile(skills_df)

    selected = [s for s in selected_skills if s in importance.columns]
    if not selected:
        return pd.DataFrame()

    # Average selected-skill importance, normalized to a 0–100 score.
    raw = importance[selected].mean(axis=1)
    score = (raw / 5.0 * 100).clip(0, 100)

    result = pd.DataFrame({
        "O*NET-SOC Code": importance.index.get_level_values(0),
        "Career": importance.index.get_level_values(1),
        "Match Score": score.values,
        "Skills Matched": (importance[selected] > 0).sum(axis=1).values,
    })

    return (
        result.sort_values(
            ["Match Score", "Skills Matched"],
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def get_skill_gaps(occupation_title, selected_skills, skills_df, top_n=10):
    """Show the most important skills for an occupation and selected-skill gaps."""
    importance, _ = get_skill_profile(skills_df)

    rows = importance.reset_index()
    rows = rows[rows["Title"] == occupation_title]

    if rows.empty:
        return pd.DataFrame()

    row = rows.iloc[0]
    skill_columns = [c for c in importance.columns]
    values = pd.DataFrame({
        "Skill": skill_columns,
        "Importance": [row.get(c, 0) for c in skill_columns],
    })

    values["Selected"] = values["Skill"].isin(selected_skills)
    values = values.sort_values("Importance", ascending=False).head(top_n)

    return values.reset_index(drop=True)


def recommend_jobs(
    user_skills,
    tfidf,
    tfidf_matrix,
    postings_df,
    top_n=10,
):
    """Recommend real postings using TF-IDF similarity."""
    user_text = " ".join(user_skills)
    user_vector = tfidf.transform([user_text])
    similarities = cosine_similarity(user_vector, tfidf_matrix)[0]

    result = postings_df[
        [
            c for c in [
                "title",
                "company_name",
                "location",
                "formatted_work_type",
                "formatted_experience_level",
                "remote_label",
                "normalized_salary",
                "job_posting_url",
            ]
            if c in postings_df.columns
        ]
    ].copy()

    result["Match Score"] = similarities * 100

    return (
        result.sort_values("Match Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def filter_jobs(
    postings,
    title=None,
    location=None,
    work_type=None,
    experience=None,
    remote=None,
    min_salary=None,
):
    """Filter the job-posting table for the Job Explorer page."""
    df = postings

    if title and title != "All":
        mask = df["title"].str.contains(title, case=False, na=False)
        df = df[mask]

    if location and location != "All":
        mask = df["location"].str.contains(location, case=False, na=False)
        df = df[mask]

    if work_type and work_type != "All" and "formatted_work_type" in df:
        df = df[df["formatted_work_type"].fillna("Unknown") == work_type]

    if experience and experience != "All" and "formatted_experience_level" in df:
        df = df[
            df["formatted_experience_level"].fillna("Unknown") == experience
        ]

    if remote and remote != "All":
        df = df[df["remote_label"] == remote]

    if min_salary is not None:
        df = df[df["normalized_salary"].fillna(0) >= min_salary]

    return df


def top_counts(series, n=10):
    return series.fillna("Unknown").astype(str).value_counts().head(n)


def dashboard_metrics(postings):
    """Calculate high-level market KPIs."""
    total = len(postings)
    companies = postings["company_name"].replace("Unknown", np.nan).nunique()
    titles = postings["title"].replace("Unknown", np.nan).nunique()

    remote = (
        (postings["remote_label"] == "Remote").sum()
        if "remote_label" in postings
        else 0
    )

    salary = postings["normalized_salary"].dropna()
    median_salary = salary.median() if not salary.empty else np.nan

    return total, companies, titles, remote, median_salary
