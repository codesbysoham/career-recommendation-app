import re

import numpy as np
import pandas as pd
import streamlit as st

from CareerClassifier import _download_postings

POSTINGS_PATH = "data/postings.csv"

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
    """Load only compact fields used by dashboard/explorer."""
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
    """Load only title + structured skills for market skill-frequency analysis."""
    postings = _read_columns(["title", "skills_desc"])
    for col in ["title", "skills_desc"]:
        if col not in postings.columns:
            postings[col] = ""
        postings[col] = postings[col].fillna("").astype(str)
    return postings


@st.cache_resource(show_spinner=False)
def load_job_engine():
    """Compatibility wrapper: do NOT build a 123k-row TF-IDF matrix."""
    return load_postings(), None, None


def _skill_pattern(skill):
    # Escape user/library skill names so names such as C++, .NET and C# work.
    return re.compile(r"(?<!\w)" + re.escape(str(skill).strip()) + r"(?!\w)", re.IGNORECASE)


def recommend_jobs(user_skills, tfidf=None, tfidf_matrix=None, postings_df=None, top_n=10):
    """Recommend jobs using bounded keyword evidence instead of a huge TF-IDF matrix.

    This is intentionally memory-safe for Streamlit Cloud.  We score the
    structured title field first and then inspect skills_desc only for the
    candidate rows, so selecting a software skill such as Alteryx cannot
    trigger a large TF-IDF allocation.
    """
    if postings_df is None:
        postings_df = load_postings()

    skills = [str(s).strip() for s in user_skills if str(s).strip()]
    if not skills:
        return pd.DataFrame()

    # Load structured skills only for the recommendation request.
    skill_text = _read_columns(["title", "skills_desc"])
    for col in ["title", "skills_desc"]:
        if col not in skill_text.columns:
            skill_text[col] = ""
        skill_text[col] = skill_text[col].fillna("").astype(str)

    score = np.zeros(len(skill_text), dtype=np.float32)
    title_text = skill_text["title"]
    skills_text = skill_text["skills_desc"]

    for skill in skills:
        pattern = _skill_pattern(skill)
        title_hit = title_text.str.contains(pattern, na=False)
        skill_hit = skills_text.str.contains(pattern, na=False)
        # Title evidence is stronger; structured skills evidence is still useful.
        score += title_hit.to_numpy(dtype=np.float32) * 3.0
        score += skill_hit.to_numpy(dtype=np.float32) * 1.0

    matched = score > 0
    if not matched.any():
        # Keep the UI useful even when the exact software name is absent.
        result = postings_df.head(top_n).copy()
        result["Match Score"] = 0.0
        return result.reset_index(drop=True)

    candidate_idx = np.flatnonzero(matched)
    order = candidate_idx[np.argsort(score[candidate_idx])[::-1]]
    order = order[:top_n]

    cols = [
        c for c in [
            "title", "company_name", "location",
            "formatted_work_type", "formatted_experience_level",
            "remote_label", "normalized_salary", "job_posting_url"
        ] if c in postings_df.columns
    ]

    result = postings_df.iloc[order][cols].copy()
    result["Match Score"] = score[order] / max(3.0 * len(skills), 1.0) * 100
    return result.reset_index(drop=True)
