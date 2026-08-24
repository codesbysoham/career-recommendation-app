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
    """Compatibility wrapper; no large TF-IDF matrix is constructed."""
    return load_postings(), None, None


def _normalise_text(value):
    """Make matching tolerant of punctuation, symbols and naming variations."""
    text = str(value).lower()
    text = text.replace("&", " and ")
    text = text.replace(".net", " dotnet ")
    text = text.replace("c++", " cpp ")
    text = text.replace("c#", " csharp ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _skill_terms(skill):
    """Return useful tokens for fallback matching of arbitrary library skills."""
    normalised = _normalise_text(skill)
    terms = [t for t in normalised.split() if len(t) >= 2]

    # Common O*NET / software-library phrasing where the exact phrase may
    # differ from the wording used in a job posting.
    aliases = {
        "ios": ["ios", "iphone", "ipad", "swift"],
        "android": ["android", "kotlin"],
        "website development software": ["web", "website", "development", "developer", "frontend", "backend"],
        "web development": ["web", "website", "frontend", "backend"],
        "data analysis software": ["data", "analytics", "analysis"],
        "database software": ["database", "sql", "mysql", "postgresql", "oracle"],
        "spreadsheet software": ["excel", "spreadsheet"],
        "presentation software": ["powerpoint", "presentation"],
    }
    key = normalised
    if key in aliases:
        terms.extend(aliases[key])

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(terms))


def _safe_contains(series, pattern):
    try:
        return series.str.contains(pattern, case=False, regex=True, na=False).to_numpy(dtype=np.float32)
    except (re.error, TypeError, ValueError):
        # User/library strings must never be able to break the app.
        literal = str(pattern).lower()
        return series.str.lower().str.contains(literal, regex=False, na=False).to_numpy(dtype=np.float32)


def recommend_jobs(user_skills, tfidf=None, tfidf_matrix=None, postings_df=None, top_n=10):
    """Recommend jobs robustly for any skill from the unified library.

    Matching uses three levels:
      1. exact phrase in title/structured skills,
      2. token/alias evidence when the posting uses different wording,
      3. deterministic market fallback when no textual evidence exists.

    No TF-IDF matrix is built, so the function stays bounded in memory.
    It always returns a non-empty result when the postings dataset exists.
    """
    if postings_df is None:
        postings_df = load_postings()

    skills = [str(s).strip() for s in user_skills if str(s).strip()]
    if not skills or postings_df is None or postings_df.empty:
        return pd.DataFrame()

    # Read only the two textual fields needed for matching.
    skill_text = _read_columns(["title", "skills_desc"])
    for col in ["title", "skills_desc"]:
        if col not in skill_text.columns:
            skill_text[col] = ""
        skill_text[col] = skill_text[col].fillna("").astype(str)

    title_norm = skill_text["title"].map(_normalise_text)
    skills_norm = skill_text["skills_desc"].map(_normalise_text)
    score = np.zeros(len(skill_text), dtype=np.float32)

    for skill in skills:
        skill_norm = _normalise_text(skill)
        if not skill_norm:
            continue

        # Exact phrase: strongest evidence.
        exact_title = title_norm.str.contains(re.escape(skill_norm), regex=True, na=False).to_numpy(dtype=np.float32)
        exact_skill = skills_norm.str.contains(re.escape(skill_norm), regex=True, na=False).to_numpy(dtype=np.float32)
        score += exact_title * 6.0
        score += exact_skill * 4.0

        # Token/alias fallback: useful when O*NET and job-market wording differ.
        terms = _skill_terms(skill)
        if terms:
            term_hits_title = np.zeros(len(skill_text), dtype=np.float32)
            term_hits_skill = np.zeros(len(skill_text), dtype=np.float32)
            for term in terms:
                if len(term) < 2:
                    continue
                pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
                term_hits_title += _safe_contains(title_norm, pattern)
                term_hits_skill += _safe_contains(skills_norm, pattern)

            score += np.minimum(term_hits_title, 3.0) * 2.0
            score += np.minimum(term_hits_skill, 4.0) * 1.0

    matched = score > 0

    # If nothing in the raw text can be linked to the selected skill, still
    # give the user useful market results instead of a dead-end message.
    if not matched.any():
        result = postings_df.head(top_n).copy()
        result["Match Score"] = 0.0
        result["Match Basis"] = "Market fallback — no direct skill evidence found"
        return result.reset_index(drop=True)

    candidate_idx = np.flatnonzero(matched)
    order = candidate_idx[np.argsort(score[candidate_idx])[::-1]][:top_n]

    cols = [
        c for c in [
            "title", "company_name", "location",
            "formatted_work_type", "formatted_experience_level",
            "remote_label", "normalized_salary", "job_posting_url"
        ] if c in postings_df.columns
    ]

    result = postings_df.iloc[order][cols].copy()
    max_possible = max(10.0 * len(skills), 1.0)
    result["Match Score"] = np.minimum(score[order] / max_possible * 100.0, 100.0).round(1)
    result["Match Basis"] = "Skill/title evidence"
    return result.reset_index(drop=True)
