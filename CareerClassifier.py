# Career Intelligence Dashboard backend
import os
import re
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


POSTINGS_PATH = "data/postings.csv"
SKILLS_PATH = "data/Skills.xlsx"
SOFTWARE_SKILLS_PATH = "data/Software Skills.xlsx"
KNOWLEDGE_PATH = "data/Knowledge.xlsx"
OCCUPATIONS_PATH = "data/Occupation Data.xlsx"
KAGGLE_DATASET = "soham1510/career-postings-dataset"


def _download_postings():
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
    software = pd.read_excel(SOFTWARE_SKILLS_PATH)

    return skills, knowledge, occupations, software


@st.cache_resource(show_spinner="Loading job market and building model...")
def load_pipeline():
    _download_postings()

    postings = pd.read_csv(POSTINGS_PATH, low_memory=False)

    required = {
        "title": "",
        "company_name": "Unknown",
        "location": "Unknown",
        "description": "",
        "skills_desc": "",
    }
    for col, default in required.items():
        if col not in postings.columns:
            postings[col] = default

    for col in [
        "max_salary", "med_salary", "min_salary", "normalized_salary",
        "views", "applies"
    ]:
        if col not in postings.columns:
            postings[col] = np.nan
        postings[col] = pd.to_numeric(postings[col], errors="coerce")

    for col in ["title", "company_name", "location", "description", "skills_desc"]:
        postings[col] = postings[col].fillna("").astype(str)

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
    else:
        postings["listed_date"] = pd.NaT

    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 1),
        stop_words="english",
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
        dtype=np.float32,
    )

    # Title is deliberately included: it is a strong career signal.
    model_text = postings["title"] + " " + postings["description"]
    tfidf_matrix = tfidf.fit_transform(model_text)

    skills, knowledge, occupations, software = load_onet_data()

    return postings, tfidf, tfidf_matrix, skills, knowledge, occupations, software


def _norm_skill(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9+#.&/-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _display_skill(value):
    value = re.sub(r"\s+", " ", str(value).strip())
    return value


# Common variants that should be treated as the same user-facing skill.
ALIASES = {
    "python programming": "Python",
    "python programming language": "Python",
    "structured query language": "SQL",
    "microsoft sql server": "SQL Server",
    "ms excel": "Microsoft Excel",
    "microsoft excel": "Microsoft Excel",
    "power bi": "Power BI",
    "microsoft power bi": "Power BI",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "natural language processing": "Natural Language Processing",
    "nlp": "Natural Language Processing",
    "artificial intelligence": "Artificial Intelligence",
    "ai": "Artificial Intelligence",
}


def _canonical(value):
    clean = _display_skill(value)
    key = _norm_skill(clean)
    return ALIASES.get(key, clean)


@st.cache_data(show_spinner="Building unified skill library...")
def build_skill_master(skills_df, software_df, postings_df):
    """
    Create ONE user-facing skill universe.

    Sources:
      1. O*NET Skills.xlsx
      2. O*NET 30.3 Software Skills.xlsx
      3. Skills explicitly appearing in the 2,439 non-empty skills_desc fields.

    skills_desc is treated as market evidence, not as a clean taxonomy.
    """
    rows = []

    # ---- O*NET general skills ----
    if "Element Name" in skills_df.columns:
        for s in skills_df["Element Name"].dropna().unique():
            skill = _canonical(s)
            if skill:
                rows.append({
                    "skill": skill,
                    "source_onet": 1,
                    "source_software": 0,
                    "market_frequency": 0,
                    "hot_technology": 0,
                    "in_demand": 0,
                })

    # ---- O*NET software/workplace examples ----
    if "Workplace Example" in software_df.columns:
        for _, r in software_df.iterrows():
            raw = r["Workplace Example"]
            if pd.isna(raw):
                continue
            skill = _canonical(raw)
            if not skill:
                continue
            rows.append({
                "skill": skill,
                "source_onet": 0,
                "source_software": 1,
                "market_frequency": 0,
                "hot_technology": int(str(r.get("Hot Technology", "N")).upper() == "Y"),
                "in_demand": int(str(r.get("In Demand", "N")).upper() == "Y"),
            })

    # First collapse the O*NET vocabulary.
    master = pd.DataFrame(rows)
    master = (
        master.groupby("skill", as_index=False)
        .agg(
            source_onet=("source_onet", "max"),
            source_software=("source_software", "max"),
            market_frequency=("market_frequency", "sum"),
            hot_technology=("hot_technology", "max"),
            in_demand=("in_demand", "max"),
        )
    )

    # ---- Market evidence from the 2,439 structured-ish skills_desc rows ----
    # Match only against the O*NET-derived vocabulary. This prevents random
    # prose words from becoming "skills".
    vocab = {
        _norm_skill(s): s
        for s in master["skill"]
        if len(_norm_skill(s)) >= 2
    }

    # Add a small set of very common job-market skills that may not appear
    # as exact O*NET Workplace Examples.
    market_seed = [
        "Python", "SQL", "R", "Java", "JavaScript", "C++", "C#", "Excel",
        "Microsoft Excel", "Power BI", "Tableau", "Git", "GitHub", "AWS",
        "Microsoft Azure", "Docker", "Kubernetes", "TensorFlow", "PyTorch",
        "Pandas", "NumPy", "Scikit-learn", "Spark", "Hadoop", "PostgreSQL",
        "MySQL", "Oracle", "MongoDB", "Machine Learning", "Deep Learning",
        "Data Analysis", "Data Science", "Statistics", "Data Visualization",
        "Natural Language Processing", "Computer Vision", "Artificial Intelligence",
    ]
    for s in market_seed:
        c = _canonical(s)
        vocab.setdefault(_norm_skill(c), c)

    freq = {}

    def ngrams(tokens, max_n=7):
        for n in range(1, min(max_n, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                yield " ".join(tokens[i:i+n])

    market_text = postings_df.loc[
        postings_df["skills_desc"].str.strip().ne(""),
        "skills_desc"
    ].tolist()

    # 2,439 rows is small enough for this controlled vocabulary scan.
    for text in market_text:
        text_norm = _norm_skill(text)
        tokens = text_norm.split()
        found = set()

        # Exact vocabulary phrase matching via n-grams.
        for phrase in ngrams(tokens, max_n=7):
            if phrase in vocab:
                found.add(vocab[phrase])

        for skill in found:
            freq[skill] = freq.get(skill, 0) + 1

    if freq:
        freq_df = pd.DataFrame(
            [{"skill": k, "market_frequency": v} for k, v in freq.items()]
        )
        master = master.merge(
            freq_df,
            on="skill",
            how="outer",
            suffixes=("", "_new"),
        )
        master["market_frequency"] = master["market_frequency_new"].fillna(
            master["market_frequency"]
        )
        master.drop(columns=["market_frequency_new"], inplace=True)

    # Ensure seed skills that genuinely occur in the controlled vocabulary
    # are visible even when they have no skills_desc evidence.
    seed_rows = []
    existing = set(master["skill"])
    for s in market_seed:
        c = _canonical(s)
        if c not in existing:
            seed_rows.append({
                "skill": c,
                "source_onet": 0,
                "source_software": 0,
                "market_frequency": int(freq.get(c, 0)),
                "hot_technology": 0,
                "in_demand": 0,
            })

    if seed_rows:
        master = pd.concat([master, pd.DataFrame(seed_rows)], ignore_index=True)

    master["source"] = np.select(
        [
            (master["source_onet"] == 1) & (master["source_software"] == 1),
            master["source_software"] == 1,
            master["source_onet"] == 1,
            master["market_frequency"] > 0,
        ],
        [
            "O*NET + Software",
            "O*NET Software",
            "O*NET Skill",
            "Job postings",
        ],
        default="Job postings",
    )

    # User-facing list: remove ultra-generic/obviously malformed software
    # names, but retain legitimate niche software technologies.
    master = master[
        master["skill"].str.len().between(2, 120)
    ].copy()

    master["market_frequency"] = master["market_frequency"].fillna(0).astype(int)
    master["hot_technology"] = master["hot_technology"].fillna(0).astype(int)
    master["in_demand"] = master["in_demand"].fillna(0).astype(int)

    return master.sort_values(
        ["market_frequency", "hot_technology", "skill"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


@st.cache_data
def get_skill_profile(skills_df):
    importance = (
        skills_df[skills_df["Scale ID"] == "IM"]
        .pivot_table(
            index=["O*NET-SOC Code", "Title"],
            columns="Element Name",
            values="Data Value",
            aggfunc="mean",
        )
        .fillna(0)
    )
    level = (
        skills_df[skills_df["Scale ID"] == "LV"]
        .pivot_table(
            index=["O*NET-SOC Code", "Title"],
            columns="Element Name",
            values="Data Value",
            aggfunc="mean",
        )
        .fillna(0)
    )
    return importance, level


def career_match(selected_skills, skills_df, skill_master, top_n=10):
    """Rank O*NET occupations by selected O*NET/general skills."""
    importance, _ = get_skill_profile(skills_df)

    # Only O*NET-mappable selected skills contribute to occupation scoring.
    selected = [
        s for s in selected_skills
        if s in importance.columns
    ]

    if not selected:
        return pd.DataFrame()

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
    importance, _ = get_skill_profile(skills_df)
    rows = importance.reset_index()
    rows = rows[rows["Title"] == occupation_title]

    if rows.empty:
        return pd.DataFrame()

    row = rows.iloc[0]
    values = pd.DataFrame({
        "Skill": importance.columns,
        "Importance": [row.get(c, 0) for c in importance.columns],
    })
    values["Selected"] = values["Skill"].isin(selected_skills)

    return (
        values.sort_values("Importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def recommend_jobs(user_skills, tfidf, tfidf_matrix, postings_df, top_n=10):
    text = " ".join(user_skills)
    vector = tfidf.transform([text])
    similarity = cosine_similarity(vector, tfidf_matrix)[0]

    cols = [
        c for c in [
            "title", "company_name", "location",
            "formatted_work_type", "formatted_experience_level",
            "remote_label", "normalized_salary", "job_posting_url"
        ]
        if c in postings_df.columns
    ]

    result = postings_df[cols].copy()
    result["Match Score"] = similarity * 100

    return (
        result.sort_values("Match Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def filter_jobs(postings, title=None, location=None, work_type=None,
                experience=None, remote=None, min_salary=None):
    df = postings

    if title:
        df = df[df["title"].str.contains(title, case=False, na=False)]

    if location:
        df = df[df["location"].str.contains(location, case=False, na=False)]

    if work_type and work_type != "All" and "formatted_work_type" in df:
        df = df[df["formatted_work_type"].fillna("Unknown") == work_type]

    if experience and experience != "All" and "formatted_experience_level" in df:
        df = df[df["formatted_experience_level"].fillna("Unknown") == experience]

    if remote and remote != "All":
        df = df[df["remote_label"] == remote]

    if min_salary is not None:
        df = df[df["normalized_salary"].fillna(0) >= min_salary]

    return df


def top_counts(series, n=10):
    return series.fillna("Unknown").astype(str).value_counts().head(n)


def dashboard_metrics(postings):
    total = len(postings)
    companies = postings["company_name"].replace("", np.nan).nunique()
    titles = postings["title"].replace("", np.nan).nunique()
    remote = int((postings["remote_label"] == "Remote").sum())
    salary = postings["normalized_salary"].dropna()
    median_salary = salary.median() if not salary.empty else np.nan
    return total, companies, titles, remote, median_salary
