# Career Intelligence Dashboard — ML Skill Gap Engine
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


@st.cache_resource(show_spinner="Loading job market and ML model...")
def load_pipeline():
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

    for col in ["max_salary", "med_salary", "min_salary",
                "normalized_salary", "views", "applies"]:
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

    # TF-IDF powers job recommendation and occupation-to-job similarity.
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

    skills, knowledge, occupations, software = load_onet_data()

    # Normalize SOC keys once so occupation profiles remain stable even when
    # the source Excel files use slightly different formatting.
    for frame in (skills, occupations, software):
        if "O*NET-SOC Code" in frame.columns:
            frame["O*NET-SOC Code"] = frame["O*NET-SOC Code"].map(_norm_soc)

    return postings, tfidf, tfidf_matrix, skills, knowledge, occupations, software


def _norm(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9+#.&/-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _norm_soc(value):
    """Normalize O*NET-SOC codes so Excel formatting differences do not break joins."""
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    value = value.replace("O*NET-SOC ", "").replace("O*NET-SOC:", "")
    value = re.sub(r"\s+", "", value)
    return value


ALIASES = {
    "python programming": "Python",
    "python programming language": "Python",
    "structured query language": "SQL",
    "ms excel": "Microsoft Excel",
    "microsoft excel": "Microsoft Excel",
    "excel": "Excel",
    "power bi": "Power BI",
    "microsoft power bi": "Power BI",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
}


def canonical_skill(value):
    clean = re.sub(r"\s+", " ", str(value).strip())
    return ALIASES.get(_norm(clean), clean)


@st.cache_data(show_spinner="Building unified skill library...")
def build_skill_master(skills_df, software_df, postings_df):
    """Build one searchable skill universe and calculate real job-market demand.

    The library combines O*NET skills and O*NET software/technology skills.
    Market frequency is calculated from the *skills_desc* and job *title*
    fields using a trie, so we avoid the impossible skill x posting nested loop.

    market_frequency = number of distinct job postings in which the skill was
    detected.  It is therefore a posting-demand signal, not a raw word count.
    """
    rows = []

    # ---------- O*NET general skills ----------
    if "Element Name" in skills_df.columns:
        for s in skills_df["Element Name"].dropna().unique():
            s = canonical_skill(s)
            if 2 <= len(s) <= 120:
                rows.append({
                    "skill": s,
                    "source_onet": 1,
                    "source_software": 0,
                    "market_frequency": 0,
                    "hot_technology": 0,
                    "in_demand": 0,
                })

    # ---------- O*NET software / technology ----------
    if "Workplace Example" in software_df.columns:
        for _, r in software_df.iterrows():
            s = r.get("Workplace Example")
            if pd.isna(s):
                continue
            s = canonical_skill(s)
            if not 2 <= len(s) <= 120:
                continue
            rows.append({
                "skill": s,
                "source_onet": 0,
                "source_software": 1,
                "market_frequency": 0,
                "hot_technology": int(
                    str(r.get("Hot Technology", "N")).upper() == "Y"
                ),
                "in_demand": int(
                    str(r.get("In Demand", "N")).upper() == "Y"
                ),
            })

    master = pd.DataFrame(rows).groupby("skill", as_index=False).agg(
        source_onet=("source_onet", "max"),
        source_software=("source_software", "max"),
        market_frequency=("market_frequency", "max"),
        hot_technology=("hot_technology", "max"),
        in_demand=("in_demand", "max"),
    )

    if master.empty:
        master["source"] = []
        return master

    # ---------- Real job-market frequency ----------
    # Match against the structured skills field first, plus title.  This is
    # much cleaner than trying to infer skills from every word in a long job
    # description and is fast enough for the 123k-posting dataset.
    text_cols = []
    if "skills_desc" in postings_df.columns:
        text_cols.append("skills_desc")
    if "title" in postings_df.columns:
        text_cols.append("title")

    if text_cols:
        docs = postings_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

        # Map normalized token sequences -> canonical skill name.
        # Restrict matching phrases to 6 tokens; longer O*NET descriptions
        # are not useful as selectable market skills.
        trie = {}
        skill_token_keys = {}
        for skill in master["skill"].tolist():
            key = _norm(skill)
            tokens = re.findall(r"[a-z0-9+#.&/-]+", key)
            if not tokens or len(tokens) > 6:
                continue
            node = trie
            for token in tokens:
                node = node.setdefault(token, {})
            node.setdefault("__skills__", []).append(skill)
            skill_token_keys[skill] = tuple(tokens)

        freq = {}
        token_pattern = re.compile(r"[a-z0-9+#.&/-]+")

        for text in docs:
            tokens = token_pattern.findall(_norm(text))
            if not tokens:
                continue

            found = set()
            n = len(tokens)
            for i in range(n):
                node = trie
                j = i
                while j < n and tokens[j] in node:
                    node = node[tokens[j]]
                    j += 1
                    if "__skills__" in node:
                        found.update(node["__skills__"])
                    if j - i >= 6:
                        break

            for skill in found:
                freq[skill] = freq.get(skill, 0) + 1

        if freq:
            master["market_frequency"] = master["skill"].map(freq).fillna(0).astype(int)

    master["source"] = np.select(
        [
            (master["source_onet"] == 1) & (master["source_software"] == 1),
            master["source_software"] == 1,
            master["source_onet"] == 1,
        ],
        ["O*NET + Software", "O*NET Software", "O*NET Skill"],
        default="Skill Library",
    )

    master["market_share"] = (
        master["market_frequency"] / max(len(postings_df), 1) * 100
    ).round(2)

    return master[
        master["skill"].str.len().between(2, 120)
    ].sort_values(
        ["market_frequency", "hot_technology", "in_demand", "skill"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


@st.cache_data
def build_occupation_profiles(skills_df):
    """Create O*NET occupation skill profiles keyed by normalized SOC code."""
    df = skills_df.copy()
    if "O*NET-SOC Code" in df.columns:
        df["O*NET-SOC Code"] = df["O*NET-SOC Code"].map(_norm_soc)
    df["Data Value"] = pd.to_numeric(df["Data Value"], errors="coerce")

    importance = df[df["Scale ID"].eq("IM")].pivot_table(
        index=["O*NET-SOC Code", "Title"],
        columns="Element Name",
        values="Data Value",
        aggfunc="mean",
    ).fillna(0)

    level = df[df["Scale ID"].eq("LV")].pivot_table(
        index=["O*NET-SOC Code", "Title"],
        columns="Element Name",
        values="Data Value",
        aggfunc="mean",
    ).fillna(0)

    return importance, level


def occupation_lookup(occupations_df):
    """Stable UI lookup: display title -> normalized SOC code."""
    x = occupations_df[["O*NET-SOC Code", "Title"]].dropna().drop_duplicates().copy()
    x["O*NET-SOC Code"] = x["O*NET-SOC Code"].map(_norm_soc)
    return x[x["O*NET-SOC Code"].ne("")].sort_values("Title").reset_index(drop=True)


def get_occupation_skill_gaps(
    soc_code,
    selected_skills,
    skills_df,
    software_df,
    postings_df,
    top_n=12,
):
    """
    Rank missing skills for a target occupation.

    General O*NET skills:
        importance drives the score.

    Software skills:
        Hot Technology + In Demand + occupation association drive the score.

    ML layer:
        skill co-occurrence with the user's existing skills provides a
        personalization bonus when the skill appears in job-posting skill text.
    """
    soc_code = _norm_soc(soc_code)
    selected_norm = {_norm(canonical_skill(s)) for s in selected_skills}

    # ---------- O*NET general skills ----------
    importance, _ = build_occupation_profiles(skills_df)
    general = pd.DataFrame()

    if soc_code in importance.index.get_level_values(0):
        row = importance.loc[soc_code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        general = pd.DataFrame({
            "skill": importance.columns,
            "importance": [float(row.get(c, 0)) for c in importance.columns],
        })
        general["skill_type"] = "O*NET Skill"
        general["software_hot"] = 0
        general["software_demand"] = 0
        general["onet_software"] = 0

    # ---------- O*NET Software Skills ----------
    software = software_df.copy()
    if "O*NET-SOC Code" in software.columns:
        software["O*NET-SOC Code"] = software["O*NET-SOC Code"].map(_norm_soc)
    software = software[
        software["O*NET-SOC Code"].eq(soc_code)
    ].copy()

    tech = pd.DataFrame()
    if not software.empty:
        tech = software[[
            "Workplace Example", "Hot Technology", "In Demand"
        ]].dropna(subset=["Workplace Example"]).copy()

        tech["skill"] = tech["Workplace Example"].map(canonical_skill)
        tech["software_hot"] = (
            tech["Hot Technology"].astype(str).str.upper().eq("Y").astype(int)
        )
        tech["software_demand"] = (
            tech["In Demand"].astype(str).str.upper().eq("Y").astype(int)
        )
        tech["importance"] = 0.0
        tech["skill_type"] = "Technical / Software"
        tech["onet_software"] = 1

        tech = tech.groupby("skill", as_index=False).agg({
            "importance": "max",
            "software_hot": "max",
            "software_demand": "max",
            "skill_type": "first",
            "onet_software": "max",
        })

    candidates = pd.concat(
        [
            general[[
                "skill", "importance", "software_hot",
                "software_demand", "skill_type", "onet_software"
            ]],
            tech[[
                "skill", "importance", "software_hot",
                "software_demand", "skill_type", "onet_software"
            ]],
        ],
        ignore_index=True,
    )

    if candidates.empty:
        return candidates

    # Deduplicate exact canonical names.
    candidates["skill_key"] = candidates["skill"].map(_norm)
    candidates = candidates.sort_values(
        ["software_hot", "software_demand", "importance"],
        ascending=False
    ).drop_duplicates("skill_key")

    # ---------- Lightweight personalization ----------
    # Vectorized over the postings column; avoids candidate x posting x skill
    # nested Python loops.
    co_bonus = {}
    if selected_norm and "skills_desc" in postings_df.columns:
        texts = postings_df["skills_desc"].fillna("").astype(str).str.lower()

        def mask_for_skill(skill_key):
            if not skill_key:
                return pd.Series(False, index=texts.index)
            pattern = r"(?<!\w)" + re.escape(skill_key) + r"(?!\w)"
            return texts.str.contains(pattern, regex=True, na=False)

        selected_mask = pd.Series(False, index=texts.index)
        for key in selected_norm:
            selected_mask |= mask_for_skill(key)

        for _, r in candidates.iterrows():
            candidate_key = r["skill_key"]
            if candidate_key in selected_norm:
                continue
            candidate_mask = mask_for_skill(candidate_key)
            target_count = int(candidate_mask.sum())
            co_bonus[candidate_key] = (
                int((candidate_mask & selected_mask).sum()) / target_count
                if target_count else 0.0
            )

    candidates["cooccurrence"] = candidates["skill_key"].map(co_bonus).fillna(0.0)

    # Score:
    # 40% O*NET importance
    # 25% software "hot" signal
    # 20% software "in demand" signal
    # 15% complementarity with current skills
    #
    # For software skills, importance is absent, so the other signals carry
    # the ranking. This is intentional rather than inventing an O*NET score.
    imp = candidates["importance"].clip(0, 5) / 5
    hot = candidates["software_hot"]
    demand = candidates["software_demand"]
    co = candidates["cooccurrence"]

    candidates["priority_score"] = (
        0.40 * imp +
        0.25 * hot +
        0.20 * demand +
        0.15 * co
    ) * 100

    candidates["already_have"] = candidates["skill_key"].isin(selected_norm)
    candidates = candidates[~candidates["already_have"]].copy()

    candidates["reason"] = np.select(
        [
            candidates["cooccurrence"] >= 0.50,
            candidates["software_hot"].eq(1),
            candidates["software_demand"].eq(1),
            candidates["importance"] >= 4.0,
        ],
        [
            "Strongly complements your existing skills in job-posting data.",
            "Marked as a Hot Technology for this occupation.",
            "Marked as In Demand for this occupation.",
            "High O*NET importance for this occupation.",
        ],
        default="Relevant to the target occupation.",
    )

    return candidates.sort_values(
        ["priority_score", "importance", "software_demand", "skill"],
        ascending=[False, False, False, True],
    ).head(top_n).reset_index(drop=True)


def get_skill_gap_summary(gaps):
    if gaps.empty:
        return "No target-occupation skill profile was found."

    high = gaps[gaps["priority_score"] >= 65]["skill"].tolist()
    if high:
        return "Your highest-priority gaps are: " + ", ".join(high[:5]) + "."
    return "The model found several relevant skills; start with the highest-ranked items below."


def career_match(selected_skills, skills_df, top_n=10):
    """Match selected O*NET general skills to occupations using SOC-keyed profiles."""
    importance, _ = build_occupation_profiles(skills_df)
    selected = []
    for skill in selected_skills:
        canonical = canonical_skill(skill)
        if canonical in importance.columns:
            selected.append(canonical)
    selected = list(dict.fromkeys(selected))

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

    return result.sort_values(
        ["Match Score", "Skills Matched"],
        ascending=False
    ).head(top_n).reset_index(drop=True)


def recommend_jobs(user_skills, tfidf, tfidf_matrix, postings_df, top_n=10):
    text = " ".join(user_skills)
    vector = tfidf.transform([text])
    similarity = cosine_similarity(vector, tfidf_matrix)[0]

    cols = [
        c for c in [
            "title", "company_name", "location",
            "formatted_work_type", "formatted_experience_level",
            "remote_label", "normalized_salary", "job_posting_url"
        ] if c in postings_df.columns
    ]

    result = postings_df[cols].copy()
    result["Match Score"] = similarity * 100
    return result.sort_values("Match Score", ascending=False).head(top_n).reset_index(drop=True)


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
