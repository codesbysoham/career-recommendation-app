"""ML/analytics layer for career ranking.

The ML layer is deliberately independent from the LLM layer. Career ranking
uses one unified feature space made from O*NET general skills plus O*NET
workplace technologies. No skill name receives a special-case rule.

ENGINE_VERSION is intentionally bumped whenever the matcher contract changes;
it also helps Streamlit Cloud replace an older cached module after deployment.
"""

ENGINE_VERSION = "2026.08.24-unified-v3"

import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


def _configure_kaggle_from_secrets():
    """Populate Kaggle auth env vars before CareerClassifier imports kaggle."""
    try:
        token = st.secrets.get("kaggle_api_token") or st.secrets.get("KAGGLE_API_TOKEN")
        if token:
            os.environ["KAGGLE_API_TOKEN"] = str(token).strip()
            return

        username = st.secrets.get("kaggle_username") or st.secrets.get("KAGGLE_USERNAME")
        key = st.secrets.get("kaggle_key") or st.secrets.get("KAGGLE_KEY")
        if username and key:
            os.environ["KAGGLE_USERNAME"] = str(username).strip()
            os.environ["KAGGLE_KEY"] = str(key).strip()
    except Exception:
        # Career matching itself does not require Kaggle until job-market data
        # is requested. Do not make optional configuration crash app startup.
        pass


# Kaggle 2.x auto-authenticates during package import, so credentials must be
# available before CareerClassifier imports KaggleApi.
_configure_kaggle_from_secrets()

from CareerClassifier import (  # noqa: E402
    build_occupation_profiles,
    canonical_skill,
    load_onet_data,
    _norm_soc,
)


def _software_skill_matrix(software_df):
    """Build an occupation x technology matrix from O*NET software data."""
    if software_df is None or software_df.empty:
        return pd.DataFrame()

    required = {"O*NET-SOC Code", "Workplace Example"}
    if not required.issubset(software_df.columns):
        return pd.DataFrame()

    df = software_df.copy()
    df["O*NET-SOC Code"] = df["O*NET-SOC Code"].map(_norm_soc)
    df["skill"] = df["Workplace Example"].map(canonical_skill)
    df = df[df["O*NET-SOC Code"].ne("") & df["skill"].notna()].copy()
    df = df[df["skill"].astype(str).str.len().between(2, 120)]
    if df.empty:
        return pd.DataFrame()

    hot = df.get("Hot Technology", pd.Series("N", index=df.index))
    demand = df.get("In Demand", pd.Series("N", index=df.index))
    df["hot"] = hot.astype(str).str.upper().eq("Y").astype(float)
    df["demand"] = demand.astype(str).str.upper().eq("Y").astype(float)
    df["signal"] = 0.55 + 0.25 * df["hot"] + 0.20 * df["demand"]

    return df.pivot_table(
        index="O*NET-SOC Code",
        columns="skill",
        values="signal",
        aggfunc="max",
        fill_value=0.0,
    )


def rank_careers(selected_skills, skills_df, software_df=None, top_n=10):
    """Rank occupations against any skill in the unified skill library."""
    importance, _ = build_occupation_profiles(skills_df)
    if importance.empty:
        return pd.DataFrame()

    if software_df is None:
        try:
            _, _, _, software_df = load_onet_data()
        except Exception:
            software_df = pd.DataFrame()

    general = importance.clip(lower=0, upper=5).astype(float) / 5.0
    technology = _software_skill_matrix(software_df)
    if not technology.empty:
        general_soc = general.index.get_level_values(0)
        technology = technology.reindex(index=general_soc, fill_value=0.0)
        technology.index = general.index
        all_features = general.columns.union(technology.columns)
        general = general.reindex(columns=all_features, fill_value=0.0)
        technology = technology.reindex(columns=all_features, fill_value=0.0)
        matrix = general.combine(technology, np.maximum)
    else:
        matrix = general

    matrix = matrix.fillna(0.0).astype(float)

    selected = []
    for skill in selected_skills:
        canonical = canonical_skill(skill)
        if canonical in matrix.columns:
            selected.append(canonical)
    selected = list(dict.fromkeys(selected))
    if not selected:
        return pd.DataFrame()

    occupation_matrix = matrix.to_numpy(dtype=np.float32)
    user_vector = np.zeros((1, matrix.shape[1]), dtype=np.float32)
    selected_idx = [matrix.columns.get_loc(s) for s in selected]
    user_vector[0, selected_idx] = 1.0
    cosine = cosine_similarity(user_vector, occupation_matrix)[0]

    selected_strength = matrix[selected].mean(axis=1).to_numpy(dtype=np.float32)
    matched_count = (matrix[selected] > 0).sum(axis=1).to_numpy(dtype=int)

    k = min(max(len(selected) * 2, 5), matrix.shape[1])
    top_k_sum = np.sort(occupation_matrix, axis=1)[:, -k:].sum(axis=1)
    matched_sum = matrix[selected].sum(axis=1).to_numpy(dtype=np.float32)
    coverage = np.divide(
        matched_sum,
        top_k_sum,
        out=np.zeros_like(matched_sum, dtype=np.float32),
        where=top_k_sum > 0,
    )
    coverage = np.clip(coverage, 0, 1)

    fit_score = (0.55 * cosine + 0.30 * selected_strength + 0.15 * coverage) * 100

    result = pd.DataFrame({
        "O*NET-SOC Code": matrix.index.get_level_values(0),
        "Career": matrix.index.get_level_values(1),
        "Match Score": np.round(np.clip(fit_score, 0, 100), 1),
        "Vector Similarity": np.round(cosine * 100, 1),
        "Skill Strength": np.round(selected_strength * 100, 1),
        "Skill Coverage": np.round(coverage * 100, 1),
        "Skills Matched": matched_count,
    })

    return (
        result.sort_values(
            ["Match Score", "Skills Matched", "Vector Similarity"],
            ascending=[False, False, False],
            kind="stable",
        )
        .head(top_n)
        .reset_index(drop=True)
    )
