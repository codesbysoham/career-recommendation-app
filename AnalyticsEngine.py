"""ML/analytics layer for career ranking.

This module keeps the core career score independent from the LLM layer.
It represents each occupation as an O*NET skill-importance vector and compares
that vector with the user's selected-skill profile using cosine similarity.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from CareerClassifier import build_occupation_profiles, canonical_skill


def rank_careers(selected_skills, skills_df, top_n=10):
    """Rank occupations using O*NET skill-vector similarity and importance.

    Components:
      - 55% cosine similarity between user skills and occupation skill profile
      - 30% mean O*NET importance of the user's matched skills
      - 15% coverage of the occupation's highest-priority skill set

    No LLM is used anywhere in this calculation.
    """
    importance, _ = build_occupation_profiles(skills_df)
    if importance.empty:
        return pd.DataFrame()

    selected = []
    for skill in selected_skills:
        canonical = canonical_skill(skill)
        if canonical in importance.columns:
            selected.append(canonical)
    selected = list(dict.fromkeys(selected))

    if not selected:
        return pd.DataFrame()

    # O*NET importance is on a 1-5 scale. Clip defensively in case a source
    # release contains an unexpected value.
    matrix = importance.clip(lower=0, upper=5).astype(float)
    occupation_matrix = matrix.to_numpy(dtype=np.float32)

    # User profile: selected skills are present, everything else is absent.
    user_vector = np.zeros((1, matrix.shape[1]), dtype=np.float32)
    selected_idx = [matrix.columns.get_loc(s) for s in selected]
    user_vector[0, selected_idx] = 1.0

    # Vector similarity captures how closely the user's skill profile aligns
    # with the complete occupational skill-importance profile.
    cosine = cosine_similarity(user_vector, occupation_matrix)[0]

    selected_importance = matrix[selected].mean(axis=1) / 5.0
    matched_count = (matrix[selected] > 0).sum(axis=1)

    # Compare the user's matched importance against the occupation's top-K
    # skills. K grows with the size of the user's profile but stays bounded.
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

    fit_score = (
        0.55 * cosine
        + 0.30 * selected_importance.to_numpy(dtype=np.float32)
        + 0.15 * coverage
    ) * 100

    result = pd.DataFrame({
        "O*NET-SOC Code": matrix.index.get_level_values(0),
        "Career": matrix.index.get_level_values(1),
        "Match Score": np.round(np.clip(fit_score, 0, 100), 1),
        "Vector Similarity": np.round(cosine * 100, 1),
        "Importance Score": np.round(selected_importance.to_numpy() * 100, 1),
        "Skill Coverage": np.round(coverage * 100, 1),
        "Skills Matched": matched_count.to_numpy(dtype=int),
    })

    return (
        result
        .sort_values(
            ["Match Score", "Skills Matched", "Vector Similarity"],
            ascending=[False, False, False],
            kind="stable",
        )
        .head(top_n)
        .reset_index(drop=True)
    )
