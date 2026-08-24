import json
import os
import re

import pandas as pd
import streamlit as st

MODEL = "gemini-2.5-flash"


def _api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("gemini_api_key")
    except Exception:
        return None


def available():
    return bool(_api_key())


def _client():
    from google import genai
    return genai.Client(api_key=_api_key())


def _clean_skill(value):
    return re.sub(r"\s+", " ", str(value).strip())


def _candidate_table(skill_master, limit=160):
    cols = [c for c in ["skill", "market_frequency", "hot_technology", "in_demand", "source"] if c in skill_master.columns]
    candidates = skill_master[cols].copy()
    if candidates.empty:
        return []
    sort_cols = [c for c in ["market_frequency", "hot_technology", "in_demand"] if c in candidates.columns]
    if sort_cols:
        candidates = candidates.sort_values(sort_cols, ascending=False, kind="stable")
    return candidates.head(limit).to_dict("records")


def _response_text(response):
    return getattr(response, "text", "") or ""


def _generate(prompt):
    try:
        response = _client().models.generate_content(
            model=MODEL,
            contents=prompt,
        )
        return _response_text(response)
    except Exception as exc:
        st.warning(f"AI advisor temporarily unavailable: {exc}")
        return ""


@st.cache_data(show_spinner=False)
def suggest_skills(selected_skills, target_career="", skill_master=None, limit=8):
    """Suggest only skills already present in the controlled skill library."""
    if not available() or skill_master is None or skill_master.empty:
        return pd.DataFrame()

    selected = [_clean_skill(s) for s in selected_skills if _clean_skill(s)]
    selected_keys = {s.casefold() for s in selected}
    candidates = [
        x for x in _candidate_table(skill_master)
        if _clean_skill(x.get("skill", "")).casefold() not in selected_keys
    ]
    allowed = {_clean_skill(x["skill"]).casefold(): _clean_skill(x["skill"]) for x in candidates}

    prompt = f"""You are the career-intelligence ranking layer for a career application.
Target career: {target_career or 'Not specified'}
Current skills: {', '.join(selected) or 'None'}

Choose up to {limit} additional skills from the supplied candidate library that
would be most useful. Prefer complementary skills and skills supported by the
market/O*NET signals. NEVER invent a skill and NEVER return a skill outside the
candidate library.

Candidate library:
{json.dumps(candidates, ensure_ascii=False)}

Return one line per recommendation exactly:
SKILL | short reason
No heading, no bullets, no extra text."""

    text = _generate(prompt)
    rows = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        raw_skill, reason = line.split("|", 1)
        key = _clean_skill(raw_skill).casefold()
        if key in allowed:
            rows.append({"skill": allowed[key], "reason": _clean_skill(reason)})
        if len(rows) >= limit:
            break

    return pd.DataFrame(rows).drop_duplicates("skill").reset_index(drop=True) if rows else pd.DataFrame(columns=["skill", "reason"])


@st.cache_data(show_spinner=False)
def explain_skill_gaps(target_career, current_skills, gaps):
    if not available() or gaps is None or gaps.empty:
        return ""

    evidence_cols = [c for c in [
        "skill", "skill_type", "priority_score", "importance",
        "software_hot", "software_demand", "cooccurrence", "reason"
    ] if c in gaps.columns]
    evidence = gaps[evidence_cols].head(12).to_dict("records")

    prompt = f"""You are explaining a data-backed career skill-gap analysis.
Target career: {target_career}
Current skills: {', '.join(current_skills)}

Treat the following ranking-engine evidence as authoritative. Do not invent
market facts or change the ranking:
{json.dumps(evidence, ensure_ascii=False)}

Write a concise career plan with:
1. What the user already has going for them.
2. The 3 highest-priority gaps and why they matter.
3. A practical learning order.
4. One sentence on what not to prioritize yet.
Keep it under 250 words."""

    return _generate(prompt).strip()
