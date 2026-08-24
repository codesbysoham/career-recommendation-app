import json
import os
import re

import pandas as pd
import streamlit as st


# Stable production model. Override with GEMINI_MODEL if needed.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
FALLBACK_MODEL = "gemini-3.6-flash"


def _api_key():
    """Read the Gemini key from Streamlit Cloud secrets or local env vars."""
    try:
        for name in ("gemini_api_key", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            key = st.secrets.get(name)
            if key:
                return str(key).strip()
    except Exception:
        pass
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def available():
    return bool(_api_key())


def _generate(prompt, *, structured_schema=None):
    """Call Gemini safely; AI failures never take down the ML application."""
    key = _api_key()
    if not key:
        return ""

    try:
        from google import genai

        client = genai.Client(api_key=key)
        config = None
        if structured_schema is not None:
            config = {
                "response_mime_type": "application/json",
                "response_schema": structured_schema,
            }

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )
        except Exception as first_exc:
            # Retry only for model/endpoint availability errors. Do not make a
            # second request for authentication, quota, or network failures.
            message = str(first_exc).lower()
            model_unavailable = any(token in message for token in (
                "not_found", "not found", "404", "model is not available",
                "model not found", "unsupported model",
            ))
            if MODEL == FALLBACK_MODEL or not model_unavailable:
                raise first_exc
            response = client.models.generate_content(
                model=FALLBACK_MODEL,
                contents=prompt,
                config=config,
            )

        return (getattr(response, "text", "") or "").strip()
    except Exception:
        st.warning("Gemini is temporarily unavailable. The ML/analytics engine is still working.")
        return ""


def _clean_skill(value):
    return re.sub(r"\s+", " ", str(value).strip())


def _candidate_table(skill_master, limit=160):
    cols = [
        c for c in ["skill", "market_frequency", "hot_technology", "in_demand", "source"]
        if c in skill_master.columns
    ]
    candidates = skill_master[cols].copy()
    if candidates.empty:
        return []
    sort_cols = [c for c in ["market_frequency", "hot_technology", "in_demand"] if c in candidates.columns]
    if sort_cols:
        candidates = candidates.sort_values(sort_cols, ascending=False, kind="stable")
    return candidates.head(limit).to_dict("records")


def suggest_skills(selected_skills, target_career="", skill_master=None, limit=8):
    """Ask Gemini to choose only skills already present in the controlled library."""
    if not available() or skill_master is None or skill_master.empty:
        return pd.DataFrame()

    selected = [_clean_skill(s) for s in selected_skills if _clean_skill(s)]
    selected_keys = {s.casefold() for s in selected}
    candidates = [
        x for x in _candidate_table(skill_master)
        if _clean_skill(x.get("skill", "")).casefold() not in selected_keys
    ]
    allowed = {
        _clean_skill(x["skill"]).casefold(): _clean_skill(x["skill"])
        for x in candidates
    }

    prompt = f"""You are the optional AI interpretation layer for a career analytics application.
Target career: {target_career or 'Not specified'}
Current skills: {', '.join(selected) or 'None'}

Choose up to {limit} additional skills from the supplied candidate library.
Prefer complementary skills and skills supported by the supplied market/O*NET
signals. Do not invent skills. Every skill must exactly match one candidate.

Candidate library (JSON):
{json.dumps(candidates, ensure_ascii=False)}
"""

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "skill": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["skill", "reason"],
        },
    }

    text = _generate(prompt, structured_schema=schema)
    if not text:
        return pd.DataFrame(columns=["skill", "reason"])

    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        payload = []

    rows = []
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            key = _clean_skill(item.get("skill", "")).casefold()
            reason = _clean_skill(item.get("reason", ""))
            if key in allowed:
                rows.append({"skill": allowed[key], "reason": reason})
            if len(rows) >= limit:
                break

    return (
        pd.DataFrame(rows, columns=["skill", "reason"])
        .drop_duplicates("skill")
        .reset_index(drop=True)
    )


def explain_skill_gaps(target_career, current_skills, gaps):
    """Turn evidence produced by the ML/O*NET layer into practical advice."""
    if not available() or gaps is None or gaps.empty:
        return ""

    evidence_cols = [
        c for c in [
            "skill", "skill_type", "priority_score", "importance",
            "software_hot", "software_demand", "cooccurrence", "reason"
        ] if c in gaps.columns
    ]
    evidence = gaps[evidence_cols].head(12).to_dict("records")

    prompt = f"""You are explaining a data-backed career skill-gap analysis.
Target career: {target_career}
Current skills: {', '.join(current_skills)}

The ranking engine produced the evidence below. Treat it as authoritative.
Do not invent additional market facts or change the ranking.
{json.dumps(evidence, ensure_ascii=False)}

Write a concise career plan with:
1. What the user already has going for them.
2. The 3 highest-priority gaps and why they matter.
3. A practical learning order.
4. One sentence on what not to prioritize yet.
Keep it under 250 words."""

    return _generate(prompt)
