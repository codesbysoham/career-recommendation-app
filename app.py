import re

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Career Intelligence Dashboard", page_icon="💼", layout="wide")
st.title("💼 Career Intelligence Dashboard")
st.caption("ML-driven job-market analytics, career matching and skill-gap analysis.")

try:
    from CareerClassifier import (
        build_skill_master,
        canonical_skill,
        dashboard_metrics,
        filter_jobs,
        get_occupation_skill_gaps,
        get_skill_gap_summary,
        load_onet_data,
        top_counts,
        occupation_lookup,
    )
    from AnalyticsEngine import rank_careers
    from JobEngine import load_postings, load_skill_postings
except Exception as e:
    st.error("The recommendation engine could not start.")
    st.exception(e)
    st.stop()

skills_df, knowledge_df, occupations_df, software_df = load_onet_data()


@st.cache_data(show_spinner=False)
def get_base_skill_master():
    empty_postings = pd.DataFrame(columns=["title", "skills_desc"])
    return build_skill_master(skills_df, software_df, empty_postings)


skill_master = get_base_skill_master()


def _skill_key(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _library_skill_keys():
    keys = set()
    if "Element Name" in skills_df.columns:
        keys.update(_skill_key(canonical_skill(x)) for x in skills_df["Element Name"].dropna())
    if "Workplace Example" in software_df.columns:
        keys.update(_skill_key(canonical_skill(x)) for x in software_df["Workplace Example"].dropna())
    return keys


LIBRARY_SKILL_KEYS = _library_skill_keys()


def skill_coverage(selected_skills):
    rows = []
    for skill in selected_skills:
        canonical = canonical_skill(skill)
        meta = skill_master[skill_master["skill"].map(_skill_key).eq(_skill_key(canonical))]
        if meta.empty:
            rows.append({
                "skill": skill,
                "source": "Not found in O*NET library",
                "market_frequency": 0,
                "market_share": 0.0,
                "hot_technology": 0,
                "in_demand": 0,
                "evidence_status": "❌ No library evidence",
            })
            continue
        row = meta.iloc[0]
        freq = int(row.get("market_frequency", 0) or 0)
        if freq == 0:
            status = "⚠️ No market evidence"
        elif freq < 10:
            status = "⚠️ Limited market evidence"
        else:
            status = "✅ Market evidence available"
        rows.append({
            "skill": canonical,
            "source": row.get("source", "O*NET"),
            "market_frequency": freq,
            "market_share": float(row.get("market_share", 0) or 0),
            "hot_technology": int(row.get("hot_technology", 0) or 0),
            "in_demand": int(row.get("in_demand", 0) or 0),
            "evidence_status": status,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def safe_job_recommendations(selected_skills, top_n=10):
    selected_skills = tuple(dict.fromkeys(str(s).strip() for s in selected_skills if str(s).strip()))
    if not selected_skills:
        return pd.DataFrame()
    try:
        base = load_postings()
        skill_text = load_skill_postings()
        n = min(len(base), len(skill_text))
        base = base.iloc[:n].copy()
        skill_text = skill_text.iloc[:n].copy()
        scores = np.zeros(n, dtype=np.float32)
        matched = np.zeros(n, dtype=np.int16)
        patterns = []
        for skill in selected_skills:
            clean = re.sub(r"\s+", " ", skill.lower()).strip()
            if clean:
                patterns.append(re.compile(r"(?<![a-z0-9])" + re.escape(clean) + r"(?![a-z0-9])", re.I))
        if not patterns:
            return pd.DataFrame()
        chunk_size = 10000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            titles = skill_text.iloc[start:end]["title"].fillna("").astype(str)
            skills = skill_text.iloc[start:end]["skills_desc"].fillna("").astype(str)
            chunk_scores = np.zeros(end - start, dtype=np.float32)
            chunk_matched = np.zeros(end - start, dtype=np.int16)
            for pattern in patterns:
                title_hit = titles.str.contains(pattern, na=False).to_numpy()
                skill_hit = skills.str.contains(pattern, na=False).to_numpy()
                hit = title_hit | skill_hit
                chunk_scores += skill_hit.astype(np.float32)
                chunk_scores += (title_hit & ~skill_hit).astype(np.float32) * 0.7
                chunk_matched += hit.astype(np.int16)
            scores[start:end] = chunk_scores
            matched[start:end] = chunk_matched
        result = base.copy()
        result["Skills Matched"] = matched
        result["Match Score"] = (scores / max(len(patterns), 1) * 100).round(1)
        result = result[result["Skills Matched"] > 0]
        if result.empty:
            return pd.DataFrame()
        sort_views = result["views"] if "views" in result.columns else pd.Series(0, index=result.index)
        result["_sort_views"] = pd.to_numeric(sort_views, errors="coerce").fillna(0)
        result = result.sort_values(["Match Score", "Skills Matched", "_sort_views"], ascending=[False, False, False], kind="stable").head(top_n).reset_index(drop=True)
        return result.drop(columns=["_sort_views"], errors="ignore")
    except Exception:
        return pd.DataFrame()


page = st.sidebar.radio("Navigate", ["📊 Market Dashboard", "🎯 Career Matcher", "🧩 Skill Gap Analyzer", "🔎 Job Explorer"])

if page == "📊 Market Dashboard":
    st.header("Job Market Overview")
    st.info("The 123,849-posting market dataset is loaded only when you request the dashboard.")
    if st.button("Load Market Dashboard", type="primary"):
        try:
            with st.spinner("Loading job-market data..."):
                postings = load_postings()
            total, companies, titles, remote, median_salary = dashboard_metrics(postings)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Job Postings", f"{total:,}")
            c2.metric("Companies", f"{companies:,}")
            c3.metric("Unique Titles", f"{titles:,}")
            c4.metric("Remote Jobs", f"{remote:,}")
            c5.metric("Median Salary", f"${median_salary:,.0f}" if pd.notna(median_salary) else "N/A")
            st.divider()
            left, right = st.columns(2)
            with left:
                st.subheader("Top Job Titles")
                st.bar_chart(top_counts(postings["title"], 10), width="stretch")
            with right:
                st.subheader("Top Locations")
                st.bar_chart(top_counts(postings["location"], 10), width="stretch")
            left, right = st.columns(2)
            with left:
                st.subheader("Work Type")
                st.bar_chart(postings["formatted_work_type"].fillna("Unknown").value_counts().head(10), width="stretch")
            with right:
                st.subheader("Experience Level")
                st.bar_chart(postings["formatted_experience_level"].fillna("Unknown").value_counts().head(10), width="stretch")
            st.subheader("Top Unified Skills")
            with st.spinner("Calculating market skill demand..."):
                skill_postings = load_skill_postings()
                market_skills = build_skill_master(skills_df, software_df, skill_postings)
            top_skills = market_skills.head(20).set_index("skill")["market_frequency"]
            if top_skills.sum() > 0:
                st.bar_chart(top_skills, width="stretch")
            else:
                st.info("No job-market skill frequency was detected yet.")
        except Exception:
            st.error("The market dashboard could not be loaded. The underlying data is unavailable or incomplete.")

elif page == "🎯 Career Matcher":
    st.header("🎯 Career Matcher")
    st.write("Choose from one unified skill library containing O*NET skills, O*NET workplace technologies, and controlled job-market evidence.")
    st.info("Career and job rankings are produced from the ML/O*NET and job-market evidence. The application does not use an LLM to make predictions.")
    selected = st.multiselect("What skills do you have?", skill_master["skill"].tolist(), placeholder="Search Python, SQL, Excel, Power BI, Critical Thinking...")
    if selected:
        evidence = skill_coverage(selected)
        with st.expander("Skill evidence", expanded=True):
            st.caption("This table shows where each selected skill comes from and how much supporting evidence exists in the available job-posting dataset.")
            st.dataframe(evidence, width="stretch", hide_index=True)
            weak = evidence[evidence["market_frequency"] < 10]
            if not weak.empty:
                names = ", ".join(weak["skill"].astype(str).tolist())
                st.warning(f"Limited market evidence for: {names}. Results using these skills may be less reliable.")

        try:
            with st.spinner("Running the ML career matcher..."):
                matches = rank_careers(selected, skills_df, software_df=software_df, top_n=10)
        except Exception:
            matches = pd.DataFrame()
            st.error("The career matcher could not evaluate these skills safely. Try another combination; no prediction was generated.")

        st.subheader("Best-Matching Careers")
        if not matches.empty:
            st.caption("ML score combines unified O*NET skill + workplace-technology similarity, evidence strength and occupation-feature coverage.")
            st.dataframe(matches, width="stretch", hide_index=True)
            with st.expander("How the ML career model works"):
                st.markdown("**Pipeline:** selected skills → canonical skill mapping → unified O*NET skill + workplace-technology occupation vectors → cosine similarity + evidence strength + feature coverage → ranked occupations.")
                st.markdown("**Score weights:** 55% vector similarity + 30% matched-feature strength + 15% occupation-feature coverage.")
        else:
            unsupported = [s for s in selected if _skill_key(canonical_skill(s)) not in LIBRARY_SKILL_KEYS]
            if unsupported:
                st.warning("No reliable career match was generated because these selected skills are not represented in the occupation feature library: " + ", ".join(unsupported) + ".")
            else:
                st.info("The available O*NET occupation data does not contain enough matching evidence for these skills to produce a reliable career ranking. This is a data-coverage limitation, not a forced prediction.")

        with st.spinner("Finding matching jobs..."):
            job_matches = safe_job_recommendations(selected, top_n=10)
        st.subheader("Recommended Job Postings")
        if job_matches.empty:
            st.info("The available job-posting dataset does not contain enough evidence to recommend postings for these skills. Try adding a broader or better-represented skill.")
        else:
            st.caption("Job match score combines skill hits in structured job-skill text and job titles; structured skill hits receive higher weight.")
            display_cols = [c for c in ["title", "company_name", "location", "formatted_work_type", "formatted_experience_level", "remote_label", "normalized_salary", "Skills Matched", "Match Score", "job_posting_url"] if c in job_matches.columns]
            st.dataframe(job_matches[display_cols], width="stretch", hide_index=True)
    else:
        st.info("Select one or more skills to start.")

elif page == "🧩 Skill Gap Analyzer":
    st.header("🧩 Skill Gap Analyzer")
    st.caption("Recommendations combine O*NET occupation profiles, O*NET technology signals and skill co-occurrence from your job-posting data.")
    lookup = occupation_lookup(occupations_df)
    if lookup.empty:
        st.warning("Occupation data is unavailable, so the skill-gap analyzer cannot run.")
        st.stop()
    selected_idx = st.selectbox("Target career", range(len(lookup)), format_func=lambda i: lookup.iloc[i]["Title"])
    target = lookup.iloc[selected_idx]
    soc_code = target["O*NET-SOC Code"]
    occupation_title = target["Title"]
    st.caption(f"O*NET-SOC: **{soc_code}**")
    selected = st.multiselect("Your current skills", skill_master["skill"].tolist(), placeholder="Search and select skills...", key="gap_current_skills")

    if not selected:
        st.info("Select one or more current skills to build your evidence-based skill-gap profile.")
    else:
        evidence = skill_coverage(selected)
        with st.expander("Current skill evidence", expanded=False):
            st.dataframe(evidence, width="stretch", hide_index=True)
        try:
            with st.spinner("Loading job-market evidence..."):
                postings = load_skill_postings()
            gaps = get_occupation_skill_gaps(soc_code=soc_code, selected_skills=selected, skills_df=skills_df, software_df=software_df, postings_df=postings, top_n=12)
        except Exception:
            gaps = pd.DataFrame()
            st.error("The skill-gap analysis could not be completed with the available data.")
        st.success(f"Target profile loaded: **{occupation_title} ({soc_code})**")
        if gaps.empty:
            st.warning("The available O*NET/job-market data does not contain enough evidence to identify reliable skill gaps for this occupation.")
        else:
            st.subheader("🎯 Suggested Skills to Consider")
            high = gaps[gaps["priority_score"] >= 65]
            medium = gaps[gaps["priority_score"] < 65]
            display_gap_cols = ["skill", "skill_type", "priority_score", "importance", "software_hot", "software_demand", "cooccurrence", "reason"]
            if not high.empty:
                st.markdown("### 🔴 High Priority")
                st.dataframe(high[display_gap_cols], width="stretch", hide_index=True)
            if not medium.empty:
                st.markdown("### 🟠 Other Relevant Skills")
                st.dataframe(medium[display_gap_cols], width="stretch", hide_index=True)
            st.info(get_skill_gap_summary(gaps))
            with st.expander("How did the ML model rank these?"):
                st.markdown("**Priority score combines:** 40% O*NET importance, 25% Hot Technology, 20% In Demand, and 15% skill co-occurrence with your current skills in job postings. Skills already selected are removed before ranking.")
            st.subheader("📈 Your Current Profile")
            st.dataframe(evidence, width="stretch", hide_index=True)

else:
    st.header("🔎 Job Explorer")
    try:
        with st.spinner("Loading job-market data..."):
            postings = load_postings()
        col1, col2 = st.columns(2)
        with col1:
            title_search = st.text_input("Job title contains", placeholder="e.g. Data Scientist")
        with col2:
            location_search = st.text_input("Location contains", placeholder="e.g. New York")
        col1, col2, col3 = st.columns(3)
        with col1:
            work_types = ["All"] + sorted(postings["formatted_work_type"].dropna().astype(str).unique().tolist())
            work_type = st.selectbox("Work type", work_types)
        with col2:
            experiences = ["All"] + sorted(postings["formatted_experience_level"].dropna().astype(str).unique().tolist())
            experience = st.selectbox("Experience", experiences)
        with col3:
            remote = st.selectbox("Remote", ["All", "Remote", "Not Remote", "Unknown"])
        min_salary = st.number_input("Minimum normalized salary", min_value=0, value=0, step=5000)
        filtered = filter_jobs(postings, title=title_search or None, location=location_search or None, work_type=work_type, experience=experience, remote=remote, min_salary=min_salary if min_salary > 0 else None)
        st.metric("Matching jobs", f"{len(filtered):,}")
        display_cols = [c for c in ["title", "company_name", "location", "formatted_work_type", "formatted_experience_level", "remote_label", "normalized_salary", "views", "applies", "job_posting_url"] if c in filtered.columns]
        st.dataframe(filtered[display_cols].head(200), width="stretch", hide_index=True)
        st.caption("Showing up to 200 matching postings.")
    except Exception:
        st.error("The job explorer could not load the market dataset. No results were generated.")
