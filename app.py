import pandas as pd
import streamlit as st

st.set_page_config(page_title="Career Intelligence Dashboard", page_icon="💼", layout="wide")
st.title("💼 Career Intelligence Dashboard")
st.caption("Job-market analytics, career matching and skill-gap analysis.")

try:
    from CareerClassifier import (
        build_skill_master,
        career_match,
        dashboard_metrics,
        filter_jobs,
        get_occupation_skill_gaps,
        get_skill_gap_summary,
        load_onet_data,
        recommend_jobs,
        top_counts,
        occupation_lookup,
    )
    from JobEngine import load_job_engine, load_postings, load_skill_postings
except Exception as e:
    st.error("The recommendation engine could not start.")
    st.exception(e)
    st.stop()

# Startup loads only the small O*NET files. Large job text is lazy-loaded.
skills_df, knowledge_df, occupations_df, software_df = load_onet_data()

@st.cache_data(show_spinner=False)
def get_base_skill_master():
    empty_postings = pd.DataFrame(columns=["title", "skills_desc"])
    return build_skill_master(skills_df, software_df, empty_postings)

skill_master = get_base_skill_master()

page = st.sidebar.radio(
    "Navigate",
    ["📊 Market Dashboard", "🎯 Career Matcher", "🧩 Skill Gap Analyzer", "🔎 Job Explorer"],
)

if page == "📊 Market Dashboard":
    st.header("Job Market Overview")
    st.info("The 123,849-posting market dataset is loaded only when you request the dashboard.")
    if st.button("Load Market Dashboard", type="primary"):
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

elif page == "🎯 Career Matcher":
    st.header("🎯 Career Matcher")
    st.write("Choose from one unified skill library containing O*NET skills, O*NET workplace technologies, and controlled job-market evidence.")
    selected = st.multiselect("What skills do you have?", skill_master["skill"].tolist(), placeholder="Search Python, SQL, Excel, Power BI, Critical Thinking...")

    if selected:
        meta = skill_master[skill_master["skill"].isin(selected)]
        with st.expander("Skill evidence"):
            evidence_cols = [c for c in ["skill", "source", "market_frequency", "hot_technology", "in_demand"] if c in meta.columns]
            st.dataframe(meta[evidence_cols], width="stretch", hide_index=True)

        matches = career_match(selected, skills_df, top_n=10)
        if not matches.empty:
            st.subheader("Best-Matching Careers")
            st.dataframe(matches, width="stretch", hide_index=True)
        else:
            st.info("No O*NET occupation profile matched the selected skills. Use the job recommendations below for direct market matching.")

        with st.spinner("Loading job-market recommendations..."):
            postings, tfidf, tfidf_matrix = load_job_engine()
        st.subheader("Recommended Job Postings")
        st.dataframe(recommend_jobs(selected, tfidf, tfidf_matrix, postings, top_n=10), width="stretch", hide_index=True)
    else:
        st.info("Select one or more skills to start.")

elif page == "🧩 Skill Gap Analyzer":
    st.header("🧩 Skill Gap Analyzer")
    st.caption("Recommendations combine O*NET occupation profiles, O*NET technology signals and skill co-occurrence from your job-posting data.")
    lookup = occupation_lookup(occupations_df)
    selected_idx = st.selectbox("Target career", range(len(lookup)), format_func=lambda i: lookup.iloc[i]["Title"])
    target = lookup.iloc[selected_idx]
    soc_code = target["O*NET-SOC Code"]
    occupation_title = target["Title"]
    st.caption(f"O*NET-SOC: **{soc_code}**")

    selected = st.multiselect("Your current skills", skill_master["skill"].tolist(), placeholder="Search and select skills...")
    if not selected:
        st.info("Select one or more skills to get personalized recommendations.")
    else:
        with st.spinner("Loading job-market evidence..."):
            postings = load_skill_postings()
        gaps = get_occupation_skill_gaps(
            soc_code=soc_code,
            selected_skills=selected,
            skills_df=skills_df,
            software_df=software_df,
            postings_df=postings,
            top_n=12,
        )
        st.success(f"Target profile loaded: **{occupation_title} ({soc_code})**")
        if gaps.empty:
            st.warning("No skill recommendations were found for this occupation.")
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
            with st.expander("How did the model rank these?"):
                st.markdown("**Priority score combines:** O*NET importance, Hot Technology, In Demand, and skill co-occurrence in job postings. Skills already selected are removed before ranking.")
            st.subheader("📈 Your Current Profile")
            selected_df = skill_master[skill_master["skill"].isin(selected)].copy()
            profile_cols = [c for c in ["skill", "source", "market_frequency", "hot_technology", "in_demand"] if c in selected_df.columns]
            st.dataframe(selected_df[profile_cols], width="stretch", hide_index=True)

else:
    st.header("🔎 Job Explorer")
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
