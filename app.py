import pandas as pd
import streamlit as st

from CareerClassifier import (
    build_skill_master,
    career_match,
    dashboard_metrics,
    filter_jobs,
    get_skill_gaps,
    load_pipeline,
    recommend_jobs,
    top_counts,
)


st.set_page_config(
    page_title="Career Intelligence Dashboard",
    page_icon="💼",
    layout="wide",
)

st.title("💼 Career Intelligence Dashboard")
st.caption("Job-market analytics, career matching and skill-gap analysis.")

try:
    (
        postings,
        tfidf,
        tfidf_matrix,
        skills_df,
        knowledge_df,
        occupations_df,
        software_df,
    ) = load_pipeline()

    skill_master = build_skill_master(
        skills_df,
        software_df,
        postings,
    )
except Exception as e:
    st.error("The recommendation engine could not start.")
    st.exception(e)
    st.stop()


page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Market Dashboard",
        "🎯 Career Matcher",
        "🧩 Skill Gap Analyzer",
        "🔎 Job Explorer",
    ],
)

# One unified searchable skill list.
# Sort by market evidence first, then O*NET technology signals.
skill_options = skill_master["skill"].tolist()


# ============================================================
# MARKET DASHBOARD
# ============================================================
if page == "📊 Market Dashboard":
    st.header("Job Market Overview")

    total, companies, titles, remote, median_salary = dashboard_metrics(postings)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Job Postings", f"{total:,}")
    c2.metric("Companies", f"{companies:,}")
    c3.metric("Unique Titles", f"{titles:,}")
    c4.metric("Remote Jobs", f"{remote:,}")
    c5.metric(
        "Median Salary",
        f"${median_salary:,.0f}" if pd.notna(median_salary) else "N/A",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Top Job Titles")
        st.bar_chart(top_counts(postings["title"], 10))

    with right:
        st.subheader("Top Locations")
        st.bar_chart(top_counts(postings["location"], 10))

    left, right = st.columns(2)

    with left:
        st.subheader("Work Type")
        if "formatted_work_type" in postings:
            st.bar_chart(
                postings["formatted_work_type"]
                .fillna("Unknown")
                .value_counts()
                .head(10)
            )

    with right:
        st.subheader("Experience Level")
        if "formatted_experience_level" in postings:
            st.bar_chart(
                postings["formatted_experience_level"]
                .fillna("Unknown")
                .value_counts()
                .head(10)
            )

    st.subheader("Top Unified Skills")
    top_skills = skill_master.head(20).set_index("skill")["market_frequency"]
    st.bar_chart(top_skills)


# ============================================================
# CAREER MATCHER
# ============================================================
elif page == "🎯 Career Matcher":
    st.header("🎯 Career Matcher")
    st.write(
        "Select from one unified skill library. It combines O*NET skills, "
        "O*NET workplace software examples, and validated skill evidence "
        "from the job-posting data."
    )

    selected = st.multiselect(
        "What skills do you have?",
        skill_options,
        placeholder="Search Python, SQL, Excel, Power BI, Critical Thinking...",
    )

    if selected:
        meta = skill_master[skill_master["skill"].isin(selected)].copy()

        st.caption(f"{len(selected)} skills selected")

        with st.expander("Skill evidence"):
            st.dataframe(
                meta[
                    [
                        "skill",
                        "source",
                        "market_frequency",
                        "hot_technology",
                        "in_demand",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        # O*NET career matching where selected skills map to O*NET's
        # general skill dimensions.
        matches = career_match(
            selected,
            skills_df,
            skill_master,
            top_n=10,
        )

        if matches.empty:
            st.info(
                "These skills are primarily technical/software skills. "
                "We can still match you to actual job postings below; "
                "O*NET occupation matching requires an O*NET general-skill mapping."
            )
        else:
            st.subheader("Best-Matching Careers")
            st.dataframe(
                matches,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Recommended Job Postings")
        jobs = recommend_jobs(
            selected,
            tfidf,
            tfidf_matrix,
            postings,
            top_n=10,
        )
        st.dataframe(
            jobs,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Select one or more skills to start.")


# ============================================================
# SKILL GAP ANALYZER
# ============================================================
elif page == "🧩 Skill Gap Analyzer":
    st.header("🧩 Skill Gap Analyzer")

    occupation_titles = sorted(
        occupations_df["Title"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    occupation = st.selectbox("Target career", occupation_titles)

    selected = st.multiselect(
        "Your current skills",
        skill_options,
        placeholder="Search your skills...",
    )

    gaps = get_skill_gaps(
        occupation,
        selected,
        skills_df,
        top_n=10,
    )

    if gaps.empty:
        st.warning("No O*NET skill profile was found for this occupation.")
    else:
        st.subheader(f"Top Skills for {occupation}")
        st.dataframe(
            gaps,
            use_container_width=True,
            hide_index=True,
        )

        matched = int(gaps["Selected"].sum())
        st.metric("Top-skill coverage", f"{matched}/{len(gaps)}")

        missing = gaps.loc[~gaps["Selected"], "Skill"].tolist()
        if missing:
            st.subheader("Potential Skill Gaps")
            st.write(", ".join(missing))


# ============================================================
# JOB EXPLORER
# ============================================================
else:
    st.header("🔎 Job Explorer")

    col1, col2 = st.columns(2)
    with col1:
        title_search = st.text_input(
            "Job title contains",
            placeholder="e.g. Data Scientist",
        )
    with col2:
        location_search = st.text_input(
            "Location contains",
            placeholder="e.g. New York",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        work_types = ["All"]
        if "formatted_work_type" in postings:
            work_types += sorted(
                postings["formatted_work_type"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        work_type = st.selectbox("Work type", work_types)

    with col2:
        experiences = ["All"]
        if "formatted_experience_level" in postings:
            experiences += sorted(
                postings["formatted_experience_level"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        experience = st.selectbox("Experience", experiences)

    with col3:
        remote = st.selectbox(
            "Remote",
            ["All", "Remote", "Not Remote", "Unknown"],
        )

    min_salary = st.number_input(
        "Minimum normalized salary",
        min_value=0,
        value=0,
        step=5000,
    )

    filtered = filter_jobs(
        postings,
        title=title_search or None,
        location=location_search or None,
        work_type=work_type,
        experience=experience,
        remote=remote,
        min_salary=min_salary if min_salary > 0 else None,
    )

    st.metric("Matching jobs", f"{len(filtered):,}")

    display_cols = [
        c for c in [
            "title",
            "company_name",
            "location",
            "formatted_work_type",
            "formatted_experience_level",
            "remote_label",
            "normalized_salary",
            "views",
            "applies",
            "job_posting_url",
        ]
        if c in filtered.columns
    ]

    st.dataframe(
        filtered[display_cols].head(200),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Showing up to 200 matching postings.")
