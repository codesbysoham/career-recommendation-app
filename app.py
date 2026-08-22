import pandas as pd
import streamlit as st

from CareerClassifier import (
    career_match,
    dashboard_metrics,
    filter_jobs,
    get_skill_gaps,
    get_skill_options,
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


# -----------------------------
# Load the cached backend once
# -----------------------------
try:
    (
        postings,
        tfidf,
        tfidf_matrix,
        skills_df,
        knowledge_df,
        occupations_df,
    ) = load_pipeline()
except Exception as e:
    st.error("The recommendation engine could not start.")
    st.exception(e)
    st.stop()


# -----------------------------
# Sidebar navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Market Dashboard",
        "🎯 Career Matcher",
        "🧩 Skill Gap Analyzer",
        "🔎 Job Explorer",
    ],
)

skill_options = get_skill_options(skills_df)


# ============================================================
# 1. MARKET DASHBOARD
# ============================================================
if page == "📊 Market Dashboard":

    st.header("Job Market Overview")
    st.write("Explore the structure of the job market represented by the dataset.")

    total, companies, titles, remote, median_salary = dashboard_metrics(postings)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Job Postings", f"{total:,}")
    c2.metric("Companies", f"{companies:,}")
    c3.metric("Unique Job Titles", f"{titles:,}")
    c4.metric("Remote Jobs", f"{remote:,}")
    c5.metric(
        "Median Salary",
        f"${median_salary:,.0f}" if pd.notna(median_salary) else "N/A",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Top Job Titles")
        title_counts = top_counts(postings["title"], 10)
        st.bar_chart(title_counts)

    with right:
        st.subheader("Top Locations")
        location_counts = top_counts(postings["location"], 10)
        st.bar_chart(location_counts)

    st.divider()

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

    st.subheader("Most Visible Job Titles")
    if "views" in postings.columns:
        views = (
            postings.groupby("title", dropna=False)["views"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(views)


# ============================================================
# 2. CAREER MATCHER
# ============================================================
elif page == "🎯 Career Matcher":

    st.header("🎯 Career Matcher")
    st.write(
        "Select skills from the actual O*NET Skills dataset and find "
        "occupations whose skill profiles best match them."
    )

    selected_skills = st.multiselect(
        "What skills do you have?",
        skill_options,
        placeholder="Search and select skills...",
    )

    if selected_skills:
        st.caption(f"{len(selected_skills)} skills selected")

        match = career_match(
            selected_skills,
            skills_df,
            top_n=10,
        )

        if not match.empty:
            st.subheader("Best-Matching Careers")
            st.dataframe(
                match,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Recommended Job Postings")

            recommendations = recommend_jobs(
                selected_skills,
                tfidf,
                tfidf_matrix,
                postings,
                top_n=10,
            )

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No matching occupations were found.")

    else:
        st.info("Select one or more skills to start.")


# ============================================================
# 3. SKILL GAP ANALYZER
# ============================================================
elif page == "🧩 Skill Gap Analyzer":

    st.header("🧩 Skill Gap Analyzer")
    st.write(
        "Choose a target occupation and compare its most important O*NET "
        "skills with the skills you already have."
    )

    occupation_titles = sorted(
        occupations_df["Title"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    occupation = st.selectbox(
        "Target career",
        occupation_titles,
    )

    selected_skills = st.multiselect(
        "Your current skills",
        skill_options,
        placeholder="Select your current skills...",
    )

    gaps = get_skill_gaps(
        occupation,
        selected_skills,
        skills_df,
        top_n=10,
    )

    if not gaps.empty:
        st.subheader(f"Top Skills for {occupation}")

        st.dataframe(
            gaps,
            use_container_width=True,
            hide_index=True,
        )

        matched = int(gaps["Selected"].sum())
        total = len(gaps)

        st.metric(
            "Top-skill coverage",
            f"{matched}/{total}",
        )

        missing = gaps.loc[~gaps["Selected"], "Skill"].tolist()

        if missing:
            st.subheader("Potential Skill Gaps")
            st.write(", ".join(missing))
        else:
            st.success("You selected all of the top skills shown.")


# ============================================================
# 4. JOB EXPLORER
# ============================================================
elif page == "🔎 Job Explorer":

    st.header("🔎 Job Explorer")
    st.write("Search and filter the underlying job-posting dataset.")

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
        if "formatted_work_type" in postings.columns:
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
        if "formatted_experience_level" in postings.columns:
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
