import streamlit as st

from CareerClassifier import load_pipeline, recommend_jobs


st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="💼",
    layout="wide",
)

st.title("Career Recommendation System")
st.write("Find relevant job postings based on your skills.")


# The complete ML pipeline is cached in CareerClassifier.py.
try:
    postings_df, tfidf, tfidf_matrix = load_pipeline()
except Exception as e:
    st.error("The recommendation engine could not start.")
    st.exception(e)
    st.stop()


st.success(f"Loaded {len(postings_df):,} job postings.")


with st.expander("View Job Postings Data"):
    st.dataframe(
        postings_df.head(20),
        use_container_width=True,
    )


user_skills = st.multiselect(
    "Select your skills:",
    [
        "Python",
        "SQL",
        "Machine Learning",
        "Excel",
        "Data Visualization",
        "Statistics",
        "Data Analysis",
        "Power BI",
        "R",
        "Deep Learning",
    ],
)


if st.button("Get Recommendations", type="primary"):
    if not user_skills:
        st.warning("Please select at least one skill.")
    else:
        recommendations = recommend_jobs(
            user_skills=user_skills,
            tfidf=tfidf,
            tfidf_matrix=tfidf_matrix,
            postings_df=postings_df,
            top_n=5,
        )

        st.subheader("Top Recommendations")
        st.dataframe(
            recommendations,
            use_container_width=True,
        )
