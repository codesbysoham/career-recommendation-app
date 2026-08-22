import streamlit as st
import pandas as pd

from CareerClassifier import load_pipeline, recommend_jobs

st.set_page_config(page_title="Career Recommendation System", layout="wide")

st.title("Career Recommendation System")
st.write("Find relevant job postings based on your skills.")

@st.cache_resource(show_spinner="Loading career recommendation engine...")
def get_pipeline():
    return load_pipeline()

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error("The recommendation engine could not start.")
    st.exception(e)
    st.stop()

postings_df = pipeline["postings_df"]

st.success(f"Loaded {len(postings_df):,} job postings.")

with st.expander("View Job Postings Data"):
    st.dataframe(postings_df.head(20), use_container_width=True)

user_skills = st.multiselect(
    "Select your skills:",
    ["Python", "SQL", "Machine Learning", "Excel", "Data Visualization",
     "Statistics", "Data Analysis", "Power BI", "R", "Deep Learning"]
)

if st.button("Get Recommendations", type="primary"):
    if not user_skills:
        st.warning("Please select at least one skill.")
    else:
        recommendations = recommend_jobs(
            user_skills=user_skills,
            tfidf=pipeline["tfidf"],
            tfidf_matrix=pipeline["tfidf_matrix"],
            postings_df=postings_df,
            top_n=5,
        )

        st.subheader("Top Recommendations")
        st.dataframe(recommendations, use_container_width=True)
