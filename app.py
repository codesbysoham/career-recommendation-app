#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd

# Import your pipeline classes from CareerClassifier.py
from CareerClassifier import RecommendationEngine, DataLoader, FeatureEngineer, HybridModel, Visualizer

st.title("Career Recommendation System")

# Upload dataset
uploaded_file = st.file_uploader("Upload job postings CSV", type="csv")
if uploaded_file is not None:
    postings_df = pd.read_csv(uploaded_file)

    # Expandable table to preview data
    with st.expander("View Job Postings Data"):
        st.dataframe(postings_df.head(20))

    # Skill selection widget
    user_skills = st.multiselect(
        "Select your skills:",
        ["Python", "SQL", "Machine Learning", "Excel", "Data Visualization"]
    )

    if st.button("Get Recommendations"):
        # Initialize pipeline
        engine = RecommendationEngine()
        feature_engineer = FeatureEngineer()
        hybrid_model = HybridModel()

        # Example: preprocess + recommend
        features = feature_engineer.transform(postings_df)
        recommendations = engine.recommend(user_skills, features, hybrid_model)

        st.write("Top Recommendations:")
        st.dataframe(recommendations)


# In[ ]:





# In[ ]:




