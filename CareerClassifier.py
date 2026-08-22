#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Core libraries
import os, re, warnings
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML + Stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150})


# In[2]:


DATA_PATHS = {
    "skills":       r"C:\Users\BIT\Downloads\Skills.xlsx",
    "knowledge":    r"C:\Users\BIT\Downloads\Knowledge.xlsx",
    "occupations":  r"C:\Users\BIT\Downloads\Occupation Data.xlsx",
    "postings":     r"C:\Users\BIT\Downloads\jobpostings.csv\postings.csv",
}

skills_df = pd.read_excel(DATA_PATHS["skills"])
knowledge_df = pd.read_excel(DATA_PATHS["knowledge"])
occupations_df = pd.read_excel(DATA_PATHS["occupations"])
postings_df = pd.read_csv(DATA_PATHS["postings"], low_memory=False)


# In[3]:


print("Skills head:\n", skills_df.head())


# In[4]:


print("\nKnowledge head:\n", knowledge_df.head())


# In[5]:


print("\nOccupations head:\n", occupations_df.head())


# In[6]:


print("\nPostings head:\n", postings_df.head())


# In[7]:


print("\nDescriptive statistics for postings:")
print(postings_df.describe(include="all"))


# In[8]:


# Posting count by location
plt.figure(figsize=(10,5))
postings_df["location"].value_counts().head(15).plot(kind="bar", color="orange")
plt.title("Top 15 Locations by Job Postings")
plt.ylabel("Count")
plt.show()


# In[9]:


# Example: TF-IDF on job descriptions
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
tfidf_matrix = tfidf.fit_transform(postings_df["description"].fillna(""))

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Flexible column detection
title_col = next((c for c in postings_df.columns if "title" in c), None)
desc_col  = next((c for c in postings_df.columns if "descrip" in c), None)
sal_col   = next((c for c in postings_df.columns if "max_sal" in c or "salary" in c), None)
loc_col   = next((c for c in postings_df.columns if "location" in c), None)

rename = {}
if title_col: rename[title_col] = "job_title"
if desc_col:  rename[desc_col]  = "description"
if sal_col:   rename[sal_col]   = "max_salary"
if loc_col:   rename[loc_col]   = "location"

postings_df.rename(columns=rename, inplace=True)

# Ensure defaults if missing
if "job_title" not in postings_df.columns:
    postings_df["job_title"] = "Unknown"
if "description" not in postings_df.columns:
    postings_df["description"] = ""
if "max_salary" not in postings_df.columns:
    postings_df["max_salary"] = np.nan


# Market signals
market = postings_df.groupby("job_title").agg(
    posting_count=("job_title","count"),
    avg_salary=("max_salary","mean")
).reset_index()

print("\nMarket signals head:\n", market.head())


# In[10]:


def recommend_jobs(user_skills, tfidf, postings_df, top_n=5):
    user_vec = tfidf.transform([" ".join(user_skills)])
    sims = cosine_similarity(user_vec, tfidf_matrix)[0]
    postings_df["similarity"] = sims
    return postings_df.sort_values("similarity", ascending=False).head(top_n)[["job_title","similarity"]]

print("\nSample Recommendations:")
print(recommend_jobs(["python","data analysis","machine learning"], tfidf, postings_df))


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
postings_df = pd.read_csv(r"C:\Users\BIT\Downloads\jobpostings.csv\postings.csv", low_memory=False)
print(postings_df.head())


# Assuming postings_df is already loaded and has a 'description' column
docs = postings_df["description"].fillna("").tolist()

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)

tfidf_matrix = tfidf.fit_transform(docs)
print("TF-IDF matrix shape:", tfidf_matrix.shape)


# In[ ]:




