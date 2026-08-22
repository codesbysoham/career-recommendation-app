# career-recommendation-app
Career Recommendation System built with **Streamlit** and **machine learning**.  
It helps users match their skills to job postings and explore career paths interactively.

## 🚀 Features
- Upload job postings dataset (CSV/Excel)
- Preview and explore data in the app
- Select your skills and get tailored job recommendations
- Hybrid recommendation engine using TF‑IDF + clustering
- Visualizations with Matplotlib/Seaborn

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit‑learn
- Matplotlib
- Seaborn
- Streamlit
- Openpyxl (Excel support)

## 📊 Datasets
We combined multiple sources to build a robust recommendation pipeline:

1. **Kaggle Job Postings Dataset**  
   - Source: Public Kaggle datasets containing job titles, descriptions, companies, salaries, and locations.  
   - Fields: `job_id`, `job_title`, `company_name`, `job_description`, `salary`, `location`, `zip_code`, `fips`, etc.  
   - Purpose: Provides the universe of available jobs to match against user skills.

2. **O*NET Skills Dataset**  
   - Source: Occupational Information Network (O*NET) database curated by the U.S. Department of Labor.  
   - Fields: `skill_name`, `category`, `importance`, `level`.  
   - Purpose: Standardized taxonomy of skills and competencies used to map user input to job requirements.

3. **User Input Data**  
   - Collected interactively in the Streamlit app.  
   - Fields: Skills selected by the user.  
   - Purpose: Forms the basis of personalized recommendations.

## 🧠 Algorithms & Models
The recommendation pipeline combines several machine learning techniques:

- **TF‑IDF Vectorization**  
  Converts job descriptions into numerical feature vectors based on term frequency–inverse document frequency.

- **KMeans Clustering**  
  Groups similar job postings together to identify career clusters and patterns.

- **Truncated SVD (Latent Semantic Analysis)**  
  Reduces dimensionality of TF‑IDF vectors for faster computation and better visualization.

- **Hybrid Recommendation Engine**  
  Combines content‑based filtering (skills matching via O*NET taxonomy) with clustering similarity scores to generate ranked recommendations.

## 📦 Installation
Clone the repository and install dependencies
