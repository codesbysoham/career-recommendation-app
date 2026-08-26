# Career Intelligence Dashboard

**ML + Job-Market Analytics | Python · Pandas · NumPy · Scikit-learn · Streamlit**

An end-to-end career intelligence application that combines **O*NET occupational data** with a **123K+ job-posting dataset** to analyze job-market trends, match user skill profiles to occupations, identify skill gaps, and explore relevant job postings.

> **Core principle:** the application is evidence-driven. It does not force a prediction when the underlying O*NET or job-market data does not provide enough support.

## Live App

[Career Intelligence Dashboard](https://codesbysoham-career-recommendation-app.streamlit.app/)

## What the project does

### 1. Unified skill intelligence

The application combines multiple O*NET sources into a common skill/technology layer and connects them with observed job-market signals.

For each skill or workplace technology, the system can track:

- O*NET source
- occupational relevance
- job-posting frequency
- Hot Technology signal
- In-Demand signal
- evidence coverage

This allows the dashboard to distinguish between **strong market evidence**, **limited evidence**, and **insufficient data coverage**.

### 2. ML-based career matching

A user's selected skills are represented in a unified occupation feature space containing O*NET general skills and workplace technologies.

The career-ranking engine uses:

- skill-vector representations
- **cosine similarity** between the user profile and occupation profiles
- skill-importance weighting
- skill-coverage scoring
- matched-skill counts

The final career score combines these signals into an interpretable ranking rather than relying on a black-box prediction.

### 3. Skill-gap analysis

For a selected target occupation, the system compares the user's current skills against occupational requirements and market signals to identify potential gaps and prioritize areas for development.

### 4. Job-market analytics

The dashboard provides exploratory analysis of the job-posting dataset, including:

- posting volume
- companies
- job titles
- locations
- remote-work availability
- experience levels
- work types
- salary information
- skill frequency

### 5. Job recommendation

Job descriptions are transformed into numerical representations using **TF-IDF**. Cosine similarity is then used to compare user/occupation-relevant text with job postings and surface relevant opportunities.

## System architecture

```text
                  ┌──────────────────────────┐
                  │       Data Sources       │
                  │                          │
                  │ O*NET Skills             │
                  │ O*NET Technologies       │
                  │ Occupation Profiles      │
                  │ 123K+ Job Postings       │
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Data Preparation         │
                  │                          │
                  │ Cleaning & normalization │
                  │ Skill canonicalization  │
                  │ Feature engineering      │
                  └────────────┬─────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
      ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
      │ Career ML   │   │ Skill Gap    │   │ Job Market  │
      │             │   │ Analytics    │   │ Analytics   │
      │ Cosine      │   │ Importance   │   │ EDA         │
      │ similarity  │   │ + demand     │   │ + TF-IDF    │
      │ + weighting │   │ + co-occurrence│  │ similarity │
      └──────┬──────┘   └──────┬───────┘   └──────┬──────┘
             └─────────────────┼─────────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │    Streamlit Dashboard  │
                  │                          │
                  │ Career Matching          │
                  │ Skill Gap Analysis       │
                  │ Job Recommendations      │
                  │ Market Analytics         │
                  └──────────────────────────┘
```

## Data sources

### O*NET

The project uses O*NET occupational information to obtain:

- general skills
- workplace technologies/software
- occupation identifiers and titles
- skill importance information
- technology demand indicators

O*NET provides the occupational structure used by the career-matching and skill-gap components.

### Job postings

The project uses a 123K+ job-posting dataset containing fields such as job title, company, description, location, salary, work type, experience level, remote status, and posting-level skill information.

The job data provides the empirical market layer used for demand and job-recommendation analysis.

## ML / analytics methodology

### Career matching

1. Normalize selected user skills.
2. Build occupation-level feature vectors from O*NET general skills.
3. Add O*NET workplace/software technology signals to the same feature space.
4. Construct a binary user skill vector.
5. Calculate **cosine similarity** between the user profile and occupation vectors.
6. Combine similarity with skill strength and coverage signals.
7. Return a ranked occupation list with interpretable component scores.

### Job matching

Job title and description text are transformed with **TF-IDF** using unigrams and bigrams. Cosine similarity is used to compare relevant text representations against job postings.

### Skill evidence

Market frequency and O*NET technology indicators are used as supporting evidence rather than as claims of absolute demand. If a selected skill has little or no representation in the available job data, the application reports the limitation.

## Handling sparse data

A key design decision is **not to fabricate a recommendation when the data cannot support one**.

For example:

- Strong O*NET + market evidence → career analysis can be produced.
- O*NET evidence but weak market representation → result is flagged as having limited market evidence.
- No meaningful occupation-level evidence → the application reports insufficient data coverage.

This makes the limitations of the underlying datasets visible to the user instead of hiding them behind a forced prediction.

## Project structure

```text
career-recommendation-app/
├── app.py                  # Streamlit application / dashboard
├── CareerClassifier.py     # Data loading, skill intelligence and core pipeline
├── AnalyticsEngine.py      # ML career-ranking engine
├── JobEngine.py            # Job-market loading and recommendation logic
├── requirements.txt        # Python dependencies
├── .gitignore
└── data/                   # O*NET reference files; job postings are downloaded when required
```

## Tech stack

**Programming:** Python

**Libraries / Frameworks:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, OpenPyXL

**ML / Analytics:** Feature Engineering, TF-IDF, Cosine Similarity, Vector Representations, Similarity Modeling, Weighted Scoring, Skill-Gap Analysis, Statistical/Data Analysis

**Data:** O*NET, job-posting data

## Running locally

```bash
git clone https://github.com/codesbysoham/career-recommendation-app.git
cd career-recommendation-app
pip install -r requirements.txt
streamlit run app.py
```

The application may require Kaggle credentials to download the job-posting dataset. Configure them through Streamlit Secrets or the supported Kaggle authentication mechanism rather than committing credentials to the repository.

## Limitations

- Career rankings depend on the coverage and quality of the O*NET occupation profiles.
- Market-frequency signals depend on the available job-posting sample and should not be interpreted as a complete representation of the global labor market.
- Sparse or missing skill evidence can limit the reliability of career rankings.
- Similarity scores represent analytical fit within the available feature space; they are not employment-probability predictions.

## Future improvements

- Formal offline evaluation using a labeled occupation/skill benchmark.
- Temporal analysis of changing skill demand.
- Improved skill synonym/entity resolution across data sources.
- Model calibration and sensitivity analysis for ranking weights.
- More rigorous validation of career-ranking performance.
