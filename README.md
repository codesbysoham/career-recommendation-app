# Career Intelligence Dashboard

**ML + Job-Market Analytics | Python · Pandas · NumPy · Scikit-learn · Streamlit**

An end-to-end ML and analytics project that combines O*NET occupational data with a 123K+ job-posting dataset. The dashboard lets a user explore the job market, see which careers fit a set of skills, identify skill gaps, and find relevant job postings.

The main goal of the project is to keep the recommendations tied to the data. If the available data does not provide enough evidence for a skill or occupation, the dashboard says so instead of forcing a result.

## Live app

[Career Intelligence Dashboard](https://codesbysoham-career-recommendation-app.streamlit.app/)

## What it does

### Career matching

Users select skills from a common skill library built from O*NET skills and workplace technologies. Those skills are mapped into an occupation-level feature space and used to rank careers.

The ranking uses:

- occupation skill vectors
- cosine similarity
- O*NET skill-importance weights
- skill coverage
- number of matched skills

The result is an interpretable ranking rather than a black-box classification.

### Skill-gap analysis

For a selected occupation, the dashboard compares the user's skills with the skills associated with that occupation. O*NET importance and available market evidence are used to help distinguish stronger gaps from weaker ones.

### Job-market analytics

The job-posting data is used to explore things such as:

- job titles and companies
- locations
- work type and remote status
- experience level
- salary information
- skill frequency
- posting volume

### Job recommendations

Job titles and descriptions are converted into TF-IDF representations. Cosine similarity is then used to find postings that are most relevant to the selected skills and career profile.

## How the data is put together

The project combines several O*NET files into a common skill/technology layer:

- **Skills** — general occupational skills
- **Software Skills** — workplace technologies and software
- **Knowledge** — knowledge areas associated with occupations
- **Occupation Data** — occupation codes and titles
- **Sample of Reported Titles** — occupation/title reference data

This O*NET layer is then connected to the job-posting dataset so that occupational information and observed market signals can be viewed together.

## ML / analytics pipeline

```text
O*NET data + job postings
          ↓
Cleaning and normalization
          ↓
Skill / technology mapping
          ↓
Occupation feature vectors
          ↓
┌─────────────────────────────────────┐
│ Career matching                     │
│ • cosine similarity                 │
│ • importance weighting              │
│ • skill coverage                    │
│ • matched-skill count               │
└─────────────────────────────────────┘
          ↓
Career ranking + skill-gap analysis

Job titles / descriptions
          ↓
TF-IDF vectorization
          ↓
Cosine similarity
          ↓
Relevant job postings
```

## Handling limited data

A useful part of the project is knowing when **not** to make a strong claim.

For example, a software skill may exist in the O*NET technology data but have little or no representation in the available job-posting sample. In that case, the dashboard reports limited or insufficient market evidence rather than pretending that the skill is highly demanded.

Similarly, an occupation is not ranked highly simply because one skill happens to match. The ranking considers the broader occupation profile and reports the components behind the score.

This is especially important for less common or poorly represented technologies, where the underlying dataset may simply not contain enough information to support a reliable market conclusion.

## Project structure

```text
career-recommendation-app/
├── app.py                  # Streamlit dashboard
├── CareerClassifier.py     # Data loading and career-matching pipeline
├── AnalyticsEngine.py      # Occupation ranking and ML scoring
├── JobEngine.py            # Job-market loading and job matching
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
└── data/
    ├── Knowledge.xlsx
    ├── Occupation Data.xlsx
    ├── Sample of Reported Titles.xlsx
    ├── Skills.xlsx
    └── Software Skills.xlsx
```

The large job-posting dataset is handled separately rather than being stored as a normal repository file.

## Tech stack

**Programming:** Python

**Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, OpenPyXL

**ML / Analytics:** Feature Engineering, TF-IDF, Cosine Similarity, Vector Representations, Weighted Scoring, Skill-Gap Analysis, Exploratory Data Analysis

**Data:** O*NET occupational data + job-posting data

## Running locally

```bash
git clone https://github.com/codesbysoham/career-recommendation-app.git
cd career-recommendation-app
pip install -r requirements.txt
streamlit run app.py
```

The job-posting data may require Kaggle authentication. Credentials should be supplied through Streamlit Secrets or the supported Kaggle authentication method and should never be committed to the repository.

## Limitations

- The career rankings depend on the coverage and quality of the O*NET occupation profiles.
- The job-posting dataset is a sample of the labour market, not a complete picture of global employment.
- Rare skills may have too little market evidence for a meaningful demand assessment.
- Similarity scores measure fit within the available feature space; they are not probabilities of getting a job.

## Future work

- Evaluate the career ranking against a labelled benchmark.
- Add time-based analysis of changing skill demand.
- Improve skill synonym and entity matching across datasets.
- Test the sensitivity of rankings to different scoring weights.
- Add more formal validation of the recommendation pipeline.
