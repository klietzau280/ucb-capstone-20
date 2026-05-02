# Bylaw Violation Classification — Capstone Project

## Research Question

Can machine learning techniques be used to classify and predict bylaw violations in residential communities, enabling proactive enforcement and more efficient compliance management?

## Dataset

**Source:** [Austin Code Complaint Cases — City of Austin Open Data Portal](https://data.austintexas.gov/Public-Safety/Austin-Code-Complaint-Cases/6wtj-zbtb)

~83,000 code enforcement complaint cases filed with Austin Code Enforcement. Key fields:

| Field | Description |
|---|---|
| `description` | Complaint/violation type (e.g., Property Abatement, Land Use Violation) |
| `priority` | Case priority level (1 = highest) |
| `status` | Open or Closed |
| `last_update` | Resolution outcome (basis for target variable) |
| `opened_date` / `closed_date` | Filing and closure dates |
| `zip_code` | Property zip code |
| `latitude` / `longitude` | Property coordinates |
| `repeatoffenderrelated` | Whether property is flagged as repeat offender |
| `reportedby` | How the complaint was filed |

## Target Variable

Binary classification derived from `last_update`:

- **1 — Violation Upheld:** Complaint resulted in an abatement or confirmed violation
- **0 — No Violation:** Inspection found no violation

Administrative closures (duplicates, withdrawn, supervisor review) are excluded from modeling.

## Techniques Used

- **Data Cleaning:** Missing value imputation, duplicate removal, date parsing, outlier capping (IQR method)
- **Feature Engineering:** Temporal features (month, day of week, year), days-to-close, boolean flags for repeat offenders and short-term rentals
- **NLP:** TF-IDF vectorization (unigrams + bigrams) on complaint description text
- **Visualization:** Matplotlib and Seaborn — bar charts, heatmaps, geographic scatter, time series
- **Baseline Model:** Logistic Regression with balanced class weights

## Results

**EDA Key Findings:**
- **Most common complaint type:** Property Abatement (36,334 cases), followed by Structure Condition Violations (21,966) and Land Use Violations (21,598)
- **Highest violation confirmation rate:** Property Abatement at 50.1%, Structure Condition at 42.4%, Land Use at 39.7%
- **Geographic hotspots:** ZIP codes 78758, 78741, 78745, 78704, and 78702 have the highest complaint volumes
- **Temporal patterns:** Complaints peak Monday–Wednesday and drop sharply on weekends, suggesting enforcement is primarily business-hours driven
- **Repeat offenders:** 58.9% violation rate vs. 43.5% for first-time complaints — a meaningful signal for the model

**Target variable:** 60,672 cases with a defined outcome — 55.3% No Violation, 44.7% Violation Upheld (well-balanced for classification)

**Baseline Model Performance (Logistic Regression):**

| Metric | Score |
|---|---|
| Accuracy | 78.9% |
| F1-Score | 0.737 |
| ROC-AUC | 0.834 |

The F1-Score is the primary evaluation metric because both false negatives (missing real violations) and false positives (wasting inspector time) carry meaningful operational costs, and F1 balances these. A ROC-AUC of 0.834 indicates the model has strong discriminative ability as a baseline.

## Notebook

[`notebooks/eda.ipynb`](notebooks/eda.ipynb)

## Project Structure

```
capstone/
├── data/
│   └── austin_code_complaints.csv   # downloaded on first notebook run
├── notebooks/
│   └── eda.ipynb
└── README.md
```

## Next Steps (Module 24)

- SVM with RBF kernel, Random Forest, and Gradient Boosting classifiers
- Hyperparameter tuning via cross-validation
- PCA dimensionality reduction on TF-IDF feature space
- Risk-scoring output for non-technical enforcement staff
