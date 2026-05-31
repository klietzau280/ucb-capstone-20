# Predicting Bylaw Violations from Code Complaints

*A machine learning capstone — UC Berkeley ML/AI Professional Certificate*

---

## Executive Summary (for a non-technical reader)

Every year the City of Austin receives tens of thousands of code-enforcement
complaints — overgrown lots, unsafe structures, illegal land use, building work
without permits. An inspector has to visit each one, but **only about 45% turn
out to be real violations.** The other 55% are dead ends that still consume a
site visit.

This project asks a simple, practical question:

> **When a complaint first comes in, can we predict how likely it is to be a real
> violation — so inspectors visit the most promising cases first?**

Using ~83,000 real Austin complaint records, we built and compared several
machine learning models. The best model (a **Random Forest**) correctly ranks a
genuine violation above a non-violation **about 74% of the time** — far better
than guessing — using only information available the moment a complaint is filed.
The single biggest clues are **what the complaint is about**, **who reported it**,
**where the property is**, and **whether the property is a repeat offender.**

Used as a triage tool, this lets a code-enforcement team work its daily queue in
**order of risk** instead of first-come-first-served, getting inspectors to real
problems sooner and wasting fewer visits on dead ends.

---

## 1. Problem Statement

**Goal.** Help a municipal code-enforcement office work smarter by predicting, at
intake, which complaints are likely to result in a confirmed bylaw violation.

**Why it matters.** Inspector time is the scarce resource. A better-ordered queue
means real violations are caught faster (better safety and compliance) and fewer
visits are spent on complaints that turn out to be nothing (lower cost).

**The challenge.** At intake we know very little — the complaint type, who filed
it, the location, and a dispatcher-assigned priority. We deliberately do **not**
use anything that is only known *after* a case closes (such as how long it took
to resolve), because a real triage tool would not have that information yet.
Predicting an inspection outcome from intake data alone is genuinely hard, which
makes an honest, leakage-free model the right benchmark.

## 2. Model Outcomes & Predictions

- **Learning type:** *Supervised* learning (each historical case has a known
  outcome label).
- **Task:** *Binary classification.*
- **Output:** a **probability from 0 to 1** that a complaint will be upheld as a
  violation. This risk score can be used directly to rank a queue, or thresholded
  into a yes/no flag that the team tunes to its inspection capacity.

**Target variable — `violation_upheld`** (derived from each case's resolution text):

| Label | Meaning | Example resolution text |
|---|---|---|
| **1 — Violation Upheld** | A real problem was confirmed or corrected | *Abated by Inspector, Voluntary Compliance, Legal Action* |
| **0 — No Violation** | Inspection found nothing | *No Violation(s) Found / Inspection Performed* |

Administrative closures (duplicates, withdrawn, referred to another agency,
supervisor review) carry no inspection verdict and are **excluded** from modeling.

## 3. Data Acquisition

**Source:** [Austin Code Complaint Cases — City of Austin Open Data Portal](https://data.austintexas.gov/Public-Safety/Austin-Code-Complaint-Cases/6wtj-zbtb)
(~83,000 records; downloaded automatically the first time a notebook runs).

Fields used as model inputs (all available **at intake**):

| Field | Description |
|---|---|
| `description` | Complaint type (Property Abatement, Structure Condition, Land Use, Work Without Permit) |
| `priority` | Dispatcher-assigned urgency (1 = highest) |
| `repeatoffenderrelated` | Whether the property is a flagged repeat offender |
| `shorttermrentalrelated` | Whether the complaint is short-term-rental related |
| `reportedby` | How the complaint was filed (complainant, anonymous, inspector, …) |
| `zip_code` | Property ZIP code |
| `opened_date` | Filing date → month, day-of-week, year |

**Does the data have signal?** Exploratory analysis ([`01_eda.ipynb`](notebooks/01_eda.ipynb))
shows clear, usable differences in the confirmed-violation rate across complaint
types and properties — exactly what a classifier can learn from:

- **Complaint type matters:** Property Abatement complaints are upheld **50.1%**
  of the time vs. just **30.0%** for Work Without Permit (overall rate: 44.7%).
- **Repeat offenders matter:** **58.9%** violation rate vs. **43.5%** for
  first-time complaints.
- **Location matters:** complaint volume and outcome cluster in specific ZIP codes
  (78758, 78741, 78745, 78704, 78702).
- **Timing:** complaints spike Monday–Wednesday and drop on weekends —
  intake is business-hours driven.

## 4. Data Preprocessing & Preparation

Detailed in [`01_eda.ipynb`](notebooks/01_eda.ipynb); applied consistently in the
modeling notebook.

- **Missing values & inconsistencies:** removed duplicate rows; parsed dates;
  standardized the complaint type to four clean categories (the raw field
  contained duplicated-phrase and plural variants); normalized the reporting
  source (`Investigator` vs `investigator`); fixed invalid `priority` codes
  (0, 9, 90 → treated as missing, then median-imputed inside the model pipeline).
- **Encoding:** numeric features are median-imputed and standardized; categorical
  features (complaint type, reporting source, ZIP) are **one-hot encoded**, with
  rare categories grouped together to avoid overfitting. All encoding lives inside
  a scikit-learn `Pipeline`/`ColumnTransformer`, so it is fit **only on training
  data** within each cross-validation fold — no leakage from the test set.
- **Avoiding target leakage:** `days_to_close` and `closed_date` are **excluded**
  because they are only known after a case is resolved.
- **Train/test split:** a **stratified 80/20 split** (48,537 train / 12,135 test)
  preserves the 55/45 class balance. The test set is untouched until final
  evaluation.

## 5. Modeling

Full workflow in [`02_modeling.ipynb`](notebooks/02_modeling.ipynb). We compared
a naive baseline against four classifiers, then tuned the strongest with grid
search:

1. **Dummy classifier** (predicts the majority class) — a sanity-check floor.
2. **Logistic Regression** — fast, interpretable linear baseline.
3. **Decision Tree** — captures simple non-linear rules.
4. **Random Forest** — ensemble of trees; robust to mixed feature types.
5. **Hist Gradient Boosting** — strong modern boosted-tree model.

Each model was evaluated with **5-fold stratified cross-validation**, and the
three strongest families were tuned with **`GridSearchCV`** (optimizing F1) over
regularization strength, tree depth, ensemble size, and learning rate.

## 6. Model Evaluation

**Evaluation metric — F1-score (primary), ROC-AUC (secondary).**
We optimize **F1** (the balance of precision and recall) because both mistakes are
costly: a **false negative** lets a real violation slip through un-inspected, while
a **false positive** wastes a scarce inspector visit. Plain accuracy would hide
this trade-off. **ROC-AUC** is reported as a threshold-independent measure of how
well each model ranks risky cases above clean ones.

**Results on the held-out test set:**

| Model | Accuracy | Precision | Recall | **F1** | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (best)** | 0.664 | 0.616 | 0.658 | **0.636** | **0.735** |
| Hist Gradient Boosting | 0.667 | 0.656 | 0.537 | 0.590 | 0.725 |
| Logistic Regression | 0.614 | 0.567 | 0.580 | 0.573 | 0.653 |
| Dummy (majority class) | 0.553 | — | 0.000 | 0.000 | 0.500 |

**Best model: Random Forest** — highest F1 (0.636) and ROC-AUC (0.735). A
ROC-AUC of 0.735 means that, given one real violation and one non-violation at
random, the model assigns the higher risk score to the real violation ~74% of the
time. Every model beats the majority-class baseline, confirming there is genuine,
learnable signal in routine intake data.

**What drives the predictions** (permutation importance): **complaint type** is
the strongest signal, followed by **reporting source**, **ZIP code**, then
**priority** and **repeat-offender** status — consistent with the EDA.

## Key Findings (plain language)

- A complaint's **type, location, and reporting source**, plus whether the
  property is a **repeat offender**, are enough to meaningfully predict whether an
  inspection will find a real violation.
- The model is a useful **prioritization aid, not an autopilot** — it ranks
  likelihood; an inspector still makes the call.
- Predicting from intake data alone has a real ceiling: the most reliable outcome
  signals (days-to-close) cannot be used honestly because they aren't known yet.
  We **prove** this in the notebook — adding that one leaked feature inflates
  ROC-AUC from 0.735 to **0.924**, a gain that would simply vanish in production.
  This points clearly to the value of connecting other (intake-time) city data
  sources instead.

## Recommendations

1. **Triage the daily queue by risk score** instead of first-in-first-out, so
   inspectors reach likely violations sooner.
2. **Tune the decision threshold to capacity:** raise it when inspectors are
   scarce (favor precision), lower it during slow periods (favor recall). The
   modeling notebook quantifies this precision/recall trade-off.
3. **Fast-track repeat-offender and Property-Abatement complaints**, which have
   the highest confirmed-violation rates.

## Next Steps

- **Add external data** (property-tax, 311, permit history, census) — likely the
  biggest lever for higher accuracy.
- **Richer geography** beyond ZIP (parcel age, lot size, neighborhood income).
- **Fairness audit** before any deployment, to ensure risk scores don't
  systematically over-target specific neighborhoods.
- **Monitoring & retraining** to handle drift as policy and complaint mix change.

## Project Structure

```
.
├── data/
│   └── austin_code_complaints.csv      # auto-downloaded on first notebook run
├── notebooks/
│   ├── 01_eda.ipynb                    # data understanding & exploratory analysis
│   └── 02_modeling.ipynb               # modeling, tuning & evaluation
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
jupyter lab            # then run notebooks/01_eda.ipynb, then 02_modeling.ipynb
```

Both notebooks run top-to-bottom with no manual steps; the dataset downloads
automatically on first run. Each notebook is also saved with all outputs so the
analysis can be read on GitHub without executing anything.
