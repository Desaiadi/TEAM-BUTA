# The 100-Meter Inequality: Know Before You Rent

**Team BUTA** · DS-GA 1007 Programming for Data Science · NYU · Fall 2026

> Does living across the street change your NYC experience?

City and borough averages hide sharp local differences: two apartment buildings 100 meters apart can have very different heat, pest, leak and safety problems. This project measures those gaps using NYC open data, explains them, and turns them into a prediction a renter can use before signing a lease.

📄 **[Read the project report (PDF)](docs/Project_Report.pdf)**

![Building report in the Know Before You Rent web app](docs/screenshots/demo_building_report.png)

## What we found
Covering 172,546 NYC apartment buildings (3+ units), January 2022 to November 2026:

| Question | Finding |
|---|---|
| **History:** does a building's past predict its future? | A building with a problem in 2022–23 was **11–15× more likely** to have it again in 2024. |
| **Landlord:** do same-landlord buildings share problems? | Buildings whose landlord's *other* buildings rank in the worst 25% were **4.5–6.8× more likely** to have each problem than the best 25%. Among similar buildings within 100 m, a neighbor with the **same landlord** shares the problem **2.7–3.1× more often**. |
| **Building type:** are older or larger buildings riskier? | 1920–59 buildings have the highest rates (heat: 19% vs 8% for 1980–99). |
| **Neighborhood:** does the block add risk? | Buildings with a clean history were **3–4× more likely** to get a problem in the busiest 25% of 100 m surroundings. |
| **Prediction:** which problems should a renter expect? | Models trained on 2024 and tested on 2025 reach **AUC 0.89–0.93** for all seven problem types, against 0.70–0.83 for "it happened before". High-risk buildings had heat complaints 72% of the time; low-risk, 4%. |

**What drives the predictions:**
- The building's own history adds the most accuracy.
- The landlord adds little once the history is known (+0.002 AUC). For buildings with *no* complaint history it adds +0.03 to +0.05: there, the landlord is the best clue.

## The model
**What it predicts:** for each apartment building, the chance of at least one tenant complaint of each type in the next 12 months. The seven types are heat & hot water, pests, leaks & mold, plumbing, paint & plaster, doors & windows, and safety. Each type gets its own yes/no classifier: **14 models in total** (7 types × 2 algorithms).

| Model | What it does | Role | Test AUC |
|---|---|---|---|
| Baseline rule | "It happens again if it happened before"; no learning | The bar to beat | 0.70–0.83 |
| Logistic regression | Weighs each input and adds them up (scikit-learn, standardized inputs) | Plain-English reasons in the app | 0.87–0.91 |
| **Gradient-boosted trees** | Up to 300 small decision trees, each correcting the last, with early stopping (`HistGradientBoostingClassifier`) | **Main model**: probabilities and risk levels | **0.89–0.93** |

**Inputs: 54 per building**

| Group | Features |
|---|---|
| Building | Age, apartments, floors, elevator or walk-up, HPD-registered |
| Own history (previous 2 years) | Complaints per problem type, violations by class A/B/C and by type, emergencies, bedbug years, evictions |
| Landlord | Complaint rates in the landlord's *other* buildings per 100 apartments (leave-one-out), portfolio size |
| Within 100 m | Complaints in neighboring buildings, number of neighbors (k-d tree + sparse matrix) |
| Crime within 200 m | Reported crimes per year by group and severity |

**How it is trained and tested.** The split is by time, so the model never sees the future:
- **Train:** 2022–23 history → 2024 outcomes.
- **Test:** 2023–24 history → 2025 outcomes.

**What we measure:**
- **AUC:** ranking quality.
- **Average precision:** finding the buildings that get the problem.
- **Calibration:** whether a predicted 30% really means 30%.
- **Ablation:** what each input group adds.

| Problem | Gradient boosting | Logistic regression | Baseline | Base rate |
|---|---|---|---|---|
| Paint & plaster | 0.929 | 0.910 | 0.820 | 8.5% |
| Doors & windows | 0.925 | 0.906 | 0.800 | 8.0% |
| Plumbing | 0.920 | 0.901 | 0.801 | 9.5% |
| Leaks & mold | 0.919 | 0.900 | 0.820 | 10.4% |
| Heat & hot water | 0.914 | 0.891 | 0.831 | 16.8% |
| Pests | 0.913 | 0.892 | 0.799 | 8.9% |
| Safety | 0.892 | 0.867 | 0.695 | 3.8% |

**Risk levels.** The app turns probabilities into risk levels: **High** means at least 2× the citywide rate, **Low** means under 0.75×. On 2025, buildings rated High had heat complaints 72% of the time, Medium 24% and Low 4%.

**What drives accuracy (ablation):**
- Building traits alone: AUC 0.83–0.87.
- Adding own history: 0.89–0.93, the biggest jump.
- Landlord: +0.002 overall, but +0.03 to +0.05 for buildings with no complaint history.
- 100 m neighborhood and crime: about +0.001 each.

**Why training is fast.** The heavy work, downloading and cleaning ~37M records and building features, happens once beforehand. The model sees a compact table of 172,546 rows × 54 numbers. Histogram-based boosting bins each feature into 255 buckets and uses all CPU cores, so each model trains in 2–4 seconds. All 14 train in about a minute on a laptop; no GPU is needed.

**For the app:** the models are retrained on the 2025 table and score every building from its 2025–26 history (`python -m src.app_data`). Code: `src/model_housing.py`, `src/features_housing.py`; walkthrough in `notebooks/10_model.ipynb`.

## The product
A Flask web app. Type an NYC address and get the building's report:
- Its risk for 7 problem types next year, with plain-English reasons.
- Red flags: open hazardous violations, lead, bedbugs, evictions.
- Its complaint and violation history.
- How the landlord's other buildings compare with NYC.
- Reported crime nearby.
- A map of the landlord's nearby buildings.

A **Data & model insights** page charts the datasets, the findings and the model evaluation. Every chart has a "View as table" toggle.

## Repository layout
```
├── src/            Python package: download, cleaning, features, models, app data
├── notebooks/      01–12: explore → clean → build features → model → analysis
├── app/            Flask web app (templates, charts, styles)
├── reports/        Cleaning audit reports, model results, figures
├── docs/           Project report, screenshots, original proposal and brief
├── archive/        An earlier direction we explored (weather sensitivity of 311 complaints)
├── data/           Downloaded and processed data (created locally, not committed)
└── models/         Trained models (created locally, not committed)
```

| Notebook | Step |
|---|---|
| `01`–`04` | Explore HPD complaints, violations, landlords, bedbugs and evictions |
| `05`–`08` | Clean each dataset, build landlord portfolios, clean PLUTO |
| `09` | Build the building feature tables; first answers to the questions |
| `10` | Train and evaluate the models |
| `11` | Do same-landlord neighbors share problems? (the 100-meter question) |
| `12` | Explore and clean NYPD crime data |

| Module (`src/`) | Role |
|---|---|
| `config.py`, `download.py` | Dataset ids and paths; resumable month-by-month downloads from the NYC Open Data API |
| `housing.py`, `clean_housing.py`, `preprocess.py` | Loading, cleaning steps with audit reports, regex problem tagging, landlord portfolios |
| `features_housing.py` | One row per building; the 100 m neighbor graph (k-d tree + sparse matrix); crime within 200 m |
| `model_housing.py`, `insights.py` | Models, ablation, importance, calibration, risk levels, chart data |
| `landlord_neighbors.py` | Same- vs different-landlord neighbor pairs with bootstrap intervals |
| `address.py`, `report.py`, `app_data.py` | Address normalization and search, renter reports, precomputed app data |
| `transform.py`, `plots.py` | Streaming the full 311 table; shared plotting helpers |

## Run it
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.download --only pluto
python -m src.download --only housing      # HPD, bedbugs, evictions, crime (~1 GB, 30–60 min)
```
Run `notebooks/01` → `12` in order (in VS Code, select the `.venv` kernel). Then start the app:
```bash
python -m src.app_data      # scores every building (~1 min)
python -m src.insights      # chart data for the insights page (~2 min)
python app/app.py           # open http://127.0.0.1:5050
```

## Data
All public [NYC Open Data](https://opendata.cityofnewyork.us/), linked by **BBL**, the 10-digit borough–block–lot building ID:

| Dataset (id) | Rows used | Provides |
|---|---|---|
| HPD Housing Complaints (`ygpa-z7cr`) | 4.0M problems | What tenants report, by apartment and problem type |
| HPD Violations (`wvxf-dwi5`) | 4.5M | Problems inspectors confirmed, by severity A/B/C |
| HPD Registrations + Contacts (`tesw-yqqr`, `feu5-w2e2`) | 187k / 745k | Owners and managing agents → 116k landlord portfolios |
| PLUTO (`64uk-42ks`) | 858k lots | Year built, apartments, floors, building class, location, address |
| Bedbug Reporting (`wz6d-d3jb`) | 727k filings | Yearly infested apartments per building |
| Evictions (`6z8x-wfk4`) | 60k residential | Executed evictions by building |
| NYPD Complaints (`qgea-i56i`, `5uac-w243`) | 2.67M | Reported crime within 200 m (no demographic fields used) |
| 311 Service Requests (`erm2-nwe9`) | 23.3M | Full history; profiled, and used in the archived analysis |

## Roadmap
- **City response times (in progress):** a model of how long NYC 311 service requests take to close, by request type and location. It will be added to `src/` and `notebooks/` and shown in the building report.
- A fair landlord ranking that compares each landlord with similar buildings.
- Risk per apartment, not just per building.
- Rent-stabilized status and changes of ownership.

## Limitations
- Complaints and crimes are *reported* events, not guaranteed problems; reporting also varies by neighborhood.
- Only rentals with 3+ apartments must register an owner, and landlord matching by agent or office address is approximate.
- Some same-landlord neighbors are parts of one complex.
- A similar public tool, JustFix "Who Owns What", maps landlord portfolios. This project adds the prediction, the 100 m comparison and the renter report.

## Team
Aditya Desai · Cavid Mammadzada · Nabi Shirinov · Ismayil Mirzə-Ağayev
