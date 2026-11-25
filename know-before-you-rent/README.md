# Know Before You Rent

The code behind **The 100-Meter Inequality** (Team BUTA · DS-GA 1007 · NYU): data pipeline, analysis notebooks, models and web app.

Before signing a lease, a renter types an address and sees which problems the building is likely to have next year, how confident we are, and whether the landlord is the reason. The problem types are heat and hot water, pests, leaks and mold, plumbing, paint and plaster (lead risk), doors and windows, and safety.

## Research questions
1. **Building history:** does a building's past predict its future problems?
2. **Landlord:** do buildings with the same landlord share problems, more than random buildings?
3. **Building type:** are older, larger or walk-up buildings riskier?
4. **Neighborhood:** do complaints within 100 m add to a building's risk?
5. **Prediction:** for a given address, which problems should a renter expect next year, and how confident are we?

## Data
All public NYC Open Data, linked by **BBL**, the 10-digit borough–block–lot building ID. Dated data covers January 2022 to November 2026.

| Dataset (id) | Rows used | Provides |
|---|---|---|
| HPD Housing Complaints (`ygpa-z7cr`) | 4.0M problems | What tenants report, by apartment and problem type |
| HPD Violations (`wvxf-dwi5`) | 4.5M | Problems inspectors confirmed, by severity A/B/C |
| HPD Registrations + Contacts (`tesw-yqqr`, `feu5-w2e2`) | 187k buildings / 745k contacts | Owners and managing agents → 116k landlord portfolios |
| PLUTO (`64uk-42ks`) | 858k lots | Year built, apartments, floors, building class, location, address |
| Bedbug Reporting (`wz6d-d3jb`) | 727k filings | Yearly infested apartments per building |
| Evictions (`6z8x-wfk4`) | 60k residential | Executed evictions by building |
| NYPD Complaints, historic + current year (`qgea-i56i`, `5uac-w243`) | 2.67M | Reported crime within 200 m of each building (no demographic fields used) |
| 311 Service Requests (`erm2-nwe9`) | 23.3M | Full history; profiled, and used in the exploratory weather analysis |

## Results
Covering 172,546 residential buildings with 3+ apartments:

| Question | Finding |
|---|---|
| Q1 History | A building with a problem in 2022–23 was **11–15× more likely** to have it again in 2024. |
| Q2 Landlord | Buildings whose landlord's *other* buildings rank in the worst 25% were **4.5–6.8× more likely** to have each problem than the best 25%. Among similar buildings within 100 m, a neighbor with the **same landlord** shares the problem **2.7–3.1× more often**. |
| Q3 Building type | 1920–59 buildings have the highest rates (heat: 19% vs 8% for 1980–99). Building traits alone rank risk at AUC 0.83–0.87. |
| Q4 Neighborhood | Buildings with a clean history were **3–4× more likely** to get a problem in the busiest 25% of 100 m surroundings. |
| Q5 Prediction | Gradient boosting reaches **AUC 0.89–0.93** on 2025 (trained on 2024), against 0.70–0.83 for "it happened before". Buildings rated **High** risk had heat complaints 72% of the time in 2025; **Low**, 4%. |

**What drives the predictions:**
- The building's own history adds the most accuracy.
- Once history is known, the landlord adds +0.002 AUC overall. But for buildings with no complaint history it adds **+0.03 to +0.05**: there, the landlord is the best clue.
- The 100 m neighborhood and nearby crime add little to prediction. They're shown to renters as context.

## How it works
**1. Download.** `src/download.py` pulls each dataset from the NYC Open Data API month by month. It can resume after an interruption, and it streams the 23M-row 311 table without loading it all at once.

**2. Clean.** `src/clean_housing.py` and `src/preprocess.py`:
- Deduplicate records and parse dates.
- Validate BBLs.
- Map HPD categories and free-text violation descriptions (regex) onto one set of problem types.
- Every step writes an audit report to `reports/`.

**3. Landlords.** For each building's latest registration, pick one landlord key: the managing-agent company, else the agent person plus their office address, else the owner's office address. Buildings sharing a key form a portfolio. (Linking every shared key transitively merged 18,703 unrelated buildings into one "landlord", so we use this priority rule instead.)

**4. Features.** `src/features_housing.py` builds one row per building:
- 2-year history: complaints by type, violations by class, bedbugs, evictions.
- Building traits.
- The landlord's complaint rates in their *other* buildings (leave-one-out).
- Complaints in other buildings within 100 m, using a k-d tree and a sparse neighbor matrix.
- Reported crime within 200 m.

**5. Models.** `src/model_housing.py` fits one classifier per problem type:
- Gradient-boosted trees (`HistGradientBoostingClassifier`) for the probabilities.
- Logistic regression for the plain-English reasons.

They train on 2024 outcomes and are tested on 2025, compared with a "happened before" baseline. Evaluation covers feature-group ablation, permutation importance and calibration.

**6. App.** `src/app_data.py` scores every building, `src/insights.py` precomputes the chart data, and `app/` is a Flask site with address search.

| Notebook | Step |
|---|---|
| `rent_01`–`rent_04` | Explore HPD complaints, violations, landlords, bedbugs and evictions |
| `rent_05`–`rent_08` | Clean each dataset, build landlord portfolios, clean PLUTO |
| `rent_09` | Build the building feature tables; first answers to Q1–Q4 |
| `rent_10` | Train and evaluate the models (Q5) |
| `rent_11` | Do same-landlord neighbors share problems? (the 100-meter question) |
| `rent_12` | Explore and clean NYPD crime data |

## Run it
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.download --only pluto
python -m src.download --only housing      # HPD, bedbugs, evictions, crime (~1 GB, 30–60 min)
```
Then run `notebooks/rent_01` → `rent_12` in order, with VS Code's kernel set to `.venv`.

**Web app:**
```bash
python -m src.app_data      # scores every building (~1 min)
python -m src.insights      # chart data for the Insights page (~2 min)
python app/app.py           # open http://127.0.0.1:5050
```
- **Search:** type an address. "235 E. 92nd St, Apt 4B" and similar spellings work.
- **Building report:** risk cards with reasons, red flags, complaint and violation history, the landlord vs NYC, crime nearby, and a map of the landlord's buildings.
- **Data & model insights:** dataset trends, findings for Q1–Q4, and the model evaluation (accuracy, ablation, risk levels, calibration).

Every chart has a "View as table" toggle. Data (`data/`) and trained models (`models/`) are created locally and not committed.

## Roadmap
- **City response times (in progress):** a model of how long NYC 311 service requests take to close, by request type and location. It will join this repository and appear in the building report.
- A fair landlord ranking that compares each landlord with similar buildings.
- Risk per apartment, not just per building.
- Rent-stabilized status and changes of ownership.

## Limitations
- Complaints and crimes are *reported* events, not guaranteed problems; reporting also varies by neighborhood.
- Only rentals with 3+ apartments must register an owner, and landlord matching by agent or office address is approximate.
- Some same-landlord neighbors are parts of one complex.
- A similar public tool, JustFix "Who Owns What", maps landlord portfolios. This project adds the prediction, the 100 m comparison and the renter report.

## Also in this folder
`scrapped_ideas/weather_sensitivity/` is an earlier direction: how much 311 complaints react to cold and rain, by neighborhood. It's kept for reference.
