# Know Before You Rent

*Subproject of The 100-Meter Inequality · Team BUTA · DS-GA 1007*

Before signing a lease, a renter types an address and sees which problems to expect (heat outages, pests, leaks/mold, plumbing, paint/plaster, doors/windows, safety) and whether the landlord is the reason.

## Questions
1. **Building history:** does a building's past predict its future problems?
2. **Landlord effect:** do buildings with the same landlord share problems?
3. **Age and type:** are older, larger or walk-up buildings riskier?
4. **Neighborhood (100 m):** do nearby complaints add to the risk?
5. **Prediction:** for an address, which problems should a renter expect next year?

## Data (NYC Open Data, linked by building ID / BBL)
| Dataset | Rows used | Provides |
|---|---|---|
| HPD Housing Complaints (`ygpa-z7cr`) | 4.0M problems, 2022 → now | What tenants report, by apartment |
| HPD Violations (`wvxf-dwi5`) | 4.5M, 2022 → now | Problems inspectors confirmed, by severity A/B/C |
| HPD Registrations + Contacts (`tesw-yqqr`, `feu5-w2e2`) | 187k buildings / 745k contacts | Owner and managing agent → landlord portfolios |
| PLUTO (`64uk-42ks`) | 858k lots | Year built, apartments, floors, building class, location |
| Bedbug Reporting (`wz6d-d3jb`) | 727k filings | Yearly infested units per building |
| Evictions (`6z8x-wfk4`) | 138k | Executed evictions by building |
| 311 Service Requests (`erm2-nwe9`) | 23.3M | Downloaded in full; used by the scrapped weather idea |

## Results so far
- **The 100-meter inequality and the landlord:** take buildings within 100 m of each other that are similar (same era and size). When one gets a problem, a neighbor with the **same landlord** also gets it **2.7–3.1×** more often than a neighbor with a different landlord. Among pairs with problems, same-landlord neighbors' overall problem rates differ **37% less**.
- **Q1:** a building with a problem in 2022–23 was **11–15× more likely** to have it again in 2024.
- **Q2:** buildings whose landlord's *other* buildings were in the worst 25% were **4.5–6.8× more likely** to have problems than the best 25%.
- **Q4:** even buildings with a clean history were **3–4× more likely** to get a problem when their 100 m neighborhood was busiest.
- **Q5:** gradient-boosted models, trained on 2024 and tested on 2025, reach **AUC 0.89–0.93** for every problem type. The "it happened before" rule reaches 0.70–0.83.
  - Buildings rated **High** risk had heat complaints 72% of the time in 2025; **Low** ones, 4%.

## Pipeline
| Notebook | Step |
|---|---|
| `rent_01`–`rent_04` | Explore HPD complaints, violations, landlords, bedbugs and evictions |
| `rent_05`–`rent_08` | Clean each dataset, map everything to one set of problem types, build landlord portfolios, clean PLUTO |
| `rent_09` | Build one row per building: traits, 2-year history, landlord record (leave-one-out), 100 m neighborhood, next-year targets |
| `rent_10` | Train and compare the models: feature-group ablation, permutation importance, calibration, renter reports |
| `rent_11` | Does the landlord explain the 100-meter inequality? Same- vs different-landlord neighbor pairs |

| Code (`src/`) | |
|---|---|
| `config.py` | Paths, date ranges, dataset ids |
| `download.py` | Resumable, month-by-month downloads from the Socrata API |
| `housing.py` | Loading, keys, concentration and persistence, regex topics, name normalization |
| `clean_housing.py` | Cleaning steps with an audit report; landlord portfolios |
| `features_housing.py` | Feature table; 100 m neighbor graph (k-d tree + sparse matrix) |
| `model_housing.py` | Models, ablation, importance, risk levels, explanations |
| `landlord_neighbors.py` | Neighbor pairs within 100 m, same vs different landlord, bootstrap CIs |
| `preprocess.py`, `transform.py`, `plots.py` | Shared cleaning, full-311 streaming, plotting |

`scrapped_ideas/weather_sensitivity/` is the first idea we explored, kept for reference.

## Run it
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.download --only pluto
python -m src.download --only housing
```
Then run `notebooks/rent_01` → `rent_10` in order, with VS Code's kernel set to `.venv`. Data goes to `data/` and models to `models/`; neither is committed.
