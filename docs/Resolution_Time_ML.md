# NYC 311 closure-hour ML project

## Shared event window

January 1, 2025–October 31, 2026, configured centrally in `src/config.py`.
The downloader caches requests filed on the 5th, 15th and 25th of each month.
This is a deterministic selected-day sample, not a probability sample or the
entire citywide dataset.

- Train: requests filed January–September 2025.
- Test: requests filed January–July 2026.
- Follow-up cutoff: October 31, 2026, New York local time.
- October–December 2025 separates training from testing so every training
  outcome has 90 days of observation before the test period begins.
- August–October 2026 filings remain available for description but are excluded
  when their 90-day follow-up is incomplete. They are not labeled as 90-day
  nonclosures merely because a closure is missing.

## Target and models

Predict elapsed hours until **recorded agency closure**, capped at 2,160 hours
(90 days). This does not verify when the physical problem was repaired.
Missing closures get the cap only after sufficient follow-up. A closure after
the observation cutoff is treated as unobserved. Invalid and negative intervals
are excluded; New York daylight-saving offsets are accounted for.

Inputs: agency, complaint type, descriptor, borough, ZIP, reporting channel,
location type, filing month, weekday and hour. ZIP codes remain strings.

Compare four fixed pipelines: linear regression, decision tree, random forest
and histogram gradient boosting. Preprocessing is fitted only on training data.
Linear regression and ordinary trees use one-hot categorical encoding and
standardized calendar inputs. Boosting uses native categorical features with
training-fitted ordinal codes. Rare/unseen categories are handled explicitly.

All predict `log1p(hours)` before converting back to hours and clipping to the
cap. The first three use squared log error; boosting uses absolute log error.
Thus these are comparisons of fixed pipelines, not isolated algorithm changes
under an identical loss. The forest has 100 trees, depth at most 18, and leaf
size at least 40. Gradient boosting remains the fixed deployment default;
held-out results do not automatically select the deployment algorithm.

## Run

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m src.resolution train
py -3.12 -m src.resolution predict --created-date "2026-07-20 08:30" --agency HPD --complaint-type "HEAT/HOT WATER" --descriptor "ENTIRE BUILDING" --borough BROOKLYN --incident-zip 11226 --open-data-channel-type ONLINE --location-type "RESIDENTIAL BUILDING" --estimator random_forest
```

Supported estimators: `linear_regression`, `decision_tree`, `random_forest`,
`gradient_boosting`. An optional `NYC_APP_TOKEN` raises Socrata API rate limits.

For local CSV/parquet input:

```powershell
py -3.12 -m src.resolution train --input requests.csv
```

The input needs `unique_key`, `created_date`, `closed_date` and all seven
categorical columns. Missing categorical values are allowed. `--as-of` may
specify an earlier extraction cutoff but is never allowed to extend observation
beyond October 31, 2026. Routing/category revisions cannot be reconstructed.

Open `notebooks/13_resolution_time_ml.ipynb` for charts, error breakdowns and
side-by-side request predictions. It reuses artifacts with the matching study
ID; set `RETRAIN = True` to refit. Training takes several minutes.

## Outputs and validation

- `models/resolution.joblib`: four fitted pipelines and study provenance.
- `reports/resolution/metrics.csv`: MAE, median absolute error and RMSE in hours.
- `reports/resolution/model_predictions.parquet`: every model's held-out estimates.
- `reports/resolution/errors_by_complaint.csv`: default model's segment errors.
- `reports/resolution/holdout_predictions.parquet`: default model diagnostics.
- `reports/resolution/training.json`: cohort dates, observation cutoff and audit.

These metrics must be regenerated for this window; previous 2025-only metrics
are not carried forward. Compare average and median errors since long-running
requests can dominate MAE. Estimates are point predictions without a validated
uncertainty interval. Only load trusted joblib files.

```powershell
py -3.12 -m unittest discover -s tests -v
```
