# The 100-Meter Inequality: Know Before You Rent

**Team BUTA · NYU · Programming for Data Science**

NYC building complaint prediction, landlord/neighborhood analysis, and 311
recorded-closure-time regression, using one event study window:
**January 1, 2025–October 31, 2026**.

## Shared study design

All event downloads, cleaning, features and notebooks use the bounds in
`src/config.py`. Descriptive monthly charts cover all 22 months. Annual
comparisons use January–October in both years rather than comparing a complete
2025 with a partial 2026.

| Task | Training | Testing / forecast |
|---|---|---|
| Housing complaint classifiers | Jan–Jun 2025 history → Jul–Oct 2025 outcomes | Jan–Jun 2026 history → Jul–Oct 2026 outcomes |
| Housing app forecast | Refit on Jan–Jun 2026 history → Jul–Oct 2026 outcomes | May–Oct 2026 history → Nov 2026–Feb 2027 forecast |
| 311 closure-hour regressors | Jan–Sep 2025 request cohorts | Jan–Jul 2026 request cohorts |

Housing models predict a complaint within **four months**, using **six months**
of history. Historical annual-model performance from the previous study is not
applicable to this design. Forecast outcomes beyond October 2026 are not used
as observed labels.

311 closures are observed only through October 31, 2026. The 90-day gap before
January 2026 prevents training labels from needing observations during the test
period. August–October 2026 requests can appear in descriptive statistics but
are not fully matured 90-day evaluation cohorts.

## Models and data

- Housing: one logistic regression and histogram gradient boosting classifier
  per problem type (heat, pests, leaks/mold, plumbing, paint/plaster,
  doors/windows, safety). Feature-group ablation and calibration accompany AUC,
  average precision and Brier score.
- 311: linear regression, decision tree, random forest and histogram gradient
  boosting regression. Compare MAE, median absolute error and RMSE in hours.
- Building features: complaint and violation history, latest annual bedbug filings available
  before the feature cutoff (restricted to study-period filings), evictions, landlord leave-one-out rates, neighbors
  within 100 m, and annualized crime counts within 200 m.

NYC Open Data sources: HPD complaints (`ygpa-z7cr`), violations (`wvxf-dwi5`),
registrations/contacts (`tesw-yqqr`, `feu5-w2e2`), PLUTO (`64uk-42ks`), bedbugs
(`wz6d-d3jb`), evictions (`6z8x-wfk4`), NYPD reports (`qgea-i56i`, `5uac-w243`),
and 311 requests (`erm2-nwe9`). Buildings are joined by BBL. The 311 model uses
all returned requests on the 5th, 15th and 25th of each study month, not every
citywide request.

## Explore interface

`/` provides the new 3D Explore map, one-building reports and side-by-side
building comparisons, address search, nearby-building selection, and CSV export.
`/building/<bbl>` retains the full report and `/insights` shows model evaluation.
Street-level 311 reports are fetched directly from NYC Open Data for a selected
lookback ending October 31, 2026, within the shared study window. Map tiles and
geocoding require internet access. These descriptive street summaries are
separate from the trained closure-hour models.

The map still opens without processed building data and explains which pipeline
steps are missing. Outdated study caches are not used for building predictions.

## Run

From the project root on Windows:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m src.download
py -3.12 -m src.clean_housing
py -3.12 -m src.features_housing
py -3.12 -m src.app_data
py -3.12 -m src.insights
py -3.12 -m src.resolution train
py -3.12 app/app.py
```

Open http://127.0.0.1:5050 for building reports. The 311 predictor is currently
available through the command line and notebook, separately from the Flask UI:

```powershell
py -3.12 -m src.resolution predict --created-date "2026-07-20 08:30" --agency HPD --complaint-type "HEAT/HOT WATER" --borough BROOKLYN --incident-zip 11226 --estimator random_forest
py -3.12 -m unittest discover -s tests -v
```

Notebooks 01–12 cover housing exploration, cleaning, features, models and
landlord neighbors. Notebook 13 covers 311 closure-time regression. They can
also be run in order after downloading. See [the 311 guide](docs/Resolution_Time_ML.md).

## Source coverage

The downloaded NYPD source currently ends September 30, 2026; its October
file is empty. `reports/study_coverage.json` records this gap. Missing months
are not interpreted as zero crime, and forecast crime rates are annualized
over the observed months only. Bedbug filing availability is also limited
by publication dates; annual recurrence is not reported without two observed
filing-period years.

## Interpretation and reproducibility

- Complaint and crime data are reported events and reflect reporting behavior.
- 311 predicts recorded agency closure, not confirmed physical repair, capped
  at 90 days. Predictions do not include validated uncertainty intervals.
- PLUTO, ownership/contact information and current violation statuses are
  downloaded snapshots. Their historical revisions cannot be reconstructed;
  event-date filtering does not establish a point-in-time backtest for these
  fields. The housing universe excludes known post-2025 construction to keep
  the evaluated building cohort stable across years.
- Download caches are reusable; readers filter cached event files to the study
  window. Keep caches and study metadata to reproduce results. New downloads
  can include revisions to historical records.
- The PDFs and screenshot under `docs/` describe the previous study design.
  Use the executed notebooks and regenerated reports for current-window results.

Team: Aditya Desai · Cavid Mammadzada · Nabi Shirinov · Ismayil Mirzə-Ağayev
