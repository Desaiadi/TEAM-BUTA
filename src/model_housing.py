"""Models for Know Before You Rent: one classifier per problem type.

Question 5: for a building, how likely is each problem type next 4 months?

* Train on Jan–Jun 2025 history / Jul–Oct 2025 outcomes; test on the matching 2026 windows.
* Compare a simple baseline ("it happens again if it happened before"),
  logistic regression, and gradient-boosted trees.
* Feature-group ablation shows how much each kind of information adds:
  building traits → + own history → + landlord → + 100 m neighborhood.
* Risk levels and plain-English reasons turn probabilities into a renter report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features_housing import TARGET_TYPES, feature_columns, slug

MODELS_DIR = config.PROJECT_ROOT / "models"


# --------------------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------------------
def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Split model inputs into the four groups behind questions Q1-Q4."""
    groups = {
        "building": [c for c in columns if c in {"building_age", "unitsres", "numfloors", "is_elevator",
                                                 "is_walkup", "is_registered"}],
        "landlord": [c for c in columns if c.startswith("ll_rate_") or c in {"portfolio_buildings", "has_portfolio"}],
        "neighborhood": [c for c in columns if c.startswith("nb100_")],
        "crime": [c for c in columns if c.startswith("crime_")],
    }
    used = set().union(*groups.values())
    groups["history"] = [c for c in columns if c not in used]
    order = ["building", "history", "landlord", "neighborhood", "crime"]
    return {k: groups[k] for k in order if groups[k]}


def to_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Numeric model matrix: booleans → 0/1, everything float (NaN kept for the tree model)."""
    return df[cols].astype(float)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_model(kind: str):
    """'logit' = scaled logistic regression (NaN filled with 0); 'gbm' = gradient-boosted trees (NaN native)."""
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                              min_samples_leaf=100, l2_regularization=1.0, random_state=0)
    raise ValueError(kind)


def _fit_predict(kind, X_tr, y_tr, X_te):
    model = make_model(kind)
    if kind == "logit":
        X_tr, X_te = X_tr.fillna(0), X_te.fillna(0)
    model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_te)[:, 1]


def scores(y, p) -> dict:
    """AUC (ranking), average precision (finding the positives) and Brier score (calibration)."""
    return {"auc": roc_auc_score(y, p), "avg_precision": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "base_rate": float(np.mean(y))}


def baseline_scores(test: pd.DataFrame, problem: str) -> dict:
    """Naive rule: P(problem next 4 months) = 1 if the building had it in its history window, else 0.

    Scored with its own counts so ties are broken by how often it happened.
    """
    s = slug(problem)
    y, counts = test[f"y_{s}"], test[f"c_{s}"].astype(float)
    out = scores(y, counts.clip(upper=1))        # brier needs a probability: had it = 1, else 0
    out["auc"] = roc_auc_score(y, counts)        # ranking uses the counts, so more = riskier
    out["avg_precision"] = average_precision_score(y, counts)
    return out


@dataclass
class Result:
    problem: str
    model: str
    features: str
    auc: float
    avg_precision: float
    brier: float
    base_rate: float


def evaluate(train: pd.DataFrame, test: pd.DataFrame, kinds=("logit", "gbm")) -> tuple[pd.DataFrame, dict]:
    """Fit every (problem, model) on all features; return the scores and the fitted models."""
    cols = feature_columns(train)
    X_tr, X_te = to_matrix(train, cols), to_matrix(test, cols)
    rows, fitted = [], {}
    for problem in TARGET_TYPES:
        y_tr, y_te = train[f"y_{slug(problem)}"], test[f"y_{slug(problem)}"]
        rows.append(Result(problem, "baseline: had it before", "history", **baseline_scores(test, problem)))
        for kind in kinds:
            model, p = _fit_predict(kind, X_tr, y_tr, X_te)
            fitted[(problem, kind)] = model
            rows.append(Result(problem, kind, "all", **scores(y_te, p)))
    return pd.DataFrame([r.__dict__ for r in rows]), fitted


def ablation(train: pd.DataFrame, test: pd.DataFrame, kind: str = "gbm") -> pd.DataFrame:
    """Add feature groups one at a time and record test AUC / AP for each problem type."""
    groups = feature_groups(feature_columns(train))
    steps, used = [], []
    names = list(groups)          # building, history, landlord, neighborhood[, crime]
    for k, name in enumerate(names):
        used = used + groups[name]
        steps.append((" + ".join(names[:k + 1]), list(used)))
    rows = []
    for problem in TARGET_TYPES:
        y_tr, y_te = train[f"y_{slug(problem)}"], test[f"y_{slug(problem)}"]
        for label, cols in steps:
            _, p = _fit_predict(kind, to_matrix(train, cols), y_tr, to_matrix(test, cols))
            rows.append({"problem": problem, "features": label, **scores(y_te, p)})
    return pd.DataFrame(rows)


def group_importance(model, test: pd.DataFrame, problem: str, n: int = 20_000,
                     seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Permutation importance summed by feature group: AUC lost when a group's columns are shuffled."""
    cols = feature_columns(test)
    sample = test.sample(min(n, len(test)), random_state=seed)
    X, y = to_matrix(sample, cols), sample[f"y_{slug(problem)}"]
    imp = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=3, random_state=seed, n_jobs=4)
    per_feature = pd.Series(imp.importances_mean, index=cols)
    groups = feature_groups(cols)
    return pd.Series({g: per_feature[c].sum() for g, c in groups.items()}).rename(problem), per_feature


# --------------------------------------------------------------------------
# Turning probabilities into a renter report
# --------------------------------------------------------------------------
def risk_level(p: np.ndarray, base_rate: float) -> np.ndarray:
    """Low / Medium / High relative to how common the problem is citywide."""
    ratio = np.asarray(p) / base_rate
    return np.select([ratio >= 2, ratio >= 0.75], ["High", "Medium"], "Low")


REASONS = {
    "c_": "this building had {v:.0f} {t} complaint(s) in the last 6 months",
    "v_": "inspectors issued {v:.0f} {t} violation(s) in the last 6 months",
    "ll_rate_": "the landlord's other buildings average {v:.1f} {t} complaints per 100 apartments",
    "nb100_": "{v:.0f} {t} complaints in other buildings within 100 m",
    "crime_": "{v:.0f} {t} crimes reported per year within 200 m",
}


def explain(logit_model, row: pd.DataFrame, cols: list[str], problem: str, top: int = 3) -> list[str]:
    """Top reasons for one building, from the logistic model's (coefficient × standardized value)."""
    scaler, lr = logit_model.named_steps["standardscaler"], logit_model.named_steps["logisticregression"]
    x = row[cols].astype(float).fillna(0).to_numpy()[0]
    contrib = pd.Series((x - scaler.mean_) / scaler.scale_ * lr.coef_[0], index=cols).sort_values(ascending=False)
    out = []
    for col, c in contrib.head(top).items():
        if c <= 0:
            break
        v = row[col].iat[0]
        prefix = next((p for p in REASONS if col.startswith(p)), None)
        if prefix:
            t = col[len(prefix):].replace("_", " ")
            out.append(REASONS[prefix].format(v=v, t=t))
        elif col == "building_age":
            out.append(f"the building is about {v:.0f} years old")
        else:
            out.append(f"{col.replace('_', ' ')} = {v}")
    return out
