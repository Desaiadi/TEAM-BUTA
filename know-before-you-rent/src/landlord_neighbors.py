"""Does the landlord explain the 100-meter inequality?

Take every pair of residential buildings within 100 m of each other and ask:
are buildings with the SAME landlord more alike in their problems than
neighbors with DIFFERENT landlords?

* Same location (≤100 m), so the block, the weather, the city services and
  the neighborhood are shared. What differs between the two pair types is
  whether one landlord runs both buildings.
* A fair version also requires the two buildings to be similar (same era and
  size band), so "same landlord" is not just "same kind of building".
* Uncertainty comes from a bootstrap over pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.features_housing import TARGET_TYPES, slug


def neighbor_pairs(universe: pd.DataFrame, A: sparse.csr_matrix) -> pd.DataFrame:
    """One row per unordered pair of buildings within the neighbor radius."""
    upper = sparse.triu(A, k=1).tocoo()
    i, j = upper.row, upper.col
    u = universe.reset_index(drop=True)
    pairs = pd.DataFrame({"i": i, "j": j})
    pairs["same_landlord"] = u["landlord_id"].to_numpy()[i] == u["landlord_id"].to_numpy()[j]
    # Both buildings must be registered: unregistered ones have no landlord to compare
    pairs["both_registered"] = u["is_registered"].to_numpy()[i] & u["is_registered"].to_numpy()[j]
    size = pd.cut(u["unitsres"], [0, 5, 10, 20, 50, 100, 1e6], labels=False).to_numpy()
    era = u["era"].fillna("unknown").astype(str).to_numpy()
    pairs["similar_buildings"] = (size[i] == size[j]) & (era[i] == era[j])
    return pairs


def building_rates(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Per building: complaints per 100 apartments (history window) overall and by type, plus next-year flags."""
    s = snapshot.reset_index(drop=True)
    out = pd.DataFrame({"all problems": s["c_total"] / s["unitsres"] * 100})
    for t in TARGET_TYPES:
        out[t] = s[f"c_{slug(t)}"] / s["unitsres"] * 100
    return out


def pair_gaps(pairs: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Absolute difference in log(1 + rate) for every pair and measure.

    The log keeps a few extreme buildings from dominating; a gap of 0.69 means
    one building's rate is about double the other's.
    """
    lr = np.log1p(rates.to_numpy())
    gaps = np.abs(lr[pairs["i"].to_numpy()] - lr[pairs["j"].to_numpy()])
    return pd.DataFrame(gaps, columns=rates.columns, index=pairs.index)


def at_least_one_problem(pairs: pd.DataFrame, rates: pd.DataFrame) -> pd.Series:
    """True for pairs where at least one building reported a problem.

    Two problem-free buildings always look "alike" and say nothing about the
    landlord; they make up ~49% of different-landlord pairs but only ~31% of
    same-landlord pairs, which would bias a naive comparison.
    """
    has = (rates["all problems"] > 0).to_numpy()
    return pd.Series(has[pairs["i"].to_numpy()] | has[pairs["j"].to_numpy()], index=pairs.index)


def compare(pairs: pd.DataFrame, gaps: pd.DataFrame, mask: pd.Series, n_boot: int = 500,
            seed: int = 0) -> pd.DataFrame:
    """Mean gap for same- vs different-landlord pairs (within `mask`), with a bootstrap 95% CI of the ratio."""
    rng = np.random.default_rng(seed)
    p, g = pairs[mask], gaps[mask]
    same, diff = g[p["same_landlord"]].to_numpy(), g[~p["same_landlord"]].to_numpy()
    rows = []
    for k, col in enumerate(g.columns):
        s, d = same[:, k], diff[:, k]
        ratios = []
        for _ in range(n_boot):
            ratios.append(rng.choice(s, len(s)).mean() / rng.choice(d, len(d)).mean())
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        rows.append({"measure": col, "same-landlord pairs": len(s), "different-landlord pairs": len(d),
                     "gap: same landlord": s.mean(), "gap: different landlord": d.mean(),
                     "same ÷ different": s.mean() / d.mean(), "95% CI low": lo, "95% CI high": hi})
    return pd.DataFrame(rows).set_index("measure")


def outcome_agreement(pairs: pd.DataFrame, snapshot: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Next-year check: when one building in a pair gets a problem, how often does its neighbor too?"""
    s = snapshot.reset_index(drop=True)
    p = pairs[mask]
    rows = []
    for t in TARGET_TYPES:
        y = s[f"y_{slug(t)}"].to_numpy()
        yi, yj = y[p["i"].to_numpy()], y[p["j"].to_numpy()]
        any_ = (yi + yj) > 0
        both = (yi + yj) == 2
        same = p["same_landlord"].to_numpy()
        rows.append({"problem": t,
                     "same landlord: neighbor also has it": both[same & any_].sum() / max(1, (same & any_).sum()),
                     "different landlord: neighbor also has it": both[~same & any_].sum() / max(1, (~same & any_).sum())})
    out = pd.DataFrame(rows).set_index("problem")
    out["ratio"] = out.iloc[:, 0] / out.iloc[:, 1]
    return out
