"""Shared plotting style and helpers for the notebooks.

One seaborn theme and palette for every notebook chart; `save()` writes a
figure to reports/figures/ for the report and slides.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import config

FIG_DIR = config.REPORTS_DIR / "figures"
ACCENT = "#57068c"      # NYU violet
MUTED = "#9aa0a6"
sns.set_theme(style="whitegrid", context="notebook")


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG (200 dpi) under reports/figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


# --------------------------------------------------------------------------
# Street noise & sanitation (src/street.py, notebook 14)
# --------------------------------------------------------------------------
# Same colors as the app: building A teal, building B orange, the reference grey.
# A and B are close for some colour-blind readers, so B is always dashed or hatched too.
COLOR_A, COLOR_B, COLOR_REF = "#178570", "#d87844", "#8a9a94"
# Noise blue, sanitation gold: a validated pair, and neither is a building's color.
GROUP_COLORS = {"noise": "#2a78d6", "sanitation": "#d9a300"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plot_compare(pa: dict, pb: dict, name_a: str = "A", name_b: str = "B") -> plt.Figure:
    """Two buildings side by side: how much, when, and what (4 panels)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    groups = list(pa["groups"])

    # 1. Complaints per year, each place against its district's typical building
    ax = axes[0, 0]
    y = np.arange(len(groups))
    h = 0.26
    ax.barh(y - h, [pa["groups"][g]["per_year"] for g in groups], h, color=COLOR_A, label=name_a)
    ax.barh(y, [pb["groups"][g]["per_year"] for g in groups], h, color=COLOR_B, hatch="//",
            edgecolor="white", label=name_b)
    ax.barh(y + h, [pa["groups"][g]["district_median"] for g in groups], h, color=COLOR_REF,
            label="Typical building in A's district")
    ax.set_yticks(y, [g.capitalize() for g in groups])
    ax.invert_yaxis()
    ax.set_xlabel("Complaints per year within 100 m")
    ax.set_title("How many complaints?", loc="left", fontweight="bold")
    # One legend for the whole figure, above the panels, so it never covers a bar
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", ncol=3, frameon=False,
               bbox_to_anchor=(0.01, 1.0))

    # 2. When noise happens: share of each place's noise complaints by hour, noon to noon,
    #    so the night (10pm-7am) is one band in the middle
    ax = axes[0, 1]
    order = [(i + 12) % 24 for i in range(24)]
    for p, name, color, style in ((pa, name_a, COLOR_A, "-"), (pb, name_b, COLOR_B, "--")):
        hours = np.asarray(p["groups"]["noise"]["by_hour"], dtype=float)
        if hours.sum() >= 20:          # fewer complaints than this make an hourly profile meaningless
            ax.plot(range(24), hours[order] / hours.sum() * 100, style, color=color, lw=2, label=name)
        else:
            ax.plot([], [], style, color=color, lw=2, label=f"{name} (too few complaints to show)")
    ax.axvspan(9.5, 18.5, color=COLOR_REF, alpha=0.12, lw=0)        # 10pm to 6am, positions 10-18
    ax.text(14, ax.get_ylim()[1] * 0.95, "night", color="#4a5a55", va="top", ha="center")
    ax.set_xticks(range(0, 24, 3), [f"{order[i] % 12 or 12}{'am' if order[i] < 12 else 'pm'}" for i in range(0, 24, 3)])
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylabel("% of the place's noise complaints")
    ax.set_title("When is it noisy?", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    # 3-4. What the complaints are about
    for ax, g in zip(axes[1], groups):
        labels = list(dict.fromkeys([t["label"] for p in (pa, pb) for t in p["groups"][g]["top"]]))
        if not labels:
            ax.text(0.5, 0.5, f"No {g} complaints", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        counts = lambda p: {t["label"]: t["count"] / p_years(p) for t in p["groups"][g]["top"]}
        ca, cb = counts(pa), counts(pb)
        y = np.arange(len(labels))
        ax.barh(y - 0.18, [ca.get(l, 0) for l in labels], 0.36, color=COLOR_A, label=name_a)
        ax.barh(y + 0.18, [cb.get(l, 0) for l in labels], 0.36, color=COLOR_B, hatch="//",
                edgecolor="white", label=name_b)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Complaints per year within 100 m")
        ax.set_title(f"Most common {g} complaints", loc="left", fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.96))   # leave room for the figure legend
    return fig


def p_years(p: dict) -> float:
    """Years covered by a profile (count / per_year), for turning counts into yearly rates."""
    g = next(iter(p["groups"].values()))
    return g["count"] / g["per_year"] if g["per_year"] else 1.0


def plot_neighbor_gaps(findings: dict) -> plt.Figure:
    """Q2: gap between two cells by distance, against random pairs (one panel per group)."""
    gaps = findings["q2_neighbor_gaps"]
    fig, axes = plt.subplots(1, len(gaps), figsize=(12, 4.5), sharey=True)
    for ax, (g, r) in zip(np.atleast_1d(axes), gaps.items()):
        d = [row["distance_m"] for row in r["by_distance"]]
        ax.plot(d, [row["mean_gap"] for row in r["by_distance"]], "-o", color=GROUP_COLORS[g], lw=2, ms=6,
                label="Cells this far apart")
        ax.fill_between(d, [row["ci_low"] for row in r["by_distance"]], [row["ci_high"] for row in r["by_distance"]],
                        color=GROUP_COLORS[g], alpha=0.2, lw=0)
        for value, text, style in ((r["random_same_district"], "Random cells, same district", "--"),
                                   (r["random_citywide"], "Random cells, anywhere", ":")):
            ax.axhline(value, ls=style, color="#4a5a55", lw=1.5, label=text)
        ax.legend(frameon=False, loc="lower right")   # the lines can sit close together: a legend, not labels
        ax.set_title(g.capitalize(), loc="left", fontweight="bold")
        ax.set_xlabel("Distance between the two 100 m cells (m)")
    top = max(r["random_citywide"] for r in gaps.values())
    np.atleast_1d(axes)[0].set_ylim(0, top * 1.15)
    np.atleast_1d(axes)[0].set_ylabel("Mean gap, log(1 + complaints per year)")
    fig.suptitle("Nearby cells are more alike than random ones, but far from identical", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_hour_weekday(findings: dict, group: str) -> plt.Figure:
    """Q3: complaints by weekday and hour, as a heatmap (one hue, light = few, dark = many)."""
    grid = np.asarray(findings["q3_timing"][group]["hour_weekday"])
    fig, ax = plt.subplots(figsize=(12, 3.6))
    sns.heatmap(grid / 1000, ax=ax, cmap=sns.light_palette(GROUP_COLORS[group], as_cmap=True),
                cbar_kws={"label": "Complaints (thousands)"}, linewidths=0.5, linecolor="white")
    ax.set_yticklabels(WEEKDAYS, rotation=0)
    ax.set_xticks(np.arange(0, 24, 3) + 0.5, [f"{h}:00" for h in range(0, 24, 3)], rotation=0)
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("")
    ax.set_title(f"When {group} complaints are made", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_persistence(cells, group: str) -> plt.Figure:
    """Q4: each cell's complaints in Jan-Oct 2025 against Jan-Oct 2026 (log scales)."""
    a, b = cells[f"{group}_2025"], cells[f"{group}_2026"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(a + 1, b + 1, s=4, alpha=0.15, color=GROUP_COLORS[group], lw=0)
    top = max(a.max(), b.max()) + 1
    ax.plot([1, top], [1, top], color="#4a5a55", lw=1, ls="--")
    ax.text(top, top, "same in both years", ha="right", va="bottom", color="#4a5a55")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Complaints in the cell, Jan–Oct 2025 (+1)")
    ax.set_ylabel("Complaints in the cell, Jan–Oct 2026 (+1)")
    ax.set_title(f"{group.capitalize()} hotspots stay hotspots", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cell_map(cells, group: str, cd: int) -> plt.Figure:
    """The 100 m map: one community district, each cell colored by complaints per year."""
    sub = cells[cells["cd"] == cd]
    x0, y0 = sub["cx"].min(), sub["cy"].min()
    grid = np.full((sub["cy"].max() - y0 + 1, sub["cx"].max() - x0 + 1), np.nan)
    grid[sub["cy"] - y0, sub["cx"] - x0] = np.log10(1 + sub[f"{group}_per_year"])
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(grid, origin="lower", cmap=sns.light_palette(GROUP_COLORS[group], as_cmap=True))
    bar = fig.colorbar(im, ax=ax, shrink=0.7)
    values = [v for v in (0, 1, 3, 10, 30, 100, 300, 1000) if np.log10(1 + v) <= np.nanmax(grid)]
    bar.set_ticks(np.log10(1 + np.array(values)), labels=[f"{v:,}" for v in values])
    bar.set_label("Complaints per year in the cell")
    ax.set_axis_off()
    ax.set_title(f"{group.capitalize()} complaints per 100 m cell, {sub['district'].iloc[0]}",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    return fig
