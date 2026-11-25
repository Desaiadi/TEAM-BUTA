"""Shared plotting style and helpers for the notebooks.

One seaborn theme and palette for every notebook chart; `save()` writes a
figure to reports/figures/ for the report and slides.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
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
