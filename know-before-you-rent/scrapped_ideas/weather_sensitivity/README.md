# Scrapped idea: Weather Sensitivity ("Same Sky, Different Streets")

**Status:** explored and analyzed, then set aside for *Know Before You Rent*.

**The idea:** every NYC neighborhood gets the same weather, so if some areas' 311 complaints react much more to cold or rain, the difference comes from the buildings and infrastructure.

**What we found (2023–2025, 3.0M 311 requests):**
- 10°F colder than a normal day in the same month → **+68%** heat complaints
- +1 inch of rain → **+259%** sewer complaints
- ZIP codes differ a lot (+23% to +192% for the same cold), and neighbors are no more alike than random ZIPs (heat). Sewer sensitivity is spatially clustered.
- Pre-1940 buildings: ~4× heat complaints per apartment vs 2000+

**Why we moved on:** it was hard to explain as a product, and it overlapped with the team's other 311 analyses. Building age and the 100 m neighborhood carried over into Know Before You Rent.

## Contents
| Folder | What |
|---|---|
| `notebooks/` | 01–06: explore/clean 311, explore/clean weather + PLUTO, build tables, sensitivity analysis |
| `src/` | `analysis.py` (Poisson models, ZIP sensitivities, neighbor permutation test), `build_features.py` (ZIP × day panel) |
| `figures/` | Exported charts |
| `docs/` | Original plan, README and exploration notes |

To re-run: copy `src/*.py` back into the project's `src/` folder and run the notebooks from `notebooks/` in order (they use `src/config.py`, `download.py`, `preprocess.py`, `transform.py`, `plots.py` from the main project).
