"""Feature modules, one per visibility tier (FR-2.6.3). Each module exposes
a TIER constant and a build(df, config) -> pd.DataFrame function returning
only the new feature columns (indexed like the input), so evaluate.py's
per-tier ablation (FR-8.8) can select subsets by tier without re-deriving
features already computed for another tier.
"""
