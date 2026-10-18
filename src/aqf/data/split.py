import pandas as pd


def make_folds(index, val_years, test_year, gap_hours):
    """Expanding-window chronological folds.

    Each fold trains on everything from the first timestamp up to `gap_hours`
    before its evaluation year and evaluates on that whole year. The gap is
    an embargo: with gap >= forecast horizon, no training target can fall
    inside the evaluation period. Intervals are half-open [start, end).
    """
    years = [*val_years, test_year]
    if years != sorted(set(years)):
        raise ValueError("val_years must be increasing and before test_year")
    if gap_hours < 0:
        raise ValueError("gap_hours must be >= 0")

    gap = pd.Timedelta(hours=gap_hours)
    start, end = index[0], index[-1] + pd.Timedelta(hours=1)
    folds = []
    for i, y in enumerate(years):
        lo = pd.Timestamp(year=y, month=1, day=1)
        hi = min(pd.Timestamp(year=y + 1, month=1, day=1), end)
        if lo - gap <= start or lo >= end:
            raise ValueError(f"year {y} leaves no training data or is outside the series")
        folds.append({
            "name": "test" if y == test_year else f"cv{i + 1}",
            "train": [start, lo - gap],
            "eval": [lo, hi],
        })
    return folds


def fold_mask(index, span):
    lo, hi = pd.Timestamp(span[0]), pd.Timestamp(span[1])
    return (index >= lo) & (index < hi)
