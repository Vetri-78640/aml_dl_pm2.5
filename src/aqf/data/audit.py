import numpy as np

from aqf.data.split import fold_mask


def gaps(miss):
    """Length and start time of every run of missing values."""
    run = miss.ne(miss.shift()).cumsum()
    g = miss.groupby(run).agg(["first", "size"])
    g["start"] = miss.index.to_series().groupby(run.values).first().values
    return g.loc[g["first"], ["size", "start"]]


def audit(df, folds, limit):
    """Summary numbers for the preprocessing report."""
    pm = df["pm25"]
    g = gaps(pm.isna())
    top = pm.nlargest(5)

    return {
        "rows": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "pm25_missing": int(pm.isna().sum()),
        "pm25_missing_pct": round(pm.isna().mean() * 100, 2),
        "pm25_missing_by_year": {int(k): int(v) for k, v in pm.isna().groupby(df.index.year).sum().items()},
        "gaps": {
            "count": len(g),
            "len_1": int((g["size"] == 1).sum()),
            "le_ffill_limit": int((g["size"] <= limit).sum()),
            "gt_24": int((g["size"] > 24).sum()),
            "longest": int(g["size"].max()),
            "longest_start": str(g.loc[g["size"].idxmax(), "start"]),
        },
        "pm25_in_missing_after_fill": int(df["pm25_in"].isna().sum()),
        "pm25_quantiles": {str(q): float(v) for q, v in pm.quantile([0, .5, .9, .99, 1]).items()},
        "pm25_top5": {str(k): float(v) for k, v in top.items()},
        "skew": {
            c: {"raw": round(float(df[c].skew()), 2), "log1p": round(float(np.log1p(df[c]).skew()), 2)}
            for c in ["pm25", "ws", "ws_cum", "snow_cum", "rain_cum"]
        },
        "ws": {k: round(float(v), 2) for k, v in df["ws"].describe().items()},
        "cbwd_counts": {str(k): int(v) for k, v in df["cbwd"].value_counts().items()},
        "folds": [
            {
                "name": f["name"],
                "train": [str(t) for t in f["train"]],
                "eval": [str(t) for t in f["eval"]],
                "train_rows": int(fold_mask(df.index, f["train"]).sum()),
                "eval_rows": int(fold_mask(df.index, f["eval"]).sum()),
                "eval_labels": int(pm[fold_mask(df.index, f["eval"])].notna().sum()),
            }
            for f in folds
        ],
    }
