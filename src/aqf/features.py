"""Features known at forecast origin t, targets at t + h, and per-fold datasets."""
import numpy as np
import pandas as pd

from aqf.data.split import fold_mask

LNY = pd.to_datetime(["2010-02-14", "2011-02-03", "2012-01-23", "2013-02-10", "2014-01-31"])
WEATHER = ["dewp", "temp", "pres", "ws", "ws_cum", "rain", "snow", "wd_ne", "wd_nw", "wd_se", "wd_cv"]
CALENDAR = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow", "weekend", "lny"]


def rel_humidity(temp, dewp):
    """Relative humidity in % from temperature and dew point (Magnus formula)."""
    a, b = 17.625, 243.04
    return 100 * np.exp(a * dewp / (b + dewp) - a * temp / (b + temp))


def calendar(ts):
    """Calendar of the timestamps `ts`. Known in advance, so it may describe the target hour."""
    ts = pd.DatetimeIndex(ts)
    day = ts.normalize()
    lny = day.isin(LNY) | day.isin(LNY - pd.Timedelta(days=1))
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "month_sin": np.sin(2 * np.pi * (ts.month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (ts.month - 1) / 12),
        "dow": ts.dayofweek,
        "weekend": (ts.dayofweek >= 5).astype("int8"),
        "lny": lny.astype("int8"),
    }, index=ts)


def build(df, cfg):
    """Features at each origin t from rows <= t only, plus their group names.

    Weather, derived and physics features use rows <= t - weather_lag. A lag of 1
    keeps every weather input strictly before the target hour even if the PM2.5
    and weather clocks are offset by one hour (open question Q15). Physics
    features follow the episode literature: stagnation (wind below `stag_ws`
    and no rain, the surface part of the air stagnation index), southerly
    transport (SE wind run) and humid build-up (relative humidity above
    `rh_high`).
    `pm_anom` (needs a training climatology) and the calendar (depends on h)
    are added per fold and horizon by `fold_xy`.
    """
    pm, miss = df["pm25_in"], df["pm25_miss"]
    lags = {"pm_0": pm, "miss_0": miss, "miss_24": miss.rolling(24, min_periods=1).sum()}
    lags |= {f"pm_lag{k}": pm.shift(k) for k in cfg["lags"]}

    rolling = {}
    for w in cfg["windows"]:
        r = pm.rolling(w, min_periods=1)
        rolling |= {f"pm_mean{w}": r.mean(), f"pm_std{w}": r.std(), f"pm_max{w}": r.max()}
    rolling |= {f"pm_diff{k}": pm - pm.shift(k) for k in cfg["diffs"]}

    w = df.shift(cfg["weather_lag"])
    weather = {c: w[c] for c in WEATHER}

    run = w["cbwd"].ne(w["cbwd"].shift()).cumsum()
    derived = {
        "rh": rel_humidity(w["temp"], w["dewp"]),
        "dir_hours": w.groupby(run).cumcount() + 1,
        "nw_ws24": (w["ws"] * w["wd_nw"]).rolling(cfg["nw_window"], min_periods=1).sum(),
    }
    derived |= {f"pres_diff{k}": w["pres"] - w["pres"].shift(k) for k in cfg["pres_diffs"]}

    n = cfg["phys_window"]
    stag = ((w["ws"] < cfg["stag_ws"]) & (w["rain"] == 0)).astype("int8")
    physics = {
        "stag24": stag.rolling(n, min_periods=1).sum(),
        "stag_run": stag.groupby(stag.ne(stag.shift()).cumsum()).cumsum(),
        "se_ws24": (w["ws"] * w["wd_se"]).rolling(n, min_periods=1).sum(),
        "humid24": (derived["rh"] > cfg["rh_high"]).astype("int8").rolling(n, min_periods=1).sum(),
    }

    F = pd.DataFrame(lags | rolling | weather | derived | physics)
    groups = {"lags": list(lags), "rolling": [*rolling, "pm_anom"], "calendar": CALENDAR,
              "weather": WEATHER, "derived": list(derived), "physics": list(physics)}
    return F, groups


def target(df, h):
    """pm25 at t + h for every origin t; NaN where it was not observed."""
    return df["pm25"].shift(-h)


def fold_xy(F, df, fold, h):
    """Train and eval sets for one fold and horizon.

    Train origins lie in the fold's train span, so with gap >= h every training
    target is observed before the eval year. Eval rows need both the origin and
    the target inside the eval year, so no label from a later year is scored.
    The hour-of-day climatology behind `pm_anom` uses training labels only. Eval
    rows need an observed target and both baselines, so every model is scored
    on the same rows.
    """
    tr = fold_mask(df.index, fold["train"])
    ev = fold_mask(df.index, fold["eval"]) & fold_mask(df.index + pd.Timedelta(hours=h), fold["eval"])
    hour = df.index.hour
    clim = df["pm25"][tr].groupby(hour[tr]).mean()
    cal = calendar(df.index + pd.Timedelta(hours=h)).set_axis(df.index)
    X = F.assign(pm_anom=df["pm25_in"].to_numpy() - clim.reindex(hour).to_numpy()).join(cal)
    y = target(df, h)
    pm = df["pm25_in"]
    ok = y.notna().to_numpy()
    base = (pm.notna() & pm.shift(24 - h).notna()).to_numpy()
    return X[tr & ok], y[tr & ok], X[ev & ok & base], y[ev & ok & base]
