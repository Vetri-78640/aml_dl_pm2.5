"""Baselines, Ridge and XGBoost for direct multi-horizon forecasting.

Fitted models are wrapped in `Forecaster`, which undoes the target transform,
so predictions are always in ug/m3.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

LEVELS = ("pm_0", "pm_lag", "pm_mean", "pm_max", "pm_std")


class Forecaster:
    def __init__(self, model, log, smear=1.0, delta=False):
        self.model, self.log, self.smear, self.delta = model, log, smear, delta

    def predict(self, X):
        z = self.model.predict(X)
        if self.delta:
            z = z + anchor(X, self.log)
        p = np.exp(z) * self.smear - 1 if self.log else z
        return np.clip(p, 0, None)


def anchor(X, log):
    """PM2.5 at the origin on the target's scale; a change target is added to it."""
    pm = X["pm_0"].to_numpy()
    return np.log1p(pm) if log else pm


def baselines(df, h):
    """Persistence y_t and 24 h seasonal-naive y_(t+h-24), from the causal input copy."""
    if not 1 <= h <= 24:
        raise ValueError("baselines need 1 <= h <= 24")
    pm = df["pm25_in"]
    return pd.DataFrame({"persistence": pm, "seasonal": pm.shift(24 - h)})


def log_levels(X):
    """log1p on PM level columns so a linear model sees them on the target's scale."""
    cols = [c for c in X.columns if c.startswith(LEVELS)]
    return X.assign(**np.log1p(X[cols]))


def fit_ridge(X, y, alpha, log=True):
    m = make_pipeline(FunctionTransformer(log_levels), SimpleImputer(strategy="median"),
                      StandardScaler(), Ridge(alpha=alpha))
    m.fit(X, np.log1p(y) if log else y)
    return Forecaster(m, log)


def fit_xgb(X, y, params, tail_hours, h, log=True, smear=False, seed=0, delta=False):
    """Early-stop on the last `tail_hours` of the training span, then refit on all of it.

    Fitting rows end h hours before the tail, so no fitting target falls inside
    it. With `smear`, Duan's factor mean(exp(residual)) on the tail corrects the
    downward bias of undoing log1p. With `delta`, the model learns the change
    from PM2.5 at the origin, so forecasts are not capped by the largest
    training target; rows without PM2.5 at the origin are dropped. Returns the
    model and its tree count.
    """
    z = np.log1p(y) if log else y
    if delta:
        z = z - anchor(X, log)
        keep = z.notna().to_numpy()
        X, z = X[keep], z[keep]
    cut = X.index[-1] - pd.Timedelta(hours=tail_hours)
    fit = X.index <= cut - pd.Timedelta(hours=h)
    tail = X.index > cut
    p = {**params, "random_state": seed}
    es = xgb.XGBRegressor(**p).fit(X[fit], z[fit], eval_set=[(X[tail], z[tail])], verbose=False)
    n = es.best_iteration + 1
    s = float(np.mean(np.exp(z[tail] - es.predict(X[tail])))) if log and smear else 1.0
    p = {k: v for k, v in p.items() if k != "early_stopping_rounds"}
    full = xgb.XGBRegressor(**{**p, "n_estimators": n}).fit(X, z, verbose=False)
    return Forecaster(full, log, s, delta), n


def sample_params(space, n, seed):
    """n random draws from a grid given as {param: [values]}."""
    rng = np.random.default_rng(seed)
    return [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(n)]
