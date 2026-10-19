import numpy as np


def score(y, p):
    """MAE, RMSE, R2 and mean bias (prediction minus truth), in the units of y."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    e = p - y
    return {"mae": np.abs(e).mean(), "rmse": np.sqrt(np.mean(e ** 2)),
            "r2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), "bias": e.mean()}
