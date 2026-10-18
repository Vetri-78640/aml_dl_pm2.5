import json

import numpy as np
import pandas as pd


class Scaler:
    """log1p (optional) then z-score, with statistics from training rows only."""

    def __init__(self, cols, log=(), eps=1e-6):
        self.cols = list(cols)
        self.log = [c for c in log if c in self.cols]
        self.eps = eps
        self.mean = {}
        self.std = {}

    def _pre(self, df):
        x = df[self.cols].astype("float64")
        for c in self.log:
            x[c] = np.log1p(x[c])
        return x

    def fit(self, df):
        x = self._pre(df)
        self.mean = x.mean().to_dict()
        self.std = x.std(ddof=0).clip(lower=self.eps).to_dict()
        return self

    def transform(self, df):
        x = (self._pre(df) - pd.Series(self.mean)) / pd.Series(self.std)
        return df.assign(**x)

    def inverse(self, z, col):
        x = np.asarray(z) * self.std[col] + self.mean[col]
        return np.expm1(x) if col in self.log else x

    def to_dict(self):
        return {"cols": self.cols, "log": self.log, "eps": self.eps,
                "mean": self.mean, "std": self.std}

    def save(self, path):
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text())
        s = cls(d["cols"], d["log"], d["eps"])
        s.mean, s.std = d["mean"], d["std"]
        return s
