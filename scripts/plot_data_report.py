"""Save the data report figures to reports/data/figures.

Figures come from the original csv (validated by load_raw, values untouched).
Pattern figures use the years before the test year only.
"""
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf

from aqf import viz
from aqf.config import ROOT, load_config
from aqf.data.clean import decode_wind
from aqf.data.load import load_raw
from aqf.data.split import make_folds


def main(cfg_path="configs/data.yaml"):
    cfg = load_config(cfg_path)
    df = decode_wind(load_raw(cfg["paths"]["raw"]))
    dev = df[df.index.year < cfg["split"]["test_year"]]
    out = ROOT / "reports" / "data" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    z = np.log1p(dev["pm25"])
    acf = pd.Series({k: z.autocorr(k) for k in range(1, 73)})
    zf = z.ffill().dropna()
    pa = pd.Series(pacf(zf, nlags=48)[1:], index=range(1, 49))

    viz.use_style()
    figs = {
        "missing": viz.missing_by_month(df.assign(pm25_miss=df["pm25"].isna().astype("int8"))),
        "wind_decode": viz.wind_decode(df),
        "target_hist": viz.target_hist(dev),
        "folds": viz.fold_timeline(make_folds(df.index, **cfg["split"])),
        "daily": viz.daily_series(dev),
        "diurnal": viz.diurnal_profile(dev),
        "monthly": viz.monthly_profile(dev),
        "acf_pacf": viz.acf_pacf(acf, pa, len(zf)),
        "wind_effect": viz.wind_effect(dev),
        "episode": viz.episode_example(dev, "2010-12-16", "2010-12-24"),
    }
    for name, fig in figs.items():
        fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
        plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:])
