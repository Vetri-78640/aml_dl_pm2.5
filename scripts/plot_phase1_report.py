"""Save the Phase 1 report figures to reports/phase1/figures.

Figures come from the saved validation results in reports/phase1 and reports/moe.
The forecast window refits the frozen 6 h blend on fold cv3; the test fold is never built.
"""
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aqf import viz
from aqf.config import ROOT, load_config
from aqf.data.split import make_folds
from aqf.features import build, fold_xy
from aqf.models import baselines, fit_ridge, fit_xgb


def window(dcfg, cfg, h=6):
    df = pd.read_parquet(dcfg["paths"]["interim"])
    fold = next(f for f in make_folds(df.index, **dcfg["split"]) if f["name"] == "cv3")
    ch = json.loads((ROOT / "reports/phase1/cv_choices.json").read_text())
    X_tr, y_tr, X_ev, _ = fold_xy(build(df, cfg["features"])[0], df, fold, h)
    params = {**cfg["xgb"]["base"], **ch["xgb"][str(h)]}
    xs = np.mean([fit_xgb(X_tr, y_tr, params, cfg["xgb"]["tail_hours"], h, False, False, s)[0].predict(X_ev)
                  for s in cfg["ensemble_seeds"]], axis=0)
    rg = fit_ridge(X_tr, y_tr, ch["ridge"][str(h)]["alpha"], ch["ridge"][str(h)]["log_target"]).predict(X_ev)
    w = ch["blend"][str(h)]
    at = X_ev.index + pd.Timedelta(hours=h)
    preds = {"persistence": pd.Series(baselines(df, h).loc[X_ev.index, "persistence"].to_numpy(), index=at),
             "blend": pd.Series(w["xgb_seeds"] * xs + w["ridge"] * rg, index=at)}
    return viz.forecast_window(df["pm25"], preds, "2013-01-08", "2013-01-17")


def main():
    dcfg, cfg = load_config("configs/data.yaml"), load_config("configs/phase1.yaml")
    p1, moe = ROOT / "reports/phase1", ROOT / "reports/moe"
    out = p1 / "figures"
    out.mkdir(parents=True, exist_ok=True)
    abl = pd.read_csv(p1 / "cv_ablation.csv").query("study == 'features'")

    viz.use_style()
    figs = {
        "horizon": viz.horizon_r2(pd.read_csv(p1 / "cv_horizon_sweep.csv")),
        "ablation": viz.ablation_steps(abl),
        "calibration": viz.calibration(pd.read_csv(moe / "cv_gate_calibration.csv")),
        "window": window(dcfg, cfg),
    }
    for name, fig in figs.items():
        fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
