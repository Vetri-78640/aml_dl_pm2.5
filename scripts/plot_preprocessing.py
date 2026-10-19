"""Save the preprocessing report figures to reports/preprocessing/figures."""
import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from aqf import viz
from aqf.config import ROOT, load_config


def main(cfg_path="configs/data.yaml"):
    cfg = load_config(cfg_path)
    df = pd.read_parquet(cfg["paths"]["interim"])
    folds = json.loads((cfg["paths"]["processed"] / "folds.json").read_text())
    out = ROOT / "reports" / "preprocessing" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    viz.use_style()
    figs = {
        "missing": viz.missing_by_month(df),
        "wind_decode": viz.wind_decode(df),
        "target_hist": viz.target_hist(df[df.index.year < cfg["split"]["test_year"]]),
        "folds": viz.fold_timeline(folds),
    }
    for name, fig in figs.items():
        fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
        plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:])
