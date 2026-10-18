"""Figures for the preprocessing report (IEEE single-column width)."""
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from aqf.config import ROOT, load_config

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
W = 3.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlesize": 7,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def missing_by_month(df, out):
    m = df["pm25_miss"].resample("MS").sum()
    fig, ax = plt.subplots(figsize=(W, 1.5))
    ax.bar(m.index, m.values, width=24, color=BLUE, edgecolor="white", linewidth=0.3)
    top = m.idxmax()
    ax.annotate(f"{int(m[top])} h ({top:%b %Y})", (top, m[top]), xytext=(6, -2),
                textcoords="offset points", color=INK2, va="top")
    ax.set_ylabel("missing hours")
    ax.grid(axis="x", visible=False)
    fig.savefig(out)
    plt.close(fig)


def wind_decode(df, out):
    w = df.loc["2010-01-01 00:00":"2010-01-03 23:00"]
    run = w["cbwd"].ne(w["cbwd"].shift()).cumsum()
    starts = w.index[w["cbwd"].ne(w["cbwd"].shift())]
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 2.3), sharex=True)
    a.plot(w.index, w["ws_cum"], color=BLUE, lw=1.2, solid_joinstyle="round")
    b.bar(w.index, w["ws"], width=1 / 24 * 0.8, color=BLUE)
    for t in starts[1:]:
        for ax in (a, b):
            ax.axvline(t, color=AXIS, lw=0.5, zorder=0)
    for _, g in w.groupby(run):
        if len(g) >= 4:
            mid = g.index[len(g) // 2]
            a.text(mid, a.get_ylim()[1] * 0.98, g["cbwd"].iloc[0], ha="center", va="top", color=INK2)
    a.set_ylabel("Iws (cumulative)")
    b.set_ylabel("ws (hourly)")
    for ax in (a, b):
        ax.grid(axis="x", visible=False)
    b.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    b.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=[6, 12, 18]))
    b.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    fig.align_ylabels()
    fig.savefig(out)
    plt.close(fig)


def target_hist(df, skew, out):
    pm = df["pm25"].dropna()
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.5))
    a.hist(pm, bins=60, color=BLUE, edgecolor="white", linewidth=0.2)
    b.hist(np.log1p(pm), bins=60, color=BLUE, edgecolor="white", linewidth=0.2)
    a.set_title(f"pm2.5 (skew {skew['raw']})")
    b.set_title(f"log1p(pm2.5) (skew {skew['log1p']})")
    a.set_ylabel("hours")
    a.set_xlabel("µg/m³")
    b.set_xlabel("log1p(µg/m³)")
    for ax in (a, b):
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}K" if v else "0"))
    fig.savefig(out)
    plt.close(fig)


def fold_timeline(folds, out):
    fig, ax = plt.subplots(figsize=(W, 1.35))
    for i, f in enumerate(folds):
        y = len(folds) - 1 - i
        tr0, tr1 = (pd.Timestamp(t) for t in f["train"])
        ev0, ev1 = (pd.Timestamp(t) for t in f["eval"])
        color, tag = (AQUA, "test") if f["name"] == "test" else (ORANGE, "val")
        ax.barh(y, tr1 - tr0, left=tr0, height=0.55, color=BLUE)
        ax.barh(y, ev1 - ev0, left=ev0, height=0.55, color=color)
        ax.text(tr0 + (tr1 - tr0) / 2, y, "train", ha="center", va="center", color="white")
        ax.text(ev0 + (ev1 - ev0) / 2, y, f"{tag} {ev0.year}", ha="center", va="center", color=INK)
    ax.set_yticks(range(len(folds)), [f["name"] for f in reversed(folds)])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fig.savefig(out)
    plt.close(fig)


def main(cfg_path="configs/data.yaml"):
    cfg = load_config(cfg_path)
    df = pd.read_parquet(cfg["paths"]["interim"])
    qa = json.loads(cfg["paths"]["qa"].read_text())
    folds = json.loads((cfg["paths"]["processed"] / "folds.json").read_text())
    out = ROOT / "reports" / "preprocessing" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    missing_by_month(df, out / "missing.pdf")
    wind_decode(df, out / "wind_decode.pdf")
    target_hist(df, qa["skew"]["pm25"], out / "target_hist.pdf")
    fold_timeline(folds, out / "folds.pdf")


if __name__ == "__main__":
    main(*sys.argv[1:])
