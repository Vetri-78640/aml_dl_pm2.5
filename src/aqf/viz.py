"""Report figures, sized for an IEEE single column. Each returns a Figure."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
W = 3.5

STYLE = {
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
}


def use_style():
    plt.rcParams.update(STYLE)


def legend_top(ax, n):
    ax.legend(frameon=False, ncol=n, loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0.2)


def missing_by_month(df):
    m = df["pm25_miss"].resample("MS").sum()
    fig, ax = plt.subplots(figsize=(W, 1.5))
    ax.bar(m.index, m.values, width=24, color=BLUE, edgecolor="white", linewidth=0.3)
    top = m.idxmax()
    ax.annotate(f"{int(m[top])} h ({top:%b %Y})", (top, m[top]), xytext=(6, -2),
                textcoords="offset points", color=INK2, va="top")
    ax.set_ylabel("missing hours")
    ax.grid(axis="x", visible=False)
    return fig


def wind_decode(df, start="2010-01-01 00:00", end="2010-01-03 23:00"):
    w = df.loc[start:end]
    new = w["cbwd"].ne(w["cbwd"].shift())
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 2.3), sharex=True)
    a.plot(w.index, w["ws_cum"], color=BLUE, lw=1.2, solid_joinstyle="round")
    b.bar(w.index, w["ws"], width=1 / 24 * 0.8, color=BLUE)
    for t in w.index[new][1:]:
        for ax in (a, b):
            ax.axvline(t, color=AXIS, lw=0.5, zorder=0)
    for _, g in w.groupby(new.cumsum()):
        if len(g) >= 4:
            mid = g.index[len(g) // 2]
            a.text(mid, a.get_ylim()[1] * 0.98, g["cbwd"].iloc[0], ha="center", va="top", color=INK2)
    a.set_ylabel("Iws (cumulative)")
    b.set_ylabel("ws (hourly)")
    for ax in (a, b):
        ax.grid(axis="x", visible=False)
    b.xaxis.set_major_locator(mdates.DayLocator())
    b.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.align_ylabels()
    return fig


def target_hist(df):
    pm = df["pm25"].dropna()
    lg = np.log1p(pm)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.5))
    a.hist(pm, bins=60, color=BLUE, edgecolor="white", linewidth=0.2)
    b.hist(lg, bins=60, color=BLUE, edgecolor="white", linewidth=0.2)
    a.set_title(f"pm2.5 (skew {pm.skew():.1f})")
    b.set_title(f"log1p(pm2.5) (skew {lg.skew():.1f})")
    a.set_ylabel("hours")
    a.set_xlabel("µg/m³")
    b.set_xlabel("log1p(µg/m³)")
    for ax in (a, b):
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:g}K" if v else "0"))
    return fig


def fold_timeline(folds):
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
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return fig


def daily_series(df):
    d = df["pm25"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(W, 1.6))
    ax.plot(d, color=AXIS, lw=0.5, label="daily mean")
    ax.plot(d.rolling(30, center=True, min_periods=15).mean(), color=BLUE, lw=1.2, label="30-day mean")
    ax.set_ylabel("pm2.5 (µg/m³)")
    legend_top(ax, 2)
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return fig


def diurnal_profile(df):
    pm, month = df["pm25"], df.index.month
    fig, ax = plt.subplots(figsize=(W, 1.6))
    for name, months, c in [("winter (DJF)", [12, 1, 2], BLUE), ("summer (JJA)", [6, 7, 8], ORANGE)]:
        s = pm[month.isin(months)]
        ax.plot(range(24), s.groupby(s.index.hour).median(), color=c, lw=1.2, marker="o", ms=2, label=name)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("median pm2.5 (µg/m³)")
    legend_top(ax, 2)
    return fig


def monthly_profile(df):
    g = df["pm25"].groupby(df.index.month)
    fig, ax = plt.subplots(figsize=(W, 1.6))
    ax.bar(range(1, 13), g.mean(), width=0.6, color=BLUE, label="mean")
    ax.plot(range(1, 13), g.median(), "o", color=ORANGE, ms=3, label="median")
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("pm2.5 (µg/m³)")
    legend_top(ax, 2)
    ax.grid(axis="x", visible=False)
    return fig


def acf_pacf(acf, pacf, n):
    """Bars per lag with the 95% band for white noise, 1.96 / sqrt(n)."""
    band = 1.96 / np.sqrt(n)
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 2.4))
    for ax, s, name in [(a, acf, "ACF"), (b, pacf, "PACF")]:
        ax.bar(s.index, s.values, width=0.7, color=BLUE)
        ax.axhspan(-band, band, color=GRID, zorder=0)
        ax.set_ylabel(name)
        ax.set_xticks(range(0, int(s.index.max()) + 1, 6))
        ax.grid(axis="x", visible=False)
    b.set_xlabel("lag (hours)")
    fig.subplots_adjust(hspace=0.45)
    fig.align_ylabels()
    return fig


def wind_effect(df):
    pm = df["pm25"]
    by_dir = pm.groupby(df["cbwd"], observed=True).median().sort_values()
    by_ws = pm.groupby(pd.cut(df["ws"], [0, 1, 2, 3, 4, 6, 8, 12, 25]), observed=True).median()
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.7), sharey=True)
    a.bar(by_dir.index.astype(str), by_dir.values, width=0.6, color=BLUE)
    a.set_xlabel("wind direction")
    a.set_ylabel("median pm2.5 (µg/m³)")
    b.plot(range(len(by_ws)), by_ws.values, color=BLUE, lw=1.2, marker="o", ms=2)
    b.set_xticks(range(len(by_ws)), [f"{i.left:g}-{i.right:g}" for i in by_ws.index], rotation=45)
    b.set_xlabel("wind speed (m/s)")
    for ax in (a, b):
        ax.grid(axis="x", visible=False)
    return fig


def episode_example(df, start, end):
    """pm25 over one episode with NW-wind hours shaded."""
    w = df.loc[start:end]
    fig, ax = plt.subplots(figsize=(W, 1.6))
    ax.fill_between(w.index, 0, 1, where=w["cbwd"].eq("NW").to_numpy(), transform=ax.get_xaxis_transform(),
                    color=GRID, step="mid", lw=0, label="NW wind")
    ax.plot(w.index, w["pm25"], color=BLUE, lw=1.0, label="pm2.5")
    ax.set_ylabel("pm2.5 (µg/m³)")
    legend_top(ax, 2)
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    return fig
