import numpy as np
import pandas as pd
import pytest

from aqf.config import load_config
from aqf.data import Scaler, clean, fold_mask, load_raw, make_folds
from aqf.data.clean import fill_target
from aqf.data.pipeline import fit_scalers

CFG = load_config("configs/data.yaml")


@pytest.fixture(scope="module")
def df():
    return clean(load_raw(CFG["paths"]["raw"]), CFG["clean"])


def test_hourly_grid(df):
    assert len(df) == 43824
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()


def test_only_target_missing(df):
    assert df.drop(columns=["pm25", "pm25_in"]).notna().all().all()


def test_wind_decode_rebuilds_cumulative(df):
    run = df["cbwd"].ne(df["cbwd"].shift()).cumsum()
    rebuilt = df.groupby(run)["ws"].cumsum()
    assert np.allclose(rebuilt, df["ws_cum"], atol=1e-6)
    assert (df["ws"] > 0).all()


def test_one_hot_is_exclusive(df):
    hot = df[[c for c in df.columns if c.startswith("wd_")]]
    assert (hot.sum(axis=1) == 1).all()


def test_missing_flag_and_label_untouched(df):
    raw = load_raw(CFG["paths"]["raw"])
    assert (df["pm25_miss"] == raw["pm25"].isna()).all()
    assert df["pm25"].equals(raw["pm25"])


def test_fill_is_causal():
    s = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0])
    idx = pd.date_range("2020-01-01", periods=len(s), freq="h")
    full = fill_target(pd.DataFrame({"pm25": s.values}, index=idx), limit=3)["pm25_in"]
    for t in range(len(s)):
        part = fill_target(pd.DataFrame({"pm25": s.values[:t + 1]}, index=idx[:t + 1]), limit=3)["pm25_in"]
        assert part.equals(full.iloc[:t + 1])
    assert full.isna().sum() == 1


def test_folds_are_chronological_with_gap(df):
    folds = make_folds(df.index, **CFG["split"])
    gap = pd.Timedelta(hours=CFG["split"]["gap_hours"])
    for f in folds:
        tr, ev = fold_mask(df.index, f["train"]), fold_mask(df.index, f["eval"])
        assert not (tr & ev).any()
        assert df.index[ev][0] - df.index[tr][-1] > gap
    assert folds[-1]["name"] == "test"
    for f in folds[:-1]:
        assert not fold_mask(df.index, f["train"])[fold_mask(df.index, folds[-1]["eval"])].any()


def test_scalers_ignore_eval_and_later_rows(df):
    folds = make_folds(df.index, **CFG["split"])
    a = fit_scalers(df, folds, CFG["scale"])
    for f in folds:
        noisy = df.copy()
        later = df.index >= pd.Timestamp(f["train"][1])
        noisy.loc[later, ["temp", "pm25", "pm25_in"]] += 100
        b = fit_scalers(noisy, [f], CFG["scale"])[f["name"]]
        assert a[f["name"]].mean == b.mean and a[f["name"]].std == b.std


def test_scaler_round_trip(df, tmp_path):
    s = Scaler(**CFG["scale"]).fit(df)
    s.save(tmp_path / "s.json")
    s = Scaler.load(tmp_path / "s.json")
    z = s.transform(df)
    for c in ["pm25", "temp", "ws_cum"]:
        assert np.allclose(s.inverse(z[c], c), df[c], equal_nan=True)
