import numpy as np
import pandas as pd
import pytest

from aqf.config import load_config
from aqf.data import clean, fold_mask, load_raw, make_folds
from aqf.features import build, calendar, fold_xy
from aqf.models import baselines, fit_gate, fit_xgb, gate_blend

CFG = load_config("configs/data.yaml")
P1 = load_config("configs/phase1.yaml")


@pytest.fixture(scope="module")
def df():
    return clean(load_raw(CFG["paths"]["raw"]), CFG["clean"])


@pytest.fixture(scope="module")
def F(df):
    return build(df, P1["features"])[0]


@pytest.mark.parametrize("cut", ["2011-03-05 07:00", "2013-12-31 23:00"])
def test_features_are_causal(df, F, cut):
    part, _ = build(df.loc[:cut], P1["features"])
    pd.testing.assert_frame_equal(part, F.loc[:cut])


def test_groups_cover_every_column(df, F):
    _, groups = build(df, P1["features"])
    X = fold_xy(F, df, make_folds(df.index, **CFG["split"])[0], 6)[0]
    cols = [c for g in groups.values() for c in g]
    assert len(cols) == len(set(cols))
    assert sorted(cols) == sorted(X.columns)


def test_targets_and_embargo(df, F):
    for f in make_folds(df.index, **CFG["split"]):
        lo, hi = pd.Timestamp(f["eval"][0]), pd.Timestamp(f["eval"][1])
        for h in P1["sweep"]:
            X_tr, y_tr, X_ev, y_ev = fold_xy(F, df, f, h)
            step = pd.Timedelta(hours=h)
            assert (y_tr.index + step).max() < lo
            assert X_ev.index.min() >= lo
            assert (y_ev.index + step).max() < hi
            assert np.array_equal(y_ev.to_numpy(), df["pm25"].reindex(y_ev.index + step).to_numpy())
            assert y_tr.notna().all() and y_ev.notna().all()


def test_weather_inputs_lag_the_origin(df, F):
    lag = pd.Timedelta(hours=P1["features"]["weather_lag"])
    for c in ["dewp", "temp", "pres", "ws", "wd_nw"]:
        assert np.array_equal(F[c].iloc[1:].to_numpy(), df[c].reindex(F.index[1:] - lag).to_numpy())


def test_later_years_never_change_a_fold(df, F):
    for f in make_folds(df.index, **CFG["split"]):
        hi = pd.Timestamp(f["eval"][1])
        noisy = df.copy()
        later = df.index >= hi
        noisy.loc[later, ["pm25", "pm25_in", "temp", "pres"]] += 100
        Fn, _ = build(noisy, P1["features"])
        for h in P1["sweep"]:
            for a, b in zip(fold_xy(F, df, f, h), fold_xy(Fn, noisy, f, h)):
                assert a.equals(b)


def test_climatology_ignores_eval_labels(df, F):
    f = make_folds(df.index, **CFG["split"])[0]
    noisy = df.copy()
    noisy.loc[fold_mask(df.index, f["eval"]), "pm25"] += 500
    a, b = fold_xy(F, df, f, 6), fold_xy(F, noisy, f, 6)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_series_equal(a[2]["pm_anom"], b[2]["pm_anom"])


def test_baselines(df):
    b1, b24 = baselines(df, 1), baselines(df, 24)
    assert b1["persistence"].equals(df["pm25_in"])
    assert b24["seasonal"].equals(b24["persistence"])
    assert np.array_equal(b1["seasonal"].iloc[23:].to_numpy(), df["pm25_in"].iloc[:-23].to_numpy(),
                          equal_nan=True)


def test_calendar_flags_lunar_new_year():
    c = calendar(pd.to_datetime(["2012-01-22 20:00", "2012-01-23 01:00", "2012-03-01 01:00"]))
    assert c["lny"].tolist() == [1, 1, 0]


def test_stagnation_run_counts_hours(df, F):
    w = df.shift(P1["features"]["weather_lag"])
    stag = (w["ws"] < P1["features"]["stag_ws"]) & (w["rain"] == 0)
    run, expect = 0, []
    for s in stag.iloc[:3000]:
        run = run + 1 if s else 0
        expect.append(run)
    assert F["stag_run"].iloc[:3000].tolist() == expect


def test_change_target_is_not_capped_by_training_range():
    idx = pd.date_range("2012-01-01", periods=600, freq="h")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"pm_0": rng.uniform(0, 100, 600), "noise": rng.normal(size=600)}, index=idx)
    y = X["pm_0"] + 5
    params = {"n_estimators": 200, "learning_rate": 0.1, "early_stopping_rounds": 20}
    level, _ = fit_xgb(X, y, params, 100, 1, log=False)
    change, _ = fit_xgb(X, y, params, 100, 1, log=False, delta=True)
    new = pd.DataFrame({"pm_0": [500.0], "noise": [0.0]})
    assert level.predict(new)[0] < 110
    assert abs(change.predict(new)[0] - 505) < 1


def test_gate_learns_the_threshold():
    idx = pd.date_range("2012-01-01", periods=600, freq="h")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"pm_0": rng.uniform(0, 300, 600), "noise": rng.normal(size=600)}, index=idx)
    params = {"n_estimators": 200, "learning_rate": 0.1, "early_stopping_rounds": 20}
    m, _ = fit_gate(X, X["pm_0"], 150, params, 100, 1)
    p = m.predict_proba(pd.DataFrame({"pm_0": [50.0, 250.0], "noise": [0.0, 0.0]}))[:, 1]
    assert p[0] < 0.1 and p[1] > 0.9


def test_gate_blend_limits():
    normal, spike = np.array([80.0, 80.0]), np.array([200.0, 200.0])
    assert np.allclose(gate_blend(normal, spike, np.array([0.3, 0.9]), lam=0), normal)
    assert np.allclose(gate_blend(normal, spike, np.array([1.0, 0.5])), [200, 140])
