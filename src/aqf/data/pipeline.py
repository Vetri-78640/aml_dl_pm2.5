"""Raw csv -> cleaned hourly frame, chronological folds, train-only scalers.

usage: python -m aqf.data.pipeline [--config configs/data.yaml]
"""
import argparse
import json
import logging

from aqf.config import load_config
from aqf.data.audit import audit
from aqf.data.clean import clean
from aqf.data.load import load_raw
from aqf.data.scale import Scaler
from aqf.data.split import fold_mask, make_folds

log = logging.getLogger(__name__)


def fit_scalers(df, folds, scale_cfg):
    """One scaler per fold, fitted on that fold's training rows only."""
    return {f["name"]: Scaler(**scale_cfg).fit(df[fold_mask(df.index, f["train"])]) for f in folds}


def run(cfg):
    p = cfg["paths"]
    df = clean(load_raw(p["raw"]), cfg["clean"])
    folds = make_folds(df.index, **cfg["split"])

    p["interim"].parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p["interim"])
    log.info("hourly frame %s -> %s", df.shape, p["interim"])

    out = p["processed"]
    (out / "scalers").mkdir(parents=True, exist_ok=True)
    spans = [{"name": f["name"], "train": [str(t) for t in f["train"]],
              "eval": [str(t) for t in f["eval"]]} for f in folds]
    (out / "folds.json").write_text(json.dumps(spans, indent=2))

    for (name, sc), s in zip(fit_scalers(df, folds, cfg["scale"]).items(), spans):
        sc.save(out / "scalers" / f"{name}.json")
        log.info("fold %s: train %s -> %s, eval %s -> %s", s["name"], *s["train"], *s["eval"])

    p["qa"].parent.mkdir(parents=True, exist_ok=True)
    p["qa"].write_text(json.dumps(audit(df, folds, cfg["clean"]["ffill_limit"]), indent=2))
    log.info("qa summary -> %s", p["qa"])
    return df, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
