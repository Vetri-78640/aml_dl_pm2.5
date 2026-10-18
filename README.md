# Air quality forecasting via gradient boosting and LSTM models

AML/DL Project 19 - Team T024, Section A.

Short-horizon PM2.5 forecasting on the UCI Beijing PM2.5 hourly dataset (2010 - 2014). Phase 1 is an XGBoost baseline on lag, rolling, calendar and weather features; Phase 2 is an LSTM or temporal CNN, with an optional hybrid. Models are compared with rolling chronological backtests (MAE, RMSE, R2 at a fixed horizon) against persistence and seasonal-naive baselines.

## Layout

```
configs/            run configuration (data.yaml)
data/
  raw/              original csv, never modified
  interim/          cleaned hourly frame (generated)
  processed/        folds and per-fold scalers (generated)
src/aqf/
  config.py         config loader, project root
  data/             load, clean, split, scale, audit, pipeline
scripts/            one-off scripts (report figures)
tests/              pytest suite for the data guarantees
notebooks/          exploration
reports/
  preprocessing/    LaTeX report, figures, qa.json
```

## Setup

```
pip install -e ".[dev]"
```

## Run

```
python -m aqf.data.pipeline            # raw csv -> data/interim, data/processed, reports/preprocessing/qa.json
python scripts/plot_preprocessing.py   # report figures
pytest                                 # data tests
cd reports/preprocessing && tectonic preprocessing.tex
```

## Preprocessing in short

- validate schema and a complete hourly grid; only `pm25` may be missing
- decode cumulative `Iws` into hourly wind speed `ws`; `Is` / `Ir` into `snow` / `rain` flags
- one-hot `cbwd` in a fixed order
- keep `pm25` as the untouched label, add `pm25_miss` and a causal input copy `pm25_in` (forward fill, at most 3 h)
- expanding-window folds: validate on 2011, 2012, 2013, test on 2014, 24 h embargo before each
- log1p + z-score scalers fitted on each fold's training rows only

Details and reasoning: `reports/preprocessing/preprocessing.pdf`.

## Data

Chen, S. (2015). Beijing PM2.5 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49 (CC BY 4.0).
