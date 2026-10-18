<div align="center">

# PM2.5 forecasting with gradient boosting and LSTM models

**Short-horizon air quality forecasting on five years of hourly Beijing data**

[![Python](https://img.shields.io/badge/python-3.10%2B-2a78d6?style=flat-square)](pyproject.toml)
[![Phase 1](https://img.shields.io/badge/phase_1-XGBoost-eb6834?style=flat-square)](#roadmap)
[![Phase 2](https://img.shields.io/badge/phase_2-LSTM_%2F_TCN-1baf7a?style=flat-square)](#roadmap)
[![Data](https://img.shields.io/badge/data-UCI_Beijing_PM2.5-52514e?style=flat-square)](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data)
[![Tests](https://img.shields.io/badge/tests-pytest-52514e?style=flat-square)](tests/test_data.py)

AML/DL Project 19 - Team T024, Section A

[Overview](#overview) -
[Dataset](#dataset) -
[Pipeline](#preprocessing-pipeline) -
[Evaluation](#evaluation-protocol) -
[Quick start](#quick-start) -
[Structure](#repository-structure) -
[Roadmap](#roadmap) -
[Team](#team)

</div>

---

## Overview

This project forecasts the concentration of fine particulate matter (PM2.5) a short, fixed number of hours ahead from past pollution and weather, and asks a simple question: **how much does each modelling step add over just repeating the current value?**

| Stage | Approach | Status |
|:--|:--|:--|
| Data preprocessing | Leak-free pipeline, data audit, IEEE report, walkthrough notebook | Done |
| Phase 1 (ML) | XGBoost on lag, rolling, calendar and weather features vs persistence and seasonal-naive baselines | Next |
| Phase 2 (DL) | LSTM or temporal CNN on fixed windows, then an optional ML + DL hybrid | Planned |

Every model is scored with rolling chronological backtests on MAE, RMSE and R2 at a fixed horizon, on exactly the same hours.

---

## Dataset

[UCI Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data): hourly PM2.5 from the US Embassy in Beijing, with weather from Beijing Capital International Airport.

<table>
<tr><td>

| | |
|:--|--:|
| Hourly rows | 43,824 |
| Period | 2010-01-01 to 2014-12-31 |
| Target missing | 4.72% (2,067 h) |
| Weather missing | 0 |
| Target median / max | 72 / 994 ug/m3 |

</td><td>

| Variable | Meaning |
|:--|:--|
| `pm2.5` | PM2.5 concentration, target |
| `DEWP` / `TEMP` / `PRES` | dew point, temperature, pressure |
| `cbwd` | wind direction: NE, NW, SE, calm |
| `Iws` | cumulated wind speed |
| `Is` / `Ir` | cumulated hours of snow / rain |

</td></tr>
</table>

**What the audit found**

- **Strong persistence.** Autocorrelation is 0.97 at 1 hour, 0.40 at 24 hours and 0.02 at one week, so the seasonal-naive baseline uses a 24-hour lag.
- **Daily and seasonal cycles.** Highest around midnight and in winter (February mean 126), lowest in the afternoon and in May and August (mean 80).
- **Wind direction matters most.** Mean PM2.5 is 126 under calm wind and 70 under north-west wind.
- **Running totals, not readings.** `Iws`, `Is` and `Ir` accumulate until the wind direction or weather changes, so they are decoded before use.
- **Heavy right tail.** Skewness 1.8, reduced to -0.3 by log1p. The two highest values fall in the first hours of the Lunar New Year and are kept as real episodes.

---

## Preprocessing pipeline

```mermaid
flowchart LR
    A[raw csv] --> B[validate schema<br/>and hourly grid]
    B --> C[decode running totals<br/>Iws, Is, Ir]
    C --> D[one-hot<br/>wind direction]
    D --> E[label, missing flag,<br/>causal input copy]
    E --> F[(hourly.parquet)]
    F --> G[chronological folds<br/>24 h embargo]
    G --> H[(per-fold scalers<br/>train rows only)]
```

| Step | What it does | Why |
|:--|:--|:--|
| Validate | Exact columns, complete hourly grid, no missing weather | Fail loudly instead of repairing silently |
| Decode | `Iws` to hourly wind speed, `Is` / `Ir` to snow and rain flags | The raw columns are running totals |
| Encode | Four wind-direction columns in a fixed order | Same columns in every fold |
| Missing target | Label untouched, `pm25_miss` flag, `pm25_in` forward-filled for at most 3 h | Interpolation would use values after the gap |
| Folds | Validate on 2011, 2012, 2013; test on 2014 | Every fold sees a full seasonal cycle |
| Scale | log1p for skewed columns, then z-score from training rows only | No statistics from evaluation years |

Full reasoning, figures and numbers: [`reports/preprocessing/preprocessing.pdf`](reports/preprocessing/preprocessing.pdf).

---

## Evaluation protocol

| Fold | Train | Evaluate | Train hours | Labelled eval hours |
|:--|:--|:--|--:|--:|
| cv1 | 2010 | 2011 | 8,736 | 8,032 |
| cv2 | 2010 - 2011 | 2012 | 17,496 | 8,295 |
| cv3 | 2010 - 2012 | 2013 | 26,280 | 8,678 |
| test | 2010 - 2013 | 2014 | 35,040 | 8,661 |

Each training span stops 24 hours before its evaluation year, so no training target can fall inside it for horizons up to 24 hours. 2014 is scored once, at the end.

**Leakage guarantees, each checked by a test in [`tests/test_data.py`](tests/test_data.py)**

- the filled input at hour t is identical whether or not later hours exist
- folds never overlap and respect the embargo
- scaler statistics do not change when evaluation rows change
- the target label is never modified

---

## Quick start

```bash
git clone https://github.com/Vetri-78640/aml_dl_pm2.5.git
cd aml_dl_pm2.5
pip install -e ".[dev]"

python -m aqf.data.pipeline          # build data/interim and data/processed
pytest                               # run the data tests
```

| Task | Command |
|:--|:--|
| Regenerate report figures | `python scripts/plot_preprocessing.py` |
| Rebuild the report PDF | `cd reports/preprocessing && tectonic preprocessing.tex` |
| Re-run the walkthrough | `jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_preprocessing.ipynb` |

All settings live in [`configs/data.yaml`](configs/data.yaml): fill limit, fold years, embargo length and scaled columns.

**Outputs**

| Path | Contents |
|:--|:--|
| `data/interim/hourly.parquet` | cleaned hourly frame, 17 columns |
| `data/processed/folds.json` | fold boundaries |
| `data/processed/scalers/*.json` | one scaler per fold, fitted on training rows |
| `reports/preprocessing/qa.json` | audit numbers used in the report |

---

## Repository structure

```
aml_dl_pm2.5/
├── configs/
│   └── data.yaml                  pipeline settings
├── data/
│   ├── raw/                       original csv, never modified
│   ├── interim/                   cleaned hourly frame (generated)
│   └── processed/                 folds and per-fold scalers (generated)
├── notebooks/
│   └── 01_data_preprocessing.ipynb
├── reports/
│   └── preprocessing/             IEEE report, figures, audit summary
├── scripts/
│   └── plot_preprocessing.py      report figures
├── src/aqf/
│   ├── config.py                  config loader
│   ├── viz.py                     shared figures
│   └── data/                      load, clean, split, scale, audit, pipeline
├── tests/
│   └── test_data.py
└── pyproject.toml
```

---

## Roadmap

- [x] Data audit and leak-free preprocessing pipeline
- [x] Preprocessing report (IEEE format) and walkthrough notebook
- [ ] Exploratory data analysis notebook
- [ ] Feature engineering: lags, rolling statistics, calendar and weather features
- [ ] Persistence, 24-hour seasonal-naive and linear baselines
- [ ] XGBoost with chronological tuning and feature-group ablation
- [ ] Phase 1 report and presentation
- [ ] LSTM or temporal CNN on fixed windows
- [ ] ML + DL hybrid with ablation studies
- [ ] Phase 2 report and presentation

---

## Team

| Name |
|:--|
| Vetriselvan R |
| Hrishi Dongre |
| Pampana Sree Adithya |
| Ishan Goyal |
| Vedant Madne |

---

## References

1. S. Chen, "Beijing PM2.5," UCI Machine Learning Repository, 2015. [doi:10.24432/C5JS49](https://doi.org/10.24432/C5JS49). Licensed CC BY 4.0.
2. X. Liang et al., "Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating," *Proc. Royal Society A*, 2015.
3. R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., OTexts, 2021.
4. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.
5. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, 1997.
