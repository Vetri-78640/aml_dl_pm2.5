# Model history at 6 h

How the 6-hour forecast got to where it is, step by step. 6 h is the headline horizon: at 1 h repeating the current value already explains most of the variation, and beyond 6 h XGBoost explains less than half of it (Phase 1 notebook, step 7).

All scores are pooled over the validation years 2011 to 2013 (folds cv1 to cv3) unless marked 2014. Units are µg/m³. Lower RMSE and MAE are better; R2 is the share of the variation explained. Skill is 1 - RMSE / RMSE of persistence. Bias is the mean of forecast minus truth.

## Summary

| Step | What changed | RMSE | MAE | R2 | Skill | Bias |
|:--|:--|--:|--:|--:|--:|--:|
| Persistence | forecast = PM2.5 now | 65.772 | 40.231 | 0.483 | - | 0.057 |
| Seasonal-naive | forecast = same hour yesterday | 101.417 | 69.424 | -0.230 | -54.2% | 0.147 |
| Run 1 | XGBoost (5 seeds) + Ridge blend, log1p target, chosen by MAE | 59.697 | 36.980 | 0.574 | 9.2% | -12.652 |
| Run 2 | raw XGBoost target | 56.686 | 36.103 | 0.616 | 13.8% | -2.631 |
| Runs 3 and 4 | chosen by RMSE, physics features, level or change target, spike expert tested | 56.330 | 36.410 | 0.621 | 14.4% | 1.634 |
| 2014 test | runs 3 and 4 settings, scored once | 51.963 | 33.930 | 0.690 | 15.1% | 2.341 |
| Mixture of experts | normal expert + spike expert + gate (`moe_split`) | 56.256 | 35.991 | 0.622 | 14.5% | 2.209 |

The 2014 row is the only test score. 2014 is an easier year (persistence R2 0.569 against 0.483 on 2011 to 2013), so compare it with the others through skill, not raw error.

## Run 1 - first model

[`notebooks/03_phase1_models.ipynb`](../notebooks/03_phase1_models.ipynb), first run.

- Features known at the forecast hour: PM2.5 lags, rolling statistics, calendar of the target hour, weather one hour earlier, derived weather.
- XGBoost tuned by a 20-candidate random search, averaged over 5 seeds, and blended with Ridge.
- Target log1p(PM2.5), models chosen by MAE.
- Problem: undoing log1p gives forecasts that are too low on average (blend bias -12.652).

## Run 2 - raw target

- XGBoost learns PM2.5 directly instead of log1p.
- Ridge may now choose log1p or raw together with its alpha; chosen by MAE, it kept log1p.
- RMSE 59.697 to 56.686, bias -12.652 to -2.631. The largest gain between runs.

## Runs 3 and 4 - selection by RMSE and physics features

- Every choice is made by RMSE: both models train on squared error, and on the same hours RMSE ranks models exactly as R2 does.
- Chosen by RMSE, Ridge switched to the raw target.
- Physics features from studies of Beijing winter episodes: stagnant hours (wind below 3.2 m/s and no rain), the current stagnant run, southerly wind run, humid hours (RH above 80%).
- Target form searched per horizon: level, or the change from PM2.5 now. At 6 h the level form won.
- Spike expert (XGBoost aiming at the 90th percentile) tested in the blend: best on episode hours (RMSE 75.968 against 87.084 for the blend) but far worse overall (82.700), so the blend gave it weight 0.
- Run 4 repeated the ablation as 5-seed means, because a single fit moved 0.74 RMSE from the order of its columns alone.
- Blend RMSE 56.686 to 56.330. MAE rose slightly (36.103 to 36.410) because selection now optimises RMSE.

### What each feature group adds (XGBoost, 5 seeds)

| Features | Columns | RMSE | MAE | R2 |
|:--|--:|--:|--:|--:|
| PM2.5 lags | 10 | 62.051 | 39.874 | 0.540 |
| + rolling statistics | 25 | 61.015 | 39.370 | 0.555 |
| + calendar | 32 | 60.157 | 38.802 | 0.567 |
| + weather | 43 | 58.021 | 36.832 | 0.597 |
| + derived weather | 48 | 57.170 | 36.523 | 0.609 |
| + physics | 52 | 56.808 | 36.456 | 0.614 |

### Target form (XGBoost, 5 seeds, all features)

| Target | RMSE | MAE | Bias |
|:--|--:|--:|--:|
| level, log1p | 59.965 | 37.025 | -12.628 |
| level, log1p + smearing correction | 58.106 | 37.195 | 7.099 |
| level, raw | 56.808 | 36.456 | 1.707 |
| change from now, raw | 58.055 | 36.418 | 0.891 |

## 2014 test

The runs 3 and 4 settings were retrained on 2010 to 2013 and scored once on 2014: RMSE 51.963 against 61.220 for persistence (15.1% better), MAE 33.930, R2 0.690. Skill on 2014 (15.1%) is close to validation (14.4%), so the gain carried over to unseen data. Phase 1 settings are frozen from here.

## Mixture of experts

[`notebooks/04_moe.ipynb`](../notebooks/04_moe.ipynb), validation years only. 2014 was already used for Phase 1, so it is not scored here.

**Why.** Hours above 150 are 21.2% of the validation hours but 50.6% of the blend's squared error, and the blend predicts them 55.296 too low on average. The Phase 1 spike expert failed because one fixed weight applied to every hour. A gate can weight it hour by hour.

| Part | Model | Trained on |
|:--|:--|:--|
| normal expert | the Phase 1 blend, unchanged | all training hours |
| spike expert | XGBoost, squared error, Phase 1 settings, 5 seeds | training hours with PM2.5 at t + 6 above 150 |
| gate | XGBoost classifier, same settings, 5 seeds: probability p that PM2.5 at t + 6 is above 150 | all training hours |

Combinations: `moe_gate` = normal + lambda * p * (spike - normal), lambda chosen on validation (0.1); `moe_split` = (1 - p) * quiet expert + p * spike, with the quiet expert trained on hours at or below 150.

| Model | RMSE | MAE | R2 | RMSE on episode hours |
|:--|--:|--:|--:|--:|
| normal (Phase 1 blend) | 56.330 | 36.410 | 0.621 | 87.084 |
| `moe_gate` | 56.312 | 36.401 | 0.621 | 85.267 |
| `moe_split` | 56.256 | 35.991 | 0.622 | 86.574 |
| `moe_split` with a hard switch at p = 0.5 | 61.241 | 38.117 | 0.552 | 89.833 |
| `moe_split` with the true labels as gate (not a forecast) | 39.457 | 27.286 | 0.814 | 64.875 |

**Gate quality.** AUC 0.923 (the PM2.5 level now alone: 0.904); episode starts 0.883 (0.846); episode ends 0.734 (0.645). Calibrated within 0.04 in every probability band.

**Seed check.** Rebuilt with three sets of 5 seeds:
- MAE of `moe_split` is lower than the normal expert's in every set, by 0.42 to 0.46. The normal expert itself moves by only 0.05 between sets, so this gain is real.
- RMSE drops by 0.005 to 0.202, within the 0.115 the normal expert moves between sets, so there is no clear RMSE gain.

**Result.** The mixture lowers MAE by 1.15% but does not clearly lower RMSE, the metric we select by. With perfect episode detection the same experts would reach RMSE 39.457, so the experts are good enough and the gate is the bottleneck. The gate sees only the hours up to the forecast; telling whether an episode will start or end in the next 6 hours needs more information, such as a longer history (an LSTM gate is a Phase 2 candidate) or a weather forecast.

## Files

| Step | Notebook | Results |
|:--|:--|:--|
| Runs 1 to 4, 2014 test | `notebooks/03_phase1_models.ipynb` | `reports/phase1/` |
| Mixture of experts | `notebooks/04_moe.ipynb` | `reports/moe/` |
