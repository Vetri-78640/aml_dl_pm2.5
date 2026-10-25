# Processed data

Generated, not tracked. Run `python -m aqf.data.pipeline` from the repository root to write:

| File | Contents |
|:--|:--|
| `folds.json` | train and evaluation spans of cv1, cv2, cv3 and test |
| `scalers/<fold>.json` | one scaler per fold, fitted on that fold's training rows only |

The fold design and the scaling are explained in the [data report](../../reports/data/data_report.pdf).
