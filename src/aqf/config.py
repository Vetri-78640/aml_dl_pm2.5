from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text())
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg
