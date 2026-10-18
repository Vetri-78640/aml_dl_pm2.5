import pandas as pd

RAW_COLS = ["No", "year", "month", "day", "hour", "pm2.5",
            "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]

NAMES = {
    "pm2.5": "pm25",
    "DEWP": "dewp",
    "TEMP": "temp",
    "PRES": "pres",
    "cbwd": "cbwd",
    "Iws": "ws_cum",
    "Is": "snow_cum",
    "Ir": "rain_cum",
}


def load_raw(path):
    """Read the UCI csv into an hourly frame indexed by timestamp.

    Fails loudly if the schema changes, the hourly grid has holes or
    duplicates, or any weather value is missing (only pm25 may be NA).
    """
    df = pd.read_csv(path)
    if list(df.columns) != RAW_COLS:
        raise ValueError(f"unexpected columns: {list(df.columns)}")

    ts = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.rename(columns=NAMES)[list(NAMES.values())]
    df.index = pd.DatetimeIndex(ts, name="ts")
    df = df.sort_index()

    grid = pd.date_range(df.index[0], df.index[-1], freq="h")
    if df.index.has_duplicates or not df.index.equals(grid):
        raise ValueError("timestamps are not a complete, unique hourly grid")

    met = df.drop(columns="pm25")
    if met.isna().any().any():
        raise ValueError(f"missing weather values: {met.isna().sum().to_dict()}")
    return df
