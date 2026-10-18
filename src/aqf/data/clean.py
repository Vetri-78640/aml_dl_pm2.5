import pandas as pd


def decode_wind(df):
    """Recover hourly wind speed from the cumulative Iws column.

    Iws accumulates while the wind direction stays the same and restarts
    when cbwd changes, so the hourly value is the first-difference inside
    a direction run and the raw value on the first hour of a run.
    """
    new_run = df["cbwd"].ne(df["cbwd"].shift())
    ws = df["ws_cum"].where(new_run, df["ws_cum"].diff())
    if (ws <= 0).any():
        raise ValueError("ws_cum does not increase within a direction run")
    return df.assign(ws=ws)


def decode_precip(df):
    """Is / Ir count consecutive hours of snow / rain; > 0 means it is falling now."""
    return df.assign(snow=(df["snow_cum"] > 0).astype("int8"),
                     rain=(df["rain_cum"] > 0).astype("int8"))


def encode_dir(df, dirs):
    bad = set(df["cbwd"].unique()) - set(dirs)
    if bad:
        raise ValueError(f"unknown wind directions: {bad}")
    hot = {f"wd_{d.lower()}": df["cbwd"].eq(d).astype("int8") for d in dirs}
    out = df.assign(**hot)
    out["cbwd"] = pd.Categorical(out["cbwd"], categories=dirs)
    return out


def fill_target(df, limit):
    """Keep pm25 as the untouched label and add a causal input copy.

    pm25_in carries the last observed value forward for at most `limit`
    hours. Forward fill only looks at the past, unlike interpolation, which
    would use values after the gap and leak them into earlier inputs.
    """
    miss = df["pm25"].isna()
    return df.assign(pm25_miss=miss.astype("int8"),
                     pm25_in=df["pm25"].ffill(limit=limit))


def clean(df, cfg):
    df = decode_wind(df)
    df = decode_precip(df)
    df = encode_dir(df, cfg["wind_dirs"])
    return fill_target(df, cfg["ffill_limit"])
