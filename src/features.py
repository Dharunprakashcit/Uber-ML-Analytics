"""Feature construction. Every feature for an origin time t uses only data
available at t (counts up to and including hour t), except temp_future, which
assumes a perfect weather forecast for the target hour (see README caveats).
"""
import pandas as pd

HOLIDAYS = {(1, 1), (7, 4), (12, 25)}


def build(df, horizon):
    """Return one row per (zone, origin hour) with the target y(t + horizon)."""
    d = df.sort_values(["zone", "ts"]).reset_index(drop=True).copy()
    g = d.groupby("zone")

    for lag in (1, 2, 3, 24, 168):
        d[f"lag{lag}"] = g["y"].shift(lag)
    d["roll24_mean"] = g["y"].transform(lambda s: s.shift(1).rolling(24).mean())

    # Cross-zone context: mean count in the other zones at hour t.
    tot = d.groupby("ts")["y"].transform("sum")
    cnt = d.groupby("ts")["y"].transform("count")
    d["neighbour_mean_t"] = (tot - d["y"]) / (cnt - 1)

    d["snaive"] = g["y"].shift(168 - horizon)       # same hour, one week before target
    d["target"] = g["y"].shift(-horizon)
    d["temp_future"] = g["temp"].shift(-horizon)
    d["target_ts"] = d["ts"] + pd.Timedelta(hours=horizon)

    tt = d["target_ts"]
    d["hour"] = tt.dt.hour
    d["dow"] = tt.dt.dayofweek
    d["month"] = tt.dt.month
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["is_holiday"] = [int((m, dd) in HOLIDAYS) for m, dd in zip(tt.dt.month, tt.dt.day)]
    d["horizon"] = horizon

    needed = ["lag168", "roll24_mean", "target", "temp_future", "snaive"]
    return d.dropna(subset=needed).reset_index(drop=True)
