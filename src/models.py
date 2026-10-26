"""Baseline and advanced models.

Baselines:  historical mean by zone and hour-of-week; seasonal naive.
Advanced:   Poisson gradient boosting for the mean; quantile gradient boosting
            for the median and 90% / 95% upper bounds (surge planning).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = [
    "lag1", "lag2", "lag3", "lag24", "lag168", "roll24_mean",
    "neighbour_mean_t", "temp_future", "hour", "dow", "month",
    "is_weekend", "is_holiday",
]
QUANTILES = (0.05, 0.5, 0.9, 0.95)
SEED = 47


def _how(frame):
    return frame["target_ts"].dt.dayofweek * 24 + frame["target_ts"].dt.hour


def hist_mean(train, test):
    """Mean training count for the same zone and hour-of-week as the target."""
    ref = (train.assign(how=_how(train))
                .groupby(["zone", "how"])["target"].mean()
                .rename("pred").reset_index())
    t = test.assign(how=_how(test))
    merged = t[["zone", "how"]].merge(ref, on=["zone", "how"], how="left")
    return merged["pred"].fillna(train["target"].mean()).to_numpy()


def seasonal_naive(test):
    return test["snaive"].to_numpy()


def fit_poisson(X, y):
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.05, random_state=SEED)
    return model.fit(X, y)


def fit_quantiles(X, y):
    return {
        q: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
            random_state=SEED).fit(X, y)
        for q in QUANTILES
    }


def predict_quantiles(models_by_q, X):
    """Stack quantile predictions and sort each row so quantiles do not cross."""
    stacked = np.column_stack([models_by_q[q].predict(X) for q in QUANTILES])
    return np.clip(np.sort(stacked, axis=1), 0, None)
