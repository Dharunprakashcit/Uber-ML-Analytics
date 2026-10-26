"""Smoke tests. Run with:  python -m pytest tests  (or: python tests/test_smoke.py)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import models  # noqa: E402
from src.data_gen import ZONES, generate  # noqa: E402
from src.evaluate import pinball  # noqa: E402
from src.features import build  # noqa: E402


def test_generate_shape_and_nonnegative():
    df = generate(start="2023-01-01", end="2023-02-28 23:00")
    assert set(df["zone"]) == set(ZONES)
    assert len(df) == len(ZONES) * 59 * 24
    assert (df["y"] >= 0).all()


def test_features_have_no_nans_and_no_future_leak():
    raw = generate(start="2023-01-01", end="2023-03-31 23:00")
    d = build(raw, 2)
    assert not d[models.FEATURES].isna().any().any()
    # target must be exactly two hours after the origin
    assert ((d["target_ts"] - d["ts"]) == pd.Timedelta(hours=2)).all()


def test_pinball_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball(y, y, 0.9) == 0.0


def test_quantiles_do_not_cross():
    raw = generate(start="2023-01-01", end="2023-03-31 23:00")
    d = build(raw, 1)
    qm = models.fit_quantiles(d[models.FEATURES], d["target"])
    Q = models.predict_quantiles(qm, d[models.FEATURES].head(500))
    assert (np.diff(Q, axis=1) >= 0).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
