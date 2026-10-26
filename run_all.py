"""Run the full pipeline: generate data, train models, evaluate, write outputs.

Usage:  python run_all.py
Outputs go to ./outputs (metrics CSVs, zone-level errors, a trip forecast table, figures).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import models
from src.data_gen import generate
from src.evaluate import pinball, point_metrics
from src.features import build

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
TRAIN_END = pd.Timestamp("2024-01-01")   # train on 2023 targets, test on 2024
HORIZONS = (1, 2, 3)
PLAN_ORIGIN = pd.Timestamp("2024-07-15 08:00")
QUANT_NAME = "Quantile GBM (median)"


def run():
    os.makedirs(OUT, exist_ok=True)
    raw = generate()
    metric_rows, zone_rows, plan_rows, series_rows = [], [], [], []

    for h in HORIZONS:
        d = build(raw, h)
        train = d[d["target_ts"] < TRAIN_END]
        test = d[d["ts"] >= TRAIN_END].reset_index(drop=True)
        y = test["target"].to_numpy()

        surge_thr = train.groupby("zone")["target"].quantile(0.9)
        surge = (test["target"] > test["zone"].map(surge_thr)).to_numpy()

        X_tr, y_tr = train[models.FEATURES], train["target"]
        X_te = test[models.FEATURES]

        preds = {
            "Historical mean": models.hist_mean(train, test),
            "Seasonal naive": models.seasonal_naive(test),
        }
        poisson = models.fit_poisson(X_tr, y_tr)
        preds["Poisson GBM"] = np.clip(poisson.predict(X_te), 0, None)

        qmodels = models.fit_quantiles(X_tr, y_tr)
        Q = models.predict_quantiles(qmodels, X_te)   # columns: q05, q50, q90, q95
        preds[QUANT_NAME] = Q[:, 1]

        for name, p in preds.items():
            row = {"model": name, "horizon": h, **point_metrics(y, p, surge),
                   "pinball 0.9": np.nan, "pinball 0.95": np.nan, "coverage 90%": np.nan}
            if name == QUANT_NAME:
                row["pinball 0.9"] = pinball(y, Q[:, 2], 0.9)
                row["pinball 0.95"] = pinball(y, Q[:, 3], 0.95)
                row["coverage 90%"] = float(np.mean((y >= Q[:, 0]) & (y <= Q[:, 3])))
            metric_rows.append(row)
            err = pd.Series(np.abs(y - p)).groupby(test["zone"].to_numpy()).mean()
            for zone, mae in err.items():
                zone_rows.append({"model": name, "horizon": h, "zone": zone, "MAE": mae})

        if h == 1:
            series_rows = pd.DataFrame({
                "ts": test["target_ts"], "zone": test["zone"], "actual": y,
                "pred_median": Q[:, 1], "lower90": Q[:, 0], "upper90": Q[:, 3]})

        at_origin = test[test["ts"] == PLAN_ORIGIN]
        idx = at_origin.index.to_numpy()
        for i, zone in zip(idx, at_origin["zone"]):
            plan_rows.append({
                "zone": zone, "hours_ahead": h,
                "target_time": (PLAN_ORIGIN + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M"),
                "actual_requests": int(y[i]), "expected_requests": round(float(Q[i, 1]), 2),
                "lower_90": round(float(Q[i, 0]), 2), "upper_90": round(float(Q[i, 3]), 2)})

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(OUT, "metrics_by_horizon.csv"), index=False)
    summary = (metrics.drop(columns="horizon").groupby("model", sort=False)
               .mean().reset_index())
    summary.to_csv(os.path.join(OUT, "metrics_summary.csv"), index=False)
    pd.DataFrame(zone_rows).to_csv(os.path.join(OUT, "zone_errors.csv"), index=False)
    pd.DataFrame(plan_rows).to_csv(os.path.join(OUT, "trip_forecast_example.csv"), index=False)

    plot_week(series_rows)
    plot_mae(summary)
    return summary


def plot_week(series):
    s = series[(series.zone == "Zone-A") &
               (series.ts >= "2024-07-08") & (series.ts < "2024-07-15")]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(s.ts, s.lower90, s.upper90, alpha=0.25, label="90% interval")
    ax.plot(s.ts, s.actual, color="#2b2d6e", lw=1, label="Actual requests")
    ax.plot(s.ts, s.pred_median, color="#e0a800", lw=1.5, label="Forecast (median, 1 h ahead)")
    ax.set_title("Zone-A, one week of 2024 (synthetic data)")
    ax.set_ylabel("Requests per hour")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_forecast_zone_A.png"), dpi=150)
    plt.close(fig)


def plot_mae(summary):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["model"], summary["MAE"], color=["#9b9bc9", "#9b9bc9", "#5b4fc9", "#2b2d6e"])
    ax.set_ylabel("MAE, requests per zone-hour (avg over 1-3 h)")
    ax.set_title("Model comparison on 2024 test data (synthetic)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_model_mae.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run()
    pd.set_option("display.width", 200)
    print(result.round(3).to_string(index=False))
    print("\nOutputs written to", OUT)
