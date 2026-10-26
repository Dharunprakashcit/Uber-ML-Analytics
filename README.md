# Uber ML Analytics: Trip Request Forecasting

Forecasts trip requests per city zone for 1 to 3 hours ahead, compares a baseline
ladder with gradient-boosting models, and writes a forecast table with 90% intervals.

**Important:** the bundled data is synthetic (`src/data_gen.py`). Results describe the
simulated process, not real trips. Real trip data contains personal information and
needs a governance review and anonymisation (aggregated counts per zone and hour)
before use.

## Setup

    pip install -r requirements.txt

## Run

    python run_all.py          # trains models, writes ./outputs
    python -m pytest tests     # smoke tests

## Layout

    src/data_gen.py   synthetic hourly trip requests per zone
    src/features.py   lags, rolling means, cross-zone context, calendar features
    src/models.py     baselines (historical mean, seasonal naive) and GBM models
    src/evaluate.py   MAE, RMSE, Poisson deviance, peak-hour MAE, pinball loss
    run_all.py        full pipeline and output writer
    tests/            smoke tests (shapes, no future leakage, quantile ordering)
    outputs/          metrics, zone errors, forecast table, figures

## Outputs

- `metrics_summary.csv`: metrics averaged over horizons 1 to 3, one row per model.
- `metrics_by_horizon.csv`: the same metrics for each horizon.
- `zone_errors.csv`: MAE per zone.
- `trip_forecast_example.csv`: expected requests and 90% interval per zone for the
  next three hours, from origin 2024-07-15 08:00.
- `fig_forecast_zone_A.png`, `fig_model_mae.png`: figures.

## Assumptions

- Features use request counts up to and including hour t. Targets are hours t+1 to t+3.
- The weather feature is a rain flag, measured at the target hour. It is not a forecast.
- Hyperparameters are fixed (no tuning). The single train/test split is 2023 vs 2024.
