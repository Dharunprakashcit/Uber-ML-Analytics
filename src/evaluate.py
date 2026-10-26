"""Metrics: point accuracy, count deviance, surge-hour error, quantile loss."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error


def pinball(y, q_pred, tau):
    diff = y - q_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def point_metrics(y, pred, surge_mask):
    pred = np.clip(pred, 1e-6, None)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "Poisson deviance": mean_poisson_deviance(y, pred),
        "MAE surge hours": mean_absolute_error(y[surge_mask], pred[surge_mask]),
    }
