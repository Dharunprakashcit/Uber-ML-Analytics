"""Synthetic hourly trip-request generator for a ride-hailing marketplace.

Produces hourly trip requests per city zone with commute peaks, weekend
nightlife demand, rain effects, holidays and occasional local events. It is a
stand-in for real trip data, which must be governed and anonymised before use
(see README).
"""
import numpy as np
import pandas as pd

ZONES = [f"Zone-{c}" for c in "ABCDEFGH"]
HOLIDAYS = {(1, 1), (7, 4), (12, 25)}


def generate(start="2023-01-01", end="2024-12-31 23:00", seed=47):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, end, freq="h")
    n = len(ts)
    hour = ts.hour.values
    dow = ts.dayofweek.values
    doy = ts.dayofyear.values

    # City-wide rain: wet days are more likely in winter; rain raises requests.
    day_index = ts.normalize()
    day_codes, unique_days = pd.factorize(day_index)
    day_doy = unique_days.dayofyear.values
    p_rain = 0.15 + 0.10 * np.cos(2 * np.pi * (day_doy - 15) / 365)
    rain_day = rng.random(len(unique_days)) < p_rain
    rain = rain_day[day_codes].astype(float)

    morning = np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)     # commute peak
    evening = np.exp(-0.5 * ((hour - 18) / 2.0) ** 2)    # commute peak
    night = np.exp(-0.5 * ((((hour - 1) + 12) % 24 - 12) / 2.0) ** 2)  # late night
    weekday_profile = 1 + 0.9 * morning + 1.1 * evening
    weekend_profile = 1 + 0.4 * morning + 0.5 * evening + 1.5 * night
    is_weekend = dow >= 5
    profile = np.where(is_weekend, weekend_profile, weekday_profile)
    rain_mult = 1 + 0.35 * rain
    season = 1 + 0.10 * np.cos(2 * np.pi * (doy - 15) / 365)
    holiday = np.array([1.25 if (m, d) in HOLIDAYS else 1.0
                        for m, d in zip(ts.month, ts.day)])

    base_rates = rng.uniform(4.0, 12.0, size=len(ZONES))
    frames = []
    for zone, base in zip(ZONES, base_rates):
        lam = base * profile * rain_mult * season * holiday
        # Rare local events: a 3-hour surge in one zone, about 20 per year.
        events = np.zeros(n)
        for start_idx in rng.choice(n, size=max(1, int(n / 8760 * 20)), replace=False):
            events[start_idx:start_idx + 3] += 1.0
        lam = lam * (1 + 0.8 * events)
        frames.append(pd.DataFrame({
            "zone": zone, "ts": ts, "temp": rain, "y": rng.poisson(lam),
        }))
    return pd.concat(frames, ignore_index=True)
