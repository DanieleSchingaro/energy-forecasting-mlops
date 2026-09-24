#test/test_evaluation.py

"""
Verifica split temporale e base naive.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from energy.evaluation import is_test, naive_forecast, split_date, reference_baseline

def _hourly(n:int)->pd.Series:
    idx=pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.Series(np.arange(float(n)), index=idx)

@pytest.mark.parametrize("horizon", [1,24])
def test_no_test_target_in_training(horizon:int)->None:
    y=_hourly(2000)
    split=split_date(y.index, test_months=1)

    test_mask=is_test(y.index, horizon, split)
    target_time=y.index+pd.Timedelta(hours=horizon)

    assert (target_time[~test_mask]<=split).all()
    assert (target_time[test_mask]>split).all()

def test_naive_forecast_uses_same_hour_one_season_before()->None:
    y=_hourly(500)
    horizon, season=1,24
 
    forecast=naive_forecast(y, season, horizon)
 
    t=y.index[100]
    target_time=t+pd.Timedelta(hours=horizon)
    assert forecast.loc[t]==y.loc[target_time-pd.Timedelta(hours=season)]

def test_naive_forecast_rejects_future_values()->None:
    with pytest.raises(ValueError):
        naive_forecast(_hourly(100), season=24, horizon=48)

def test_persistence_is_naive_with_season_one()->None:
    y=_hourly(50)
 
    forecast=naive_forecast(y, season=1, horizon=1)
 
    t=y.index[10]
    assert forecast.loc[t]==y.loc[t]

def test_reference_baseline_picks_the_strongest()->None:
    metrics={
        "test_mean_kw":1.0,
        "naive_1h":{"mae":0.34, "rmse":0.5},
        "naive_24h":{"mae":0.57, "rmse":0.84},
        "naive_168h":{"mae":0.59, "rmse":0.85},
    }
 
    assert reference_baseline(metrics)==("naive_1h", 0.34)
    assert reference_baseline(metrics, "naive_24h")==("naive_24h", 0.57)
 
    with pytest.raises(ValueError):
        reference_baseline(metrics, "naive_12h")