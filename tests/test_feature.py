#tests/test_feature.py

"""
Verifica che le feature a t non dipendano dal futuro.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from energy.features.build import build_features, make_supervised

LAGS=[0,1,24]
WINDOWS=[24]
TARGET="y"

def _series(values:np.ndarray)->pd.DataFrame:
    idx=pd.date_range("2020-01-01", periods=len(values), freq="h")
    return pd.DataFrame({TARGET:values}, index=idx)

def test_features_do_not_depend_on_future()->None:
    cut=300
    base=np.arange(500.0)
    perturbed=base.copy()
    perturbed[cut:]=-999.0

    features_base=build_features(_series(base), TARGET, LAGS, WINDOWS)
    features_perturbed=build_features(_series(perturbed), TARGET, LAGS, WINDOWS)

    pd.testing.assert_frame_equal(features_base.iloc[:cut], features_perturbed.iloc[:cut])

def test_target_is_shifted_forward()->None:
    raw=_series(np.arange(500.0))
    horizon=6

    X,y=make_supervised(raw, TARGET, horizon, LAGS, WINDOWS)

    timestamp=X.index[0]
    assert y.loc[timestamp]==raw[TARGET].loc[timestamp+pd.Timedelta(hours=horizon)]
    assert X.loc[timestamp, "lag_0h"]==raw[TARGET].loc[timestamp]

def test_no_nan_after_supervised()->None:
    values=np.arange(500.0)
    values[200:210]=np.nan
    X,y=make_supervised(_series(values), TARGET, 1, LAGS, WINDOWS)

    assert not X.isna().any().any()
    assert not y.isna().any()
    assert X.index.equals(y.index)

def test_rejects_index_with_gaps()->None:
    raw=_series(np.arange(500.0)).drop(index=pd.Timestamp("2020-01-05 10:00"))

    with pytest.raises(ValueError):
        build_features(raw, TARGET, LAGS, WINDOWS)