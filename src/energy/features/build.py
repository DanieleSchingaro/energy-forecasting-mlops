#src/energy/features/build.py

"""
Costruzione delle feature.
La riga indicizzata a t contiene solo informazione disponibile al tempo t incluso,
e il target associato è y(t+h) con h>=1. Ne segue che il lag 0 e le finestre chiuse a t
sono legittimi e non costituiscono leakage.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def add_calendar_features(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    idx=out.index
    out["hour"]=idx.hour
    out["dayofweek"]=idx.dayofweek
    out["month"]=idx.month
    out["is_weekend"]=(idx.dayofweek>=5).astype(int)
    out["hour_sin"]=np.sin(2*np.pi*idx.hour/24)
    out["hour_cos"]=np.cos(2*np.pi*idx.hour/24)
    out["dow_sin"]=np.sin(2*np.pi*idx.dayofweek/7)
    out["dow_cos"]=np.cos(2*np.pi*idx.dayofweek/7)
    return out

def add_lag_features(df:pd.DataFrame, target:str, lags:list[int])->pd.DataFrame:
    out=df.copy()
    for lag in lags:
        out[f"lag_{lag}h"]=out[target].shift(lag)
    return out

def add_rolling_features(df:pd.DataFrame, target:str, windows:list[int])->pd.DataFrame:
    out=df.copy()
    for window in windows:
        rolled=out[target].rolling(window, min_periods=window)
        out[f"roll_mean_{window}h"]=rolled.mean()
        out[f"roll_std_{window}h"]=rolled.std()
        out[f"roll_min_{window}h"]=rolled.min()
        out[f"roll_max_{window}h"]=rolled.max()
    return out

def _check_hourly_index(index:pd.Index)->None:
    """
    Gli shift valgono come ore solo se l'indice è un orario regolare e completo.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("serve un DatetimeIndex")
    expected=pd.date_range(index.min(), index.max(), freq="h")
    if len(index)!=len(expected) or not index.equals(expected):
        raise ValueError("l'indice deve essere un orario completo: usa reindex e non dropna")

def build_features(
        df:pd.DataFrame, target:str, lags:list[int], rolling_windows:list[int]
)->pd.DataFrame:
    """
    Serie oraria grezza -> matrice feature senza target.
    """
    _check_hourly_index(df.index)
    out=add_calendar_features(df)
    out=add_lag_features(out, target, lags)
    out=add_rolling_features(out, target, rolling_windows)
    return out.drop(columns=[target])

def make_supervised(
        raw:pd.DataFrame,
        target:str,
        horizon:int,
        lags:list[int],
        rolling_windows:list[int],
)->tuple[pd.DataFrame, pd.Series]:
    """
    Restituisce (X,y) allineati per istante di emissione t, con y=y(t+horizon).
    Le righe incomplete vengono scartate: warm-up iniziale, ore mancanti nel dataset e ultime 
    'horizon' righe senza target.
    """
    if horizon<1:
        raise ValueError("horizon deve essere >=1")

    features=build_features(raw, target, lags, rolling_windows)
    y=raw[target].shift(-horizon).rename("y")
    frame=features.join(y).dropna()
    return frame.drop(columns=["y"]), frame["y"]