#src/energy/api/service.py

"""
Logica della previsione: dalla finestra di storico alla curva delle 24 ore.
Le feature vengono costruite con lo stesso modulo usato in training
(energy.features.build): e' il punto in cui si evita il training/serving skew,
e il motivo per cui quel modulo non dipende ne' da file ne' da database.
"""

from __future__ import annotations
from typing import Any
import pandas as pd
from sqlalchemy.engine import Engine
from energy.api.registry import ModelRegistry
from energy.db import coverage, read_window
from energy.features.build import build_features


class WindowUnavailable(Exception):
    """
    La finestra di storico non copre tutte le ore richieste.
    Rifiutare e' l'unica risposta corretta: con un buco, medie mobili e lag
    sarebbero calcolati su dati diversi da quelli visti in training, e la
    previsione sembrerebbe valida pur non essendolo.
    """

    def __init__(self, expected:int, found:int, missing:list[pd.Timestamp])->None:
        super().__init__(f"finestra incompleta: {found} ore su {expected}")
        self.expected=expected
        self.found=found
        self.missing=missing


class ForecastService:
    def __init__(self, engine:Engine, registry:ModelRegistry, params:dict[str, Any])->None:
        self.engine=engine
        self.registry=registry
        self.target=params["data"]["target"]
        self.lags=params["features"]["lags"]
        self.rolling_windows=params["features"]["rolling_windows"]
        self.max_horizon=params["features"]["max_horizon"]

    @property
    def window_hours(self)->int:
        """
        Ore di storico necessarie: il lag piu' lungo, piu' l'ora corrente.
        """
        return max(max(self.lags), max(self.rolling_windows)-1)+1

    def latest_timestamp(self)->pd.Timestamp|None:
        last=coverage(self.engine)["last"]
        return pd.Timestamp(last) if last else None

    def features_at(self, issued_at:pd.Timestamp)->pd.DataFrame:
        """
        Riga di feature relativa a `issued_at`, con la finestra verificata.
        """
        window=read_window(self.engine, issued_at, self.window_hours)
        expected=pd.date_range(
            issued_at-pd.Timedelta(hours=self.window_hours-1), issued_at, freq="h"
        )
        found=pd.DatetimeIndex(window["timestamp"])
        if len(found)!=len(expected) or not found.equals(expected):
            missing=expected.difference(found)
            raise WindowUnavailable(len(expected), len(found), list(missing[:24]))

        frame=window.set_index("timestamp").rename(columns={"kw": self.target})
        frame.index.name="timestamp"
        return build_features(frame, self.target, self.lags, self.rolling_windows).tail(1)

    def forecast(self, issued_at: pd.Timestamp, horizons:list[int]|None=None)->dict:
        features=self.features_at(issued_at)
        wanted=horizons or list(range(1, self.max_horizon + 1))

        points=[]
        for horizon in wanted:
            model=self.registry.get(horizon)
            points.append(
                {
                    "timestamp":issued_at+pd.Timedelta(hours=horizon),
                    "horizon":horizon,
                    "kw":round(model.predict(features), 4),
                    "model":model.flavor,
                    "model_version":model.version,
                }
            )

        return {
            "issued_at":issued_at,
            "window_hours":self.window_hours,
            "registered_model":self.registry.name,
            "points":points,
        }