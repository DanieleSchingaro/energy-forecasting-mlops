#tests/test_api.py

"""
Verifica dell'API con database in memoria e modelli finti.
I modelli veri vivono nel registry di MLflow: qui interessa il comportamento
del servizio, cioe' che la finestra venga validata, che le feature arrivino ai
modelli e che la risposta dichiari quale versione ha prodotto ogni valore.
"""

from __future__ import annotations
from collections.abc import Iterator
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from energy.api.main import app, get_service
from energy.api.registry import LoadedModel
from energy.api.service import ForecastService, WindowUnavailable
from energy.db import create_schema, upsert_measurements

PARAMS={
    "data":{"target": "global_active_power"},
    "features":{"max_horizon": 3, "lags": [0, 1, 24], "rolling_windows": [24]},
}
LAST=pd.Timestamp("2010-01-10 00:00")


class FakeRegistry:
    """
    Restituisce un modello che somma l'orizzonte all'ultimo valore osservato.
    """

    name="energy-forecaster"

    def __init__(self)->None:
        self.requested:list[int]=[]

    def get(self, horizon:int)->LoadedModel:
        self.requested.append(horizon)
        return LoadedModel(
            horizon=horizon,
            predictor=_Predictor(horizon),
            version=str(horizon),
            run_id=f"run-{horizon}",
            alias=f"champion-h{horizon}",
            flavor="xgboost",
        )

    def describe(self)->list[dict]:
        return [{"horizon": h} for h in sorted(set(self.requested))]


class _Predictor:
    def __init__(self, horizon:int)->None:
        self.horizon=horizon

    def predict(self, features: pd.DataFrame)->np.ndarray:
        return np.array([features["lag_0h"].iloc[0] + self.horizon])


def _measurements(hours:int, end:pd.Timestamp=LAST)->pd.DataFrame:
    index=pd.date_range(end-pd.Timedelta(hours=hours-1), end, freq="h")
    return pd.DataFrame({"timestamp": index, "kw": np.arange(float(hours))})


@pytest.fixture
def service()->ForecastService:
    # un solo database condiviso fra i thread: il client di test ne usa uno suo
    engine=create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    upsert_measurements(engine, _measurements(400))
    return ForecastService(engine, FakeRegistry(), PARAMS)


@pytest.fixture
def client(service:ForecastService)->Iterator[TestClient]:
    app.dependency_overrides[get_service]=lambda:service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_window_covers_the_longest_lag(service:ForecastService)->None:
    # lag 24 e finestra mobile di 24 ore: servono 25 osservazioni
    assert service.window_hours==25


def test_forecast_returns_one_point_per_horizon(client:TestClient)->None:
    response=client.get("/forecast")

    assert response.status_code==200
    body=response.json()
    assert len(body["points"])==3
    assert [point["horizon"] for point in body["points"]]==[1, 2, 3]
    assert body["points"][0]["timestamp"].startswith("2010-01-10T01:00")
    assert body["points"][0]["model_version"]=="1"


def test_forecast_uses_the_requested_moment(client: TestClient)->None:
    response=client.get("/forecast", params={"issued_at": "2010-01-09T00:00:00"})

    assert response.status_code==200
    assert response.json()["issued_at"].startswith("2010-01-09T00:00")


def test_incomplete_window_is_refused(client:TestClient, service:ForecastService)->None:
    # un buco a ridosso dell'istante richiesto rende la finestra inutilizzabile
    with service.engine.begin() as connection:
        connection.exec_driver_sql(
            "DELETE FROM measurements WHERE timestamp = '2010-01-09 12:00:00.000000'"
        )

    response=client.get("/forecast")

    assert response.status_code==422
    detail=response.json()["detail"]
    assert detail["expected_hours"]==25
    assert detail["found_hours"]==24


def test_window_before_the_data_is_refused(service:ForecastService)->None:
    with pytest.raises(WindowUnavailable):
        service.features_at(pd.Timestamp("2009-12-25 00:00"))


def test_measurements_are_accepted_and_extend_the_series(client:TestClient)->None:
    new_hour={"timestamp": "2010-01-10T01:00:00", "kw": 1.5}

    response=client.post("/measurements", json=[new_hour])

    assert response.status_code==201
    assert response.json()["written"]==1
    assert client.get("/forecast").json()["issued_at"].startswith("2010-01-10T01:00")


def test_health_reports_the_series(client:TestClient)->None:
    body=client.get("/health").json()

    assert body["status"]=="ok"
    assert body["database"] is True
    assert body["measurements"]==400


def test_measurements_range_is_inclusive(client:TestClient)->None:
    response=client.get(
        "/measurements",
        params={"start": "2010-01-09T22:00:00", "end": "2010-01-10T00:00:00"},
    )

    assert response.status_code==200
    body=response.json()
    assert len(body)==3
    assert body[0]["timestamp"].startswith("2010-01-09T22:00")


def test_measurements_range_rejects_inverted_bounds(client:TestClient)->None:
    response=client.get(
        "/measurements",
        params={"start": "2010-01-10T00:00:00", "end": "2010-01-09T00:00:00"},
    )

    assert response.status_code==400