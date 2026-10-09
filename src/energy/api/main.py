#src/energy/api/main.py

"""
API di previsione.

Avvio in sviluppo:
    uvicorn energy.api.main:app --reload

La configurazione del modello e delle feature arriva da configs/params.yaml,
versionato insieme al codice; gli indirizzi di database e MLflow arrivano
dall'ambiente, perche' cambiano fra host e container.
"""

from __future__ import annotations
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from energy.api.registry import ModelRegistry
from energy.api.schemas import ForecastResponse, Health, IngestResponse, Measurement
from energy.api.service import ForecastService, WindowUnavailable
from energy.config import load_params
from energy.db import coverage, create_schema, get_engine, upsert_measurements
from energy.tracking import tracking_uri

logger=logging.getLogger(__name__)


def _preload_requested()->bool:
    return os.environ.get("API_PRELOAD_MODELS", "").lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(app: FastAPI)->AsyncIterator[None]:
    """
    Nel container i modelli si caricano all'avvio, non alla prima richiesta.
    Se il registry non risponde il servizio parte lo stesso: i modelli verranno
    caricati piu' tardi, e /health dichiara quanti ne risultano pronti.
    """
    if _preload_requested():
        try:
            get_service().registry.preload()
        except Exception as error:  # noqa: BLE001 - l'avvio non deve dipenderne
            logger.warning("precaricamento dei modelli non riuscito: %s", error)
    yield


app=FastAPI(
    title="Energy forecasting API",
    description="Previsione oraria del consumo elettrico fino a 24 ore di anticipo.",
    version="0.1.0",
    lifespan=lifespan,
)


@lru_cache
def get_service()->ForecastService:
    """
    Costruita una volta sola: l'engine ha un pool e i modelli restano in memoria.
    """
    params=load_params()
    # qui serve solo leggere dal registry: impostare l'esperimento comporterebbe
    # una chiamata di rete all'avvio, e un servizio che muore se MLflow tarda
    mlflow.set_tracking_uri(tracking_uri(params["mlflow"]))

    engine=get_engine()
    create_schema(engine)

    registry=ModelRegistry(params["mlflow"], params["features"]["max_horizon"])
    return ForecastService(engine, registry, params)


@app.get("/health", response_model=Health)
def health(service: ForecastService=Depends(get_service))->Health:
    try:
        stats=coverage(service.engine)
        reachable=True
    except Exception:  # il servizio resta interrogabile anche a database spento
        stats={"rows":0, "last":None}
        reachable=False

    return Health(
        status="ok" if reachable else "degraded",
        database=reachable,
        measurements=stats["rows"],
        last_measurement=stats["last"],
        models_loaded=len(service.registry.describe()),
    )


@app.get("/models")
def models(service:ForecastService=Depends(get_service))->dict:
    """
    Modelli caricati finora, con alias e versione che li identificano.
    """
    return {
        "registered_model":service.registry.name,
        "max_horizon":service.max_horizon,
        "loaded":service.registry.describe(),
    }


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    issued_at:datetime|None=Query(
        default=None,
        description="ultima ora osservata; se omessa, la piu' recente nel database",
    ),
    service: ForecastService=Depends(get_service),
)->dict:
    moment=pd.Timestamp(issued_at) if issued_at else service.latest_timestamp()
    if moment is None:
        raise HTTPException(status_code=409, detail="nessuna misurazione disponibile")

    try:
        return service.forecast(moment)
    except WindowUnavailable as problem:
        raise HTTPException(
            status_code=422,
            detail={
                "detail":str(problem),
                "expected_hours":problem.expected,
                "found_hours":problem.found,
                "missing":[str(stamp) for stamp in problem.missing],
            },
        ) from problem


@app.post("/measurements", response_model=IngestResponse, status_code=201)
def add_measurements(
    measurements:list[Measurement],
    service:ForecastService=Depends(get_service),
)->dict:
    """
    Inserisce o aggiorna misurazioni: e' cio' che tiene viva la finestra.
    """
    if not measurements:
        raise HTTPException(status_code=400, detail="nessuna misurazione nel corpo della richiesta")

    frame=pd.DataFrame([item.model_dump() for item in measurements])
    written=upsert_measurements(service.engine, frame)
    return {"written":written, **coverage(service.engine)}