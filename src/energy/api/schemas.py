#src/energy/api/schemas.py

"""
Contratto dell'API: quello che entra e quello che esce.
"""

from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class Measurement(BaseModel):
    timestamp:datetime=Field(description="ora della misurazione, senza fuso")
    kw:float=Field(ge=0, description="potenza media nell'ora, in kW")


class IngestResponse(BaseModel):
    written:int
    rows:int
    first:str|None
    last:str|None


class ForecastPoint(BaseModel):
    timestamp:datetime=Field(description="ora prevista")
    horizon:int=Field(ge=1, description="ore di anticipo rispetto all'istante di emissione")
    kw:float
    model:str=Field(description="modello che ha prodotto il valore")
    model_version:str


class ForecastResponse(BaseModel):
    issued_at:datetime=Field(description="ultima ora osservata su cui si basa la previsione")
    window_hours:int=Field(description="ore di storico usate per costruire le feature")
    registered_model:str
    points:list[ForecastPoint]


class WindowProblem(BaseModel):
    """
    Dettaglio restituito quando la finestra di storico non e' utilizzabile.
    """

    detail:str
    expected_hours:int
    found_hours:int
    missing: list[datetime]=Field(default_factory=list, max_length=24)


class Health(BaseModel):
    status:str
    database:bool
    measurements:int
    last_measurement:str|None
    models_loaded:int