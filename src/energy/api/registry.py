#src/energy/api/registry.py

"""
Caricamento dei modelli dal Model Registry di MLflow.
L'API non carica file da models/: chiede al registry il modello che porta
l'alias champion dell'orizzonte richiesto. Cosi' promuovere un modello nuovo
non richiede di ricostruire l'immagine, e la risposta puo' dichiarare quale
versione ha prodotto ciascun valore.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import mlflow
import pandas as pd


@dataclass(frozen=True)
class LoadedModel:
    """
    Un modello pronto all'uso, con la sua provenienza.
    """

    horizon:int
    predictor:Any
    version:str
    run_id:str
    alias:str
    flavor:str

    def predict(self, features: pd.DataFrame)->float:
        return float(self.predictor.predict(features)[0])


class ModelRegistry:
    """
    Carica e tiene in memoria un modello per orizzonte.
    """

    def __init__(self, config:dict[str, Any], max_horizon:int)->None:
        self.name=config["registered_model"]
        self.alias_prefix=config["champion_alias"]
        self.max_horizon=max_horizon
        self._models:dict[int, LoadedModel]={}

    def alias(self, horizon: int)->str:
        return f"{self.alias_prefix}-h{horizon}"

    def uri(self, horizon:int)->str:
        return f"models:/{self.name}@{self.alias(horizon)}"

    def get(self, horizon:int)->LoadedModel:
        """
        Carica alla prima richiesta e riusa in seguito.
        """
        if horizon not in self._models:
            self._models[horizon]=self._load(horizon)
        return self._models[horizon]

    def preload(self)->None:
        for horizon in range(1, self.max_horizon+1):
            self.get(horizon)

    def describe(self)->list[dict[str, Any]]:
        """
        Modelli attualmente caricati, per l'endpoint diagnostico.
        """
        return [
            {
                "horizon":model.horizon,
                "alias":model.alias,
                "version":model.version,
                "run_id":model.run_id,
                "flavor":model.flavor,
            }
            for model in sorted(self._models.values(), key=lambda item: item.horizon)
        ]

    def _load(self, horizon:int)->LoadedModel:
        client=mlflow.MlflowClient()
        alias=self.alias(horizon)
        version=client.get_model_version_by_alias(self.name, alias)
        predictor=mlflow.pyfunc.load_model(self.uri(horizon))
        return LoadedModel(
            horizon=horizon,
            predictor=predictor,
            version=str(version.version),
            run_id=version.run_id,
            alias=alias,
            flavor=version.tags.get("model", "sconosciuto"),
        )