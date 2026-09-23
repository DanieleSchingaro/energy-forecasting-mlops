#src/energy/tracking.py

"""
Configurazione di MLflow.
Tutti gli script passano da qui.
"""

from __future__ import annotations
from typing import Any
import mlflow

SERIALIZATION_FORMAT="cloudpickle"

def setup_mlflow(params:dict[str, Any])->dict[str, Any]:
    """
    Imposta tracking URI ed esperimento.
    Restituisce la sezione mlflow.
    """
    config=params["mlflow"]
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(config["experiment"])
    return config

def flatten(values:dict[str, Any], prefix:str="")->dict[str, Any]:
    """
    Dizionario annidato.
    Chiavi piatte addatte a mlflow.log_params
    """
    flat:dict[str, Any]={}
    for key, value in values.items():
        name=f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        else:
            flat[name]=value
    return flat