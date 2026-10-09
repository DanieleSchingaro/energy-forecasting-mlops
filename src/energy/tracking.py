#src/energy/tracking.py

"""
Configurazione unica di MLflow.
Tutti gli script passano da qui, cosi' l'URI di tracking e il nome
dell'esperimento vivono solo in configs/params.yaml.
"""

from __future__ import annotations
import os
from typing import Any
import mlflow

# I modelli ad albero vengono serializzati con cloudpickle: il formato skops,
# predefinito nelle versioni recenti di MLflow, rifiuta sklearn.tree._tree.Tree.
SERIALIZATION_FORMAT="cloudpickle"


def tracking_uri(config:dict[str, Any])->str:
    """
    Indirizzo del server: l'ambiente ha la precedenza sul file.
    """
    return os.environ.get("MLFLOW_TRACKING_URI", config["tracking_uri"])


def setup_mlflow(params:dict[str, Any])->dict[str, Any]:
    """
    Imposta tracking URI ed esperimento, e restituisce la sezione mlflow.
    La variabile d'ambiente MLFLOW_TRACKING_URI ha la precedenza sul file di
    configurazione: dentro un container l'indirizzo del server e' diverso da
    quello visto dall'host, e deve poter cambiare senza toccare params.yaml.
    """
    config=params["mlflow"]
    mlflow.set_tracking_uri(tracking_uri(config))
    mlflow.set_experiment(config["experiment"])
    return config


def flatten(values:dict[str, Any], prefix:str="")->dict[str, Any]:
    """
    Dizionario annidato -> chiavi piatte, adatte a mlflow.log_params.
    """
    flat:dict[str, Any]={}
    for key, value in values.items():
        name=f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        else:
            flat[name]=value
    return flat