#src/energy/models/promote.py

"""
Promozione del miglior modello nel Model Registry di MLflow.
Cerca fra i run dell'esperimento quello con il MAE di test piu' basso, ne
registra il modello e gli assegna l'alias `champion`, ma solo se batte la
baseline naive: un modello peggiore della naive non deve finire in servizio.
 
Gli stage (Staging/Production) sono deprecati da MLflow 2.9, quindi si usano
gli alias: l'API caricera' il modello con models:/<nome>@champion.
"""

from __future__ import annotations
import json
from pathlib import Path
import mlflow
import pandas as pd
from energy.config import load_params
from energy.tracking import setup_mlflow
 
BASELINE_PATH=Path("reports/baseline_metrics.json")
OUT=Path("reports/champion.json")
REFERENCE="naive_24h"
MODEL_ARTIFACT="model"

def best_run(experiment:str)->pd.Series:
    runs=mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="tags.kind='model'",
        order_by=["metrics.test_mae ASC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit("Nessun run di training trovato: esegui prima energy.models.train")
    return runs.iloc[0]

def reference_mae()->float|None:
    if not BASELINE_PATH.exists():
        return None
    baseline=json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return baseline.get(REFERENCE, {}).get("mae")
 
 
def existing_version(client:mlflow.MlflowClient, name:str, run_id:str)->str|None:
    """
    Evita una versione nuova per un run gia' registrato.
    """
    try:
        versions=client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
    except mlflow.MlflowException:
        return None
    return versions[0].version if versions else None

def main()->None:
    params=load_params()
    config=setup_mlflow(params)
    name=config["registered_model"]
    alias=config["champion_alias"]
 
    run=best_run(config["experiment"])
    run_id=run["run_id"]
    test_mae=float(run["metrics.test_mae"])
    model_name=run.get("tags.model", "sconosciuto")
    print(f"Run migliore: {model_name} ({run_id[:8]}), test MAE {test_mae:.4f}")
 
    client=mlflow.MlflowClient()
    version=existing_version(client, name, run_id)
    if version is None:
        version=mlflow.register_model(f"runs:/{run_id}/{MODEL_ARTIFACT}", name).version
        print(f"Registrato {name} versione {version}")
    else:
        print(f"Run gia' registrato come {name} versione {version}")
 
    client.set_model_version_tag(name, version, "model", model_name)
    client.set_model_version_tag(name, version, "test_mae", f"{test_mae:.4f}")
 
    naive_mae=reference_mae()
    promoted=naive_mae is None or test_mae<naive_mae
    if promoted:
        client.set_registered_model_alias(name, alias, version)
        print(f"Alias {alias} -> versione {version}")
    else:
        print(
            f"Alias {alias} non assegnato: MAE {test_mae:.4f} non batte "
            f"la baseline {REFERENCE} ({naive_mae:.4f})"
        )
 
    champion={
        "registered_model":name,
        "version":str(version),
        "alias":alias if promoted else None,
        "run_id":run_id,
        "model":model_name,
        "test_mae":round(test_mae, 4),
        "reference_mae":naive_mae,
        "model_uri":f"models:/{name}@{alias}" if promoted else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(champion, indent=2), encoding="utf-8")
    print(json.dumps(champion, indent=2))
 
 
if __name__=="__main__":
    main()