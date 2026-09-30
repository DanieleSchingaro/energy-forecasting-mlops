#src/energy/models/promote.py

"""
Promozione del modello migliore per ciascun orizzonte nel Model Registry.
 
Per ogni orizzonte cerca il run con il MAE di test piu' basso, registra il suo
modello e gli assegna l'alias champion-h<orizzonte>, ma solo se batte la
baseline naive di quello stesso orizzonte: un modello peggiore della naive non
deve finire in servizio.
 
Gli stage (Staging/Production) sono deprecati da MLflow 2.9, quindi si usano
gli alias: l'API carichera' models:/<nome>@champion-h<orizzonte>.
"""

from __future__ import annotations
import json
from pathlib import Path
import mlflow
import pandas as pd
from energy.config import load_params
from energy.tracking import setup_mlflow
from energy.evaluation import reference_baseline
 
BASELINE_PATH=Path("reports/baseline_metrics.json")
OUT=Path("reports/champion.json")
MODEL_ARTIFACT="model"

def model_runs(experiment:str)->pd.DataFrame:
    runs=mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="tags.kind='model'",
        order_by=["metrics.test_mae ASC"],
    )
    if runs.empty:
        raise SystemExit("Nessun run di training trovato: esegui prima energy.models.train")
    return runs
 
 
def existing_version(client:mlflow.MlflowClient, name:str, run_id:str)->str|None:
    """
    Evita una versione nuova per un run gia' registrato.
    """
    try:
        versions=client.search_model_versions(f"name = '{name}' and run_id = '{run_id}'")
    except mlflow.MlflowException:
        return None
    return versions[0].version if versions else None
 
 
def main()->None:
    params=load_params()
    config=setup_mlflow(params)
    name=config["registered_model"]
    alias_prefix=config["champion_alias"]
    reference_choice=params["evaluation"]["reference_baseline"]
 
    baseline=json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    runs=model_runs(config["experiment"])
    client=mlflow.MlflowClient()
 
    champions:dict[str, dict]={}
    for horizon in sorted({int(h) for h in runs["tags.horizon"].dropna()}):
        # i run sono gia' ordinati per MAE crescente: il primo dell'orizzonte e' il migliore
        best=runs[runs["tags.horizon"]==str(horizon)].iloc[0]
        run_id=best["run_id"]
        test_mae=float(best["metrics.test_mae"])
        model_name=best.get("tags.model", "sconosciuto")
 
        version=existing_version(client, name, run_id)
        if version is None:
            version=mlflow.register_model(f"runs:/{run_id}/{MODEL_ARTIFACT}", name).version
 
        client.set_model_version_tag(name, version, "model", model_name)
        client.set_model_version_tag(name, version, "horizon", str(horizon))
        client.set_model_version_tag(name, version, "test_mae", f"{test_mae:.4f}")
 
        reference_name, reference_mae=reference_baseline(
            baseline["horizons"][str(horizon)], reference_choice
        )
        promoted=test_mae<reference_mae
        alias=f"{alias_prefix}-h{horizon}"
        if promoted:
            client.set_registered_model_alias(name, alias, version)
 
        champions[str(horizon)]={
            "version":str(version),
            "alias":alias if promoted else None,
            "run_id":run_id,
            "model":model_name,
            "test_mae":round(test_mae, 4),
            "reference_baseline":reference_name,
            "reference_mae":reference_mae,
            "improvement_vs_reference":round(1-test_mae/reference_mae, 4),
        }
 
        status=f"-> {alias}" if promoted else "non promosso (non batte la baseline)"
        print(f"h={horizon:2d}  {model_name:14s} MAE {test_mae:.4f}  v{version}  {status}")
 
    payload={
        "registered_model":name,
        "model_uri_template": f"models:/{name}@{alias_prefix}-h{{horizon}}",
        "horizons":champions,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nScritto {OUT}")
 
 
if __name__=="__main__":
    main()