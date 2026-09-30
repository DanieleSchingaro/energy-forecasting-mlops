#src/energy/models/baseline.py

"""
Confronto di baseline stagionali su stessa ora, valutate su insieme di test condiviso.
Ogni baseline viene registrata come run MLFlow con il tag kind=baseline.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import json
import mlflow
from energy.config import load_params
from energy.evaluation import(
    evaluation_set,
    regression_metrics,
    split_date,
    seasons_for_horizon,
)
from energy.features.build import build_features, supervised_for_horizon
from energy.tracking import setup_mlflow

OUT=Path("reports/baseline_metrics.json")

def main()->None:
    params=load_params()
    target=params["data"]["target"]
    max_horizon=params["features"]["max_horizon"]

    raw=pd.read_parquet(params["data"]["processed_path"])
    features=build_features(
        raw,
        target,
        horizon,
        params["features"]["lags"],
        params["features"]["rolling_windows"],
    )

    split=split_date(raw.index, params["split"]["test_months"])
    test_hours_total=int((raw.index>split).sum())

    setup_mlflow(params)
    horizons:dict[str, dict]={}

    for horizon in range(1, max_horizon+1):
        X, y=supervised_for_horizon(features, raw[target], horizon)
        test=evaluation_set(raw, X, y, target, horizon, split)
 
        entry={
            "test_hours_evaluated":len(test),
            "test_coverage":round(len(test)/test_hours_total, 4),
            "test_mean_kw":round(float(test["y"].mean()), 4),
        }
        for season in seasons_for_horizon(horizon):
            name=f"naive_{season}h"
            entry[name]=regression_metrics(test["y"], test[name])
 
            with mlflow.start_run(run_name=f"{name}_h{horizon}"):
                mlflow.set_tags({"kind": "baseline", "model":name, "horizon":horizon})
                mlflow.log_params({"season_hours":season, "horizon":horizon})
                mlflow.log_metrics(
                    {
                        "test_mae":entry[name]["mae"],
                        "test_rmse":entry[name]["rmse"],
                        "n_test":len(test),
                    }
                )
 
        horizons[str(horizon)]=entry
        best=min(
            (s for s in seasons_for_horizon(horizon)),
            key=lambda s:entry[f"naive_{s}h"]["mae"],
        )
        print(
            f"h={horizon:2d}  ore {len(test)}  migliore naive_{best}h "
            f"MAE {entry[f'naive_{best}h']['mae']:.4f}"
        )
 
    metrics={
        "split_date":str(split),
        "max_horizon":max_horizon,
        "test_hours_total":test_hours_total,
        "horizons":horizons,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nScritto {OUT}")
 
 
if __name__=="__main__":
    main()