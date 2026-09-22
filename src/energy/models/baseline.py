#src/energy/models/baseline.py

"""
Confronto di baseline stagionali su stessa ora, valutate su insieme di test condiviso.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import json
from energy.config import load_params
from energy.evaluation import(
    NAIVE_SEASONS,
    evaluation_set,
    regression_metrics,
    split_date,
)
from energy.features.build import make_supervised

OUT=Path("reports/baseline_metrics.json")

def main()->None:
    params=load_params()
    target=params["data"]["target"]
    horizon=params["features"]["horizon"]

    raw=pd.read_parquet(params["data"]["processed_path"])
    X, y=make_supervised(
        raw,
        target,
        horizon,
        params["features"]["lags"],
        params["features"]["rolling_windows"],
    )

    split=split_date(raw.index, params["split"]["test_months"])
    test=evaluation_set(raw, X, y, target, horizon, split)
    test_hours_total=int((raw.index>split).sum())

    metrics={
        "split_date":str(split),
        "horizon":horizon,
        "test_hours_total":test_hours_total,
        "test_hours_evaluated":len(test),
        "test_coverage":round(len(test)/test_hours_total, 4),
        "test_mean_kw":round(float(test["y"].mean()), 4),
    }

    for season in NAIVE_SEASONS:
        column=f"naive_{season}h"
        metrics[column]=regression_metrics(test["y"], test[column])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__=="__main__":
    main()