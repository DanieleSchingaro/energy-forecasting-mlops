#src/energy/models/baseline.py

"""
Confronto di baseline stagionali su stessa ora.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import json

DATA=Path("data/processed/hourly.parquet")
OUT=Path("reports/baseline_metrics.json")
TEST_MONTHS=6
TARGET="global_active_power"

def evaluate(y_true:pd.Series, y_pred:pd.Series)->dict[str, float]:
    mask=y_true.notna() & y_pred.notna()
    error=y_true[mask]-y_pred[mask]
    return{
        "mae": round(float(error.abs().mean()), 4),
        "rmse": round(float((error**2).mean()**0.5), 4),
        "n_valid": int(mask.sum()),
    }

def main()->None:
    y=pd.read_parquet(DATA)[TARGET]

    split=y.index.max()-pd.DateOffset(months=TEST_MONTHS)
    test=y[y.index>split]
    print(f"Test: {test.index.min()} -> {test.index.max()} ({len(test)} ore)")

    metrics={
        "split_date":str(split),
        "test_hours":len(test),
        "test_mean_kw":round(float(test.mean()), 4),
        "naive_24h":evaluate(test, y.shift(24).loc[test.index]),
        "naive_168":evaluate(test, y.shift(168).loc[test.index]),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__=="__main__":
    main()