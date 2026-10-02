#src/test/test_report.py

"""
Verifica la tabella di sintesi costruita dalle metriche.
"""

from __future__ import annotations
import pandas as pd
from energy.report import build_table

BASELINE={
    "split_date":"2009-11-26 21:00:00",
    "test_hours_total":8760,
    "horizons":{
        "1":{
            "test_hours_evaluated":7789,
            "test_coverage":0.8892,
            "naive_1h":{"mae":0.41, "rmse":0.62},
            "naive_24h":{"mae":0.57, "rmse":0.84},
        },
        "24":{
            "test_hours_evaluated":7697,
            "test_coverage":0.8787,
            "naive_24h":{"mae":0.57, "rmse":0.84},
        },
    },
}
MODELS={
    "random_forest":{
        "1":{"test":{"mae":0.34, "rmse":0.50}},
        "24":{"test":{"mae":0.47, "rmse":0.63}},
    },
    "xgboost":{
        "1":{"test":{"mae":0.35, "rmse":0.50}},
        "24":{"test":{"mae":0.46, "rmse":0.63}},
    },
}


def test_reference_follows_the_horizon()->None:
    table=build_table(BASELINE, MODELS, "best")

    # a un'ora vince la persistenza, piu' avanti non e' nemmeno disponibile
    assert table.loc[1, "reference_baseline"]=="naive_1h"
    assert table.loc[24, "reference_baseline"]=="naive_24h"


def test_best_model_and_improvement()->None:
    table=build_table(BASELINE, MODELS, "best")

    assert table.loc[1, "best_model"]=="random_forest"
    assert table.loc[24, "best_model"]=="xgboost"
    assert table.loc[1, "improvement"]==round(1-0.34/0.41,4)


def test_missing_model_leaves_a_gap()->None:
    models={"xgboost": MODELS["xgboost"]}
    table=build_table(BASELINE, models, "best")

    assert "random_forest" not in table
    assert pd.notna(table.loc[1, "xgboost"])