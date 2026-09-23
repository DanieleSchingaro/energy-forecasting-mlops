#src/energy/models/train.py

"""
Training dei modelli, valutati sull'insieme di test condiviso con le baseline.
 
Ogni modello produce un run MLflow con parametri, metriche, importanza delle
feature e il modello serializzato con la sua signature. I file su disco
(models/*.joblib, reports/*) restano perche' sono gli output tracciati da DVC:
MLflow tiene la storia degli esperimenti, DVC la riproducibilita' della pipeline.
 
Uso:
    python -m energy.models.train                    # entrambi i modelli
    python -m energy.models.train --model xgboost    # uno solo
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from energy.config import load_params
from energy.evaluation import evaluation_set, is_test, regression_metrics, split_date
from energy.features.build import make_supervised
from energy.tracking import SERIALIZATION_FORMAT, flatten, setup_mlflow
 
MODELS_DIR=Path("models")
REPORTS_DIR=Path("reports")
METRICS_PATH=REPORTS_DIR/"model_metrics.json"
BASELINE_PATH=REPORTS_DIR/"baseline_metrics.json"
REFERENCE="naive_24h"
MODEL_ARTIFACT="model"

def build_model(name: str, params: dict):
    config=params["model"][name]
    if name=="random_forest":
        return RandomForestRegressor(**config)
    if name=="xgboost":
        return XGBRegressor(**config)
    raise ValueError(f"modello sconosciuto: {name}")

def cross_validate(model, X:pd.DataFrame, y:pd.Series, n_splits:int, gap:int)->dict:
    """
    Rolling-origin CV: ogni fold valida su un periodo successivo al proprio training.
    `gap` righe separano training e validation, cosi' le finestre mobili della
    validation non si sovrappongono alle ultime ore viste in training.
    """
    cv=TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores=[]
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X), start=1):
        fitted=clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        mae=mean_absolute_error(y.iloc[valid_idx], fitted.predict(X.iloc[valid_idx]))
        scores.append(float(mae))
        mlflow.log_metric("cv_fold_mae", mae, step=fold)
        print(f"    fold {fold}: MAE {mae:.4f}  (train {len(train_idx)}, valid {len(valid_idx)})")
 
    series=pd.Series(scores)
    return {
        "folds":[round(score, 4) for score in scores],
        "mae_mean":round(float(series.mean()), 4),
        "mae_std":round(float(series.std()), 4),
    }

def save_feature_importance(model, columns:pd.Index, name:str)->Path:
    importance=pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
    path=REPORTS_DIR/f"feature_importance_{name}.csv"
    importance.to_csv(path, header=["importance"])
    print(f"    Feature piu' importanti: {', '.join(importance.head(5).index)}")
    return path

def load_reference_mae()->float|None:
    if not BASELINE_PATH.exists():
        return None
    baseline=json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return baseline.get(REFERENCE, {}).get("mae")

def update_metrics_file(name:str, payload:dict)->None:
    metrics={}
    if METRICS_PATH.exists():
        metrics=json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metrics[name]=payload
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["random_forest", "xgboost", "all"],
        default="all",
        help="modello da addestrare (default: entrambi)",
    )
    args=parser.parse_args()
 
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
 
    train_mask=~is_test(X.index, horizon, split)
    X_train, y_train=X[train_mask], y[train_mask]
    X_test, y_test=X.loc[test.index], test["y"]
 
    print(f"Train: {len(X_train)} righe fino a {X_train.index.max()}")
    print(f"Test:  {len(X_test)} righe da {X_test.index.min()}")
    print(f"Feature: {X.shape[1]}")
 
    reference_mae=load_reference_mae()
    if reference_mae is not None:
        print(f"Riferimento {REFERENCE}: MAE {reference_mae:.4f}\n")
 
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    setup_mlflow(params)
 
    names=["random_forest", "xgboost"] if args.model=="all" else [args.model]
    for name in names:
        print(f"[{name}]")
        model=build_model(name, params)
 
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"kind": "model", "model": name})
            mlflow.log_params(flatten(params["model"][name], name))
            mlflow.log_params(flatten(params["features"], "features"))
            mlflow.log_params(flatten(params["split"], "split"))
 
            cv_scores=cross_validate(
                model,
                X_train,
                y_train,
                params["split"]["cv_splits"],
                params["split"]["cv_gap_hours"],
            )
 
            started=time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds=round(time.perf_counter()-started, 1)
 
            predictions=pd.Series(model.predict(X_test), index=X_test.index)
            metrics=regression_metrics(y_test, predictions)
 
            payload={
                "cv":cv_scores,
                "test":metrics,
                "fit_seconds":fit_seconds,
                "n_train":len(X_train),
                "n_test":len(X_test),
                "params":params["model"][name],
            }
            logged={
                "cv_mae_mean":cv_scores["mae_mean"],
                "cv_mae_std":cv_scores["mae_std"],
                "test_mae":metrics["mae"],
                "test_rmse":metrics["rmse"],
                "fit_seconds":fit_seconds,
                "n_train":len(X_train),
                "n_test":len(X_test),
            }
            if reference_mae is not None:
                improvement=round(1-metrics["mae"]/reference_mae, 4)
                payload["improvement_vs_naive_24h"]=improvement
                logged["improvement_vs_naive_24h"]=improvement
                logged["reference_mae"]=reference_mae
            mlflow.log_metrics(logged)
 
            joblib.dump(model, MODELS_DIR/f"{name}.joblib")
            importance_path=save_feature_importance(model, X.columns, name)
            update_metrics_file(name, payload)
            mlflow.log_artifact(str(importance_path), artifact_path="reports")
 
            mlflow.sklearn.log_model(
                sk_model=model,
                name=MODEL_ARTIFACT,
                signature=infer_signature(X_test.head(100), predictions.head(100)),
                input_example=X_test.head(2),
                serialization_format=SERIALIZATION_FORMAT,
            )
 
        print(f"    CV MAE {cv_scores['mae_mean']:.4f} +/- {cv_scores['mae_std']:.4f}")
        print(
            f"    Test MAE {metrics['mae']:.4f} | RMSE {metrics['rmse']:.4f} | fit {fit_seconds}s"
        )
        if reference_mae is not None:
            print(f"    Miglioramento su {REFERENCE}: {payload['improvement_vs_naive_24h']:.1%}\n")
 
    print(f"Metriche in {METRICS_PATH}, modelli in {MODELS_DIR}/")
 
 
if __name__=="__main__":
    main()