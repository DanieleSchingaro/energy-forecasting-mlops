#src/energy/models/train.py

"""
Training multi-orizzonte con strategia direct: un modello per ogni ora prevista.
 
Ogni coppia (modello, orizzonte) e' un run MLflow e un file in models/. Gli
orizzonti su cui addestrare si scelgono per modello in configs/params.yaml:
XGBoost gira su tutti, la Random Forest solo su quelli indicati, perche' ogni
foresta pesa quasi cento megabyte.

Uso:
    python -m energy.models.train                    # entrambi i modelli
    python -m energy.models.train --model xgboost    # uno solo
    python -m energy.models.train --horizon 24         # un orizzonte solo
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
from energy.evaluation import (
    evaluation_set,
    is_test,
    reference_baseline,
    regression_metrics,
    split_date,
)
from energy.features.build import build_features, supervised_for_horizon
from energy.tracking import SERIALIZATION_FORMAT, flatten, setup_mlflow
 
MODELS_DIR=Path("models")
REPORTS_DIR=Path("reports")
IMPORTANCE_DIR=REPORTS_DIR/"feature_importance"
METRICS_PATH=REPORTS_DIR/"model_metrics.json"
BASELINE_PATH=REPORTS_DIR/"baseline_metrics.json"
MODEL_ARTIFACT="model"
 
 
def build_model(name:str, config:dict):
    if name=="random_forest":
        return RandomForestRegressor(**config)
    if name=="xgboost":
        return XGBRegressor(**config)
    raise ValueError(f"modello sconosciuto: {name}")
 
 
def resolve_horizons(setting, max_horizon:int)->list[int]:
    """
    `all` significa 1..max_horizon; altrimenti l'elenco dichiarato.
    """
    if setting=="all":
        return list(range(1, max_horizon+1))
    return [h for h in setting if 1<=h<=max_horizon]
 
 
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
 
    series=pd.Series(scores)
    return{
        "folds":[round(score, 4) for score in scores],
        "mae_mean":round(float(series.mean()), 4),
        "mae_std":round(float(series.std()), 4),
    }
 
 
def save_feature_importance(model, columns:pd.Index, name:str, horizon:int)->Path:
    importance=pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    path=IMPORTANCE_DIR / f"{name}_h{horizon}.csv"
    importance.to_csv(path, header=["importance"])
    return path
 
 
def load_baseline()->dict|None:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
 
 
def write_metrics(metrics:dict)->None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
 
 
def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["random_forest", "xgboost", "all"],
        default="all",
        help="modello da addestrare (default: tutti)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="limita il training a un singolo orizzonte, utile per prove rapide",
    )
    args=parser.parse_args()
 
    params=load_params()
    target=params["data"]["target"]
    max_horizon=params["features"]["max_horizon"]
    reference_choice=params["evaluation"]["reference_baseline"]
 
    raw=pd.read_parquet(params["data"]["processed_path"])
    features=build_features(
        raw,
        target,
        params["features"]["lags"],
        params["features"]["rolling_windows"],
    )
    split=split_date(raw.index, params["split"]["test_months"])
    baseline=load_baseline()
 
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    setup_mlflow(params)
 
    names=["random_forest", "xgboost"] if args.model=="all" else [args.model]
    metrics: dict[str, dict]={}
    if METRICS_PATH.exists():
        metrics=json.loads(METRICS_PATH.read_text(encoding="utf-8"))
 
    for name in names:
        config=params["model"][name]
        horizons=resolve_horizons(config["horizons"], max_horizon)
        if args.horizon is not None:
            horizons=[h for h in horizons if h==args.horizon]
        metrics.setdefault(name, {})
        print(f"\n[{name}] orizzonti:{horizons}")
 
        for horizon in horizons:
            X, y=supervised_for_horizon(features, raw[target], horizon)
            test=evaluation_set(raw, X, y, target, horizon, split)
 
            train_mask=~is_test(X.index, horizon, split)
            X_train, y_train=X[train_mask], y[train_mask]
            X_test, y_test=X.loc[test.index], test["y"]
 
            reference=None
            if baseline is not None:
                reference=reference_baseline(baseline["horizons"][str(horizon)], reference_choice)
 
            with mlflow.start_run(run_name=f"{name}_h{horizon}"):
                mlflow.set_tags({"kind":"model", "model":name, "horizon":horizon})
                mlflow.log_params(flatten(config["params"], name))
                mlflow.log_params(flatten(params["split"], "split"))
                mlflow.log_params({"horizon":horizon, "n_features":X.shape[1]})
 
                model=build_model(name, config["params"])
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
                scores=regression_metrics(y_test, predictions)
 
                payload={
                    "cv":cv_scores,
                    "test":scores,
                    "fit_seconds":fit_seconds,
                    "n_train":len(X_train),
                    "n_test":len(X_test),
                }
                logged={
                    "cv_mae_mean":cv_scores["mae_mean"],
                    "cv_mae_std":cv_scores["mae_std"],
                    "test_mae":scores["mae"],
                    "test_rmse":scores["rmse"],
                    "fit_seconds":fit_seconds,
                    "n_train":len(X_train),
                    "n_test":len(X_test),
                }
                if reference is not None:
                    reference_name, reference_mae = reference
                    improvement=round(1-scores["mae"]/reference_mae, 4)
                    payload["reference"]={"baseline":reference_name, "mae":reference_mae}
                    payload["improvement_vs_reference"]=improvement
                    logged["improvement_vs_reference"]=improvement
                    logged["reference_mae"]=reference_mae
                    mlflow.set_tag("reference_baseline", reference_name)
                mlflow.log_metrics(logged)
 
                joblib.dump(model, MODELS_DIR/f"{name}_h{horizon}.joblib")
                mlflow.log_artifact(
                    str(save_feature_importance(model, X.columns, name, horizon)),
                    artifact_path="reports",
                )
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name=MODEL_ARTIFACT,
                    signature=infer_signature(X_test.head(100), predictions.head(100)),
                    input_example=X_test.head(2),
                    serialization_format=SERIALIZATION_FORMAT,
                )
 
            metrics[name][str(horizon)]=payload
            write_metrics(metrics)
 
            summary=f"  h={horizon:2d}  CV {cv_scores['mae_mean']:.4f}  test {scores['mae']:.4f}"
            if reference is not None:
                summary+=(
                    f"  {reference_name} {reference_mae:.4f}  "
                    f"miglioramento {payload['improvement_vs_reference']:+.1%}"
                )
            print(f"{summary}  ({fit_seconds}s)")
 
    print(f"\nMetriche in {METRICS_PATH}, modelli in {MODELS_DIR}/")
 
 
if __name__=="__main__":
    main()