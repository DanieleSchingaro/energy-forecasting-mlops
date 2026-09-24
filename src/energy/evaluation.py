#src/energy/evaluation.py

"""
Split temporale, baseline naive e metriche, condivise da tutti i metodi.

Il test set è definito nel target (t+h), non su quello di emissione della previsione, così da non esser inserito nel training.

Ogni metrica si calcola sulle metriche restituite da evaluation_set, in modo da far avvenire confronti sulle stesse ore.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# season=1 con horizon=1 da' la persistenza (y_hat(t+1) = y(t)), che a un'ora
# di distanza e' la baseline piu' difficile da battere; 24 e 168 coprono le
# stagionalita' giornaliera e settimanale.
NAIVE_SEASONS=(1, 24,168)

def split_date(index:pd.DatetimeIndex, test_months:int)->pd.Timestamp:
    """
    Ultimo istante del training. Dopo avviene il test.
    """
    return index.max()-pd.DateOffset(months=test_months)

def is_test(index:pd.DatetimeIndex, horizon:int, split:pd.Timestamp)->np.array:
    """
    True per righe in cui target y(t+h) cade dopo lo split.
    """
    return np.asarray(index+pd.Timedelta(hours=horizon)>split)

def naive_forecast(y:pd.Series, season:int, horizon:int)->pd.Series:
    """
    Previsione naive stagionale indicizzata per istante di emissione t.

    y_hat(t+h)=y(t+h-season) valore alla stessa ora ma di un periodo differente. 
    Richiede season>=horizon.
    """
    if season<horizon:
        raise ValueError(f"season={season}<horizion={horizon}: userebbe il futuro")
    return y.shift(season-horizon)

def regression_metrics(y_true:pd.Series, y_pred:pd.Series)->dict[str, float]:
    error=y_true-y_pred
    return{
        "mae":round(float(error.abs().mean()), 4),
        "rmse":round(float((error**2).mean()**0.5), 4),
    }

def evaluation_set(
        raw:pd.DataFrame,
        X:pd.DataFrame,
        y:pd.Series,
        target:str,
        horizon:int,
        split:pd.Timestamp,
)->pd.DataFrame:
    """
    Test valutabilit da tutti i metodi.
    Una riga entra se ha feature complete (garantito da X), target presente e previsioni disponibili.
    Colonne: y e una naive_<season>h per ciascuna stagionalità.
    """
    frame=pd.DataFrame({"y":y}, index=X.index)
    for season in NAIVE_SEASONS:
        frame[f"naive_{season}h"]=naive_forecast(raw[target], season, horizon)
    frame=frame[is_test(frame.index, horizon, split)]
    return frame.dropna()

def reference_baseline(metrics:dict, choice:str="best")->tuple[str, float]:
    """
    Baseline con cui confrontare i modelli, letta da reports/baseline_metrics.json
    Con choice="best" vince la naive con il MAE più basso.
    In alternative si può fissare una baseline per nome.
    """
    available={
        name:payload["mae"]
        for name, payload in metrics.items()
        if name.startswith("naive_") and isinstance(payload, dict)
    }
    if not available:
        raise ValueError("nessuna baseline nelle metriche: esegui prima energy.models.baseline")

    if choice=="best":
        name=min(available, key=available.get)
    elif choice in available:
        name=choice
    else:
        raise ValueError(f"baseline '{choice}' non disponibile: {sorted(available)}")

    return name, available[name]