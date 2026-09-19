#src/energy/data/make_dataset.py

"""
Converte le misure da minuto a serie oraria pulita, salvata in parquet.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RAW=Path("data/raw/household_power_consumption.txt")
OUT=Path("data/processed/hourly.parquet")
MIN_OBS_PER_HOUR=45

def load_raw()->pd.Series:
    df=pd.read_csv(
            RAW,
            sep=";",
            na_values=["?", ""],
            low_memory=False,
            usecols=["Date", "Time", "Global_active_power"],
    )
    timestamp=pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    series=pd.to_numeric(df["Global_active_power"], errors="coerce")
    series.index=pd.DatetimeIndex(timestamp, name="timestamp")
    return series.rename("global_active_power")

def to_hourly(minute:pd.Series)->pd.Series:
    hourly=minute.resample("h").mean()
    observed=minute.resample("h").count()
    hourly[observed<MIN_OBS_PER_HOUR]=np.nan
    return hourly

def main()->None:
    hourly=to_hourly(load_raw())
    full_index=pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly=hourly.reindex(full_index)
    hourly.index.name="timestamp"

    missing=int(hourly.isna().sum())
    print(f"Periodo: {hourly.index.min()} -> {hourly.index.max()}")
    print(
        f"Ore totali: {len(hourly)} | mancanti: {missing}"
        f"({missing/len(hourly):.2%})"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_frame().to_parquet(OUT)
    print(f"Scritto {OUT}")

if __name__=="__main__":
    main()