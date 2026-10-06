#src/energy/data/ingest.py

"""
Carica la serie oraria nel database delle misurazioni.
E' lo stage che rende il sistema interrogabile: l'API non legge il parquet, ma
questa tabella, come farebbe con misurazioni che arrivano dal campo.
Rilanciarlo e' sicuro: le righe gia' presenti vengono aggiornate, non duplicate.
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from energy.config import load_params
from energy.db import coverage, create_schema, get_engine, upsert_measurements

OUT=Path("reports/ingest.json")


def main()->None:
    params=load_params()
    target=params["data"]["target"]

    frame=(
        pd.read_parquet(params["data"]["processed_path"])
        .reset_index()
        .rename(columns={"timestamp":"timestamp", target:"kw"})
    )

    engine=get_engine()
    create_schema(engine)
    written=upsert_measurements(engine, frame[["timestamp", "kw"]])

    report ={"rows_written":written, **coverage(engine)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__=="__main__":
    main()