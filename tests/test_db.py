#tests/test_db.py

"""
Verifica scrittura e lettura delle misurazioni, su SQLite in memoria.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from energy.db import DatabaseSettings, coverage, create_schema, read_window, upsert_measurements


@pytest.fixture
def engine():
    engine=create_engine("sqlite://")
    create_schema(engine)
    return engine


def _frame(n:int, start:str="2010-01-01")->pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp":pd.date_range(start, periods=n, freq="h"),
            "kw":np.arange(float(n)),
        }
    )


def test_upsert_is_idempotent(engine)->None:
    frame=_frame(50)

    upsert_measurements(engine, frame)
    upsert_measurements(engine, frame)

    assert coverage(engine)["rows"]==50


def test_upsert_updates_existing_values(engine)->None:
    upsert_measurements(engine, _frame(10))
    corrected=_frame(10)
    corrected.loc[3, "kw"]=99.0

    upsert_measurements(engine, corrected)
    window=read_window(engine, pd.Timestamp("2010-01-01 09:00"), hours=10)

    assert window.loc[3, "kw"]==99.0
    assert coverage(engine)["rows"]==10


def test_missing_values_are_not_stored(engine)->None:
    frame=_frame(10)
    frame.loc[5, "kw"]=np.nan

    written=upsert_measurements(engine, frame)

    assert written==9
    assert coverage(engine)["rows"]==9


def test_read_window_is_closed_on_both_ends(engine)->None:
    upsert_measurements(engine, _frame(400))
    end=pd.Timestamp("2010-01-10 00:00")

    window=read_window(engine, end, hours=168)

    assert len(window)==168
    assert window["timestamp"].iloc[-1]==end
    assert window["timestamp"].iloc[0]==end-pd.Timedelta(hours=167)


def test_database_url_wins_over_pieces()->None:
    settings=DatabaseSettings(database_url="postgresql://a:b@c:1/d", postgres_host="ignorato")

    assert settings.url()=="postgresql://a:b@c:1/d"