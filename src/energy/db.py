#src/energy/db.py

"""
Accesso al database delle misurazioni.
La connessione arriva da variabili d'ambiente, mai da configs/params.yaml: le
credenziali non vanno in un file versionato, e l'indirizzo cambia fra host e
container senza che il codice ne sappia nulla.
"""

from __future__ import annotations
import pandas as pd
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    MetaData,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

METADATA=MetaData()

MEASUREMENTS=Table(
    "measurements",
    METADATA,
    Column("timestamp", DateTime(timezone=False), primary_key=True),
    Column("kw", Float, nullable=False),
    Column("ingested_at", DateTime(timezone=False), server_default=func.now()),
)


class DatabaseSettings(BaseSettings):
    """
    Lette da .env o dall'ambiente; DATABASE_URL ha la precedenza sui pezzi.
    """

    model_config=SettingsConfigDict(env_file=".env", extra="ignore")

    database_url:str=""
    postgres_user:str="energy"
    postgres_password:str="energy"
    postgres_db:str="energy"
    postgres_host:str="localhost"
    postgres_port:int=5432

    def url(self)->str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def get_engine(url:str|None=None)->Engine:
    return create_engine(url or DatabaseSettings().url(), pool_pre_ping=True)


def create_schema(engine:Engine)->None:
    METADATA.create_all(engine)


def upsert_measurements(engine:Engine, frame:pd.DataFrame, chunk_size:int=5000)->int:
    """
    Inserisce o aggiorna le misurazioni, a blocchi. Rilanciarlo non duplica nulla.
    `frame` ha colonne timestamp e kw. Le righe senza valore vengono scartate:
    un'ora mancante resta assente, non diventa uno zero.
    """
    clean=frame.dropna(subset=["kw"])
    # conversione esplicita: i tipi numpy non sono accettati da tutti i driver
    rows=[
        {"timestamp":pd.Timestamp(stamp).to_pydatetime(), "kw":float(value)}
        for stamp, value in zip(clean["timestamp"], clean["kw"], strict=True)
    ]
    if not rows:
        return 0

    insert=_insert_for(engine)
    written=0
    with engine.begin() as connection:
        for start in range(0, len(rows), chunk_size):
            chunk=rows[start:start+chunk_size]
            statement=insert(MEASUREMENTS).values(chunk)
            statement=statement.on_conflict_do_update(
                index_elements=["timestamp"],
                set_={"kw":statement.excluded.kw},
            )
            connection.execute(statement)
            written+=len(chunk)
    return written


def read_window(engine:Engine, end, hours:int)->pd.DataFrame:
    """
    Le `hours` ore che terminano a `end` incluso, ordinate nel tempo.
    E' la finestra che serve all'API per costruire le feature.
    """
    start=pd.Timestamp(end)-pd.Timedelta(hours=hours-1)
    query=(
        select(MEASUREMENTS.c.timestamp, MEASUREMENTS.c.kw)
        .where(MEASUREMENTS.c.timestamp>=start.to_pydatetime())
        .where(MEASUREMENTS.c.timestamp<=pd.Timestamp(end).to_pydatetime())
        .order_by(MEASUREMENTS.c.timestamp)
    )
    with engine.connect() as connection:
        frame=pd.DataFrame(connection.execute(query).fetchall(), columns=["timestamp", "kw"])
    frame["timestamp"]=pd.to_datetime(frame["timestamp"])
    return frame


def coverage(engine:Engine)->dict:
    """
    Numero di misurazioni e intervallo coperto, per le diagnostiche.
    """
    query=select(
        func.count(MEASUREMENTS.c.timestamp),
        func.min(MEASUREMENTS.c.timestamp),
        func.max(MEASUREMENTS.c.timestamp),
    )
    with engine.connect() as connection:
        rows, first, last=connection.execute(query).one()
    return {
        "rows":int(rows),
        "first":str(first) if first else None,
        "last":str(last) if last else None,
    }


def _insert_for(engine:Engine):
    """
    L'upsert non e' standard SQL: serve il costrutto del dialetto in uso.
    Postgres in esercizio, SQLite nei test, cosi' la stessa funzione e'
    verificabile senza un database in piedi.
    """
    if engine.dialect.name=="postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif engine.dialect.name=="sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        raise NotImplementedError(f"dialetto non supportato: {engine.dialect.name}")
    return insert


def read_range(engine:Engine, start, end, limit:int=5000)->pd.DataFrame:
    """
    Misurazioni fra due istanti inclusi, in ordine di tempo.
    """
    query=(
        select(MEASUREMENTS.c.timestamp, MEASUREMENTS.c.kw)
        .where(MEASUREMENTS.c.timestamp>=pd.Timestamp(start).to_pydatetime())
        .where(MEASUREMENTS.c.timestamp<=pd.Timestamp(end).to_pydatetime())
        .order_by(MEASUREMENTS.c.timestamp)
        .limit(limit)
    )
    with engine.connect() as connection:
        frame=pd.DataFrame(connection.execute(query).fetchall(), columns=["timestamp", "kw"])
    frame["timestamp"]=pd.to_datetime(frame["timestamp"])
    return frame