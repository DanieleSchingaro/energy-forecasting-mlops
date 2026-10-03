#!/bin/bash
# Eseguito una sola volta, alla creazione del volume: separa i metadati di
# MLflow dalle misurazioni, pur restando nella stessa istanza Postgres.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE mlflow;
EOSQL