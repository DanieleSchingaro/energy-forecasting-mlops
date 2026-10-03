# Server MLflow con il driver Postgres: l'immagine ufficiale non lo include.
FROM python:3.12-slim

ARG MLFLOW_VERSION=3.16.1

RUN pip install --no-cache-dir \
    "mlflow==${MLFLOW_VERSION}" \
    "psycopg2-binary>=2.9,<3"

WORKDIR /mlflow
EXPOSE 5000

# Il comando effettivo arriva da docker-compose.yml, cosi' gli indirizzi del
# backend restano in un punto solo.
CMD ["mlflow", "server", "--host=0.0.0.0", "--port=5000"]