#docker/api/Dockerfile

# Immagine del servizio di previsione.
# Si costruisce dalla radice del progetto:
#   docker build -f docker/api/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Le dipendenze prima del codice: cambiano di rado, e questo strato resta
# in cache quando modifichi i sorgenti.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# --no-deps: le dipendenze sono gia' installate e bloccate sopra
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && rm -rf src

# params.yaml accompagna il codice: feature e orizzonti devono restare quelli
# del training. Gli indirizzi di database e MLflow arrivano dall'ambiente.
COPY configs ./configs

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

CMD ["uvicorn", "energy.api.main:app", "--host", "0.0.0.0", "--port", "8000"]