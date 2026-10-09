# Immagine della dashboard. Si costruisce dalla radice del progetto:
#   docker build -f docker/dashboard/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements-dashboard.txt

COPY dashboard ./dashboard

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]