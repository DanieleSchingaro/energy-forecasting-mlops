#dashboard/app.py

"""
Dashboard di previsione.
Parla solo con l'API, via HTTP: non importa nulla dal pacchetto energy e non
tocca il database. E' quindi un'applicazione a se', che si puo' spostare o
sostituire senza toccare il resto del sistema.

Avvio in sviluppo:
    streamlit run dashboard/app.py
"""

from __future__ import annotations
import os
from datetime import date, datetime, time
import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL=os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT=30
HISTORY_HOURS=48

OBSERVED="#2a78d6"
FORECAST="#eb6834"
INK="#1a1a19"
MUTED="#6b6a63"

st.set_page_config(page_title="Previsione consumi", page_icon="⚡", layout="wide")


@st.cache_data(ttl=60)
def get_health()->dict:
    return requests.get(f"{API_URL}/health", timeout=TIMEOUT).json()


@st.cache_data(ttl=60)
def get_models()->dict:
    return requests.get(f"{API_URL}/models", timeout=TIMEOUT).json()


@st.cache_data(ttl=60)
def get_forecast(issued_at:str|None)->tuple[dict|None, dict|None]:
    """
    Restituisce (previsione, errore): il 422 della finestra incompleta non e' un guasto.
    """
    params={"issued_at": issued_at} if issued_at else None
    response=requests.get(f"{API_URL}/forecast", params=params, timeout=TIMEOUT)
    if response.status_code==422:
        return None, response.json()["detail"]
    response.raise_for_status()
    return response.json(), None


@st.cache_data(ttl=60)
def get_measurements(start:str, end:str)->pd.DataFrame:
    response=requests.get(
        f"{API_URL}/measurements", params={"start":start, "end":end}, timeout=TIMEOUT
    )
    response.raise_for_status()
    frame=pd.DataFrame(response.json())
    if not frame.empty:
        frame["timestamp"]=pd.to_datetime(frame["timestamp"])
    return frame


def consumption_chart(history:pd.DataFrame, forecast:pd.DataFrame, issued_at:pd.Timestamp):
    """
    Storico e previsione sullo stesso asse, separati dall'istante di emissione.
    """
    series=pd.concat(
        [
            history.assign(serie="osservato"),
            forecast.rename(columns={"kw":"kw"})[["timestamp", "kw"]].assign(serie="previsto"),
        ]
    )

    lines=(
        alt.Chart(series)
        .mark_line(point=alt.OverlayMarkDef(size=30), strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T", title="ora"),
            y=alt.Y("kw:Q", title="consumo (kW)", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "serie:N",
                title=None,
                scale=alt.Scale(domain=["osservato", "previsto"], range=[OBSERVED, FORECAST]),
                legend=alt.Legend(orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "serie:N",
                scale=alt.Scale(domain=["osservato", "previsto"], range=[[1, 0], [6, 3]]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="ora"),
                alt.Tooltip("kw:Q", title="kW", format=".3f"),
                alt.Tooltip("serie:N", title="serie"),
            ],
        )
    )
    divider=(
        alt.Chart(pd.DataFrame({"timestamp": [issued_at]}))
        .mark_rule(color=MUTED, strokeDash=[4, 4])
        .encode(x="timestamp:T")
    )
    return (lines + divider).properties(height=380).interactive()


st.title("Previsione del consumo elettrico")
st.caption(
    "Storico osservato e previsione oraria fino a 24 ore. "
    "Ogni orizzonte e' servito dal modello che porta l'alias champion corrispondente."
)

try:
    health=get_health()
except requests.RequestException as error:
    st.error(f"API non raggiungibile su {API_URL}: {error}")
    st.stop()

with st.sidebar:
    st.subheader("Stato del servizio")
    st.metric("misurazioni", f"{health['measurements']:,}".replace(",", "."))
    st.metric("modelli caricati", health["models_loaded"])
    st.write(f"ultima ora disponibile: **{health['last_measurement']}**")
    st.write(f"database: {'connesso' if health['database'] else 'non raggiungibile'}")
    st.divider()

    last=pd.Timestamp(health["last_measurement"]) if health["last_measurement"] else None
    use_latest=st.toggle("usa l'ultima ora disponibile", value=True)
    chosen:pd.Timestamp|None=None
    if not use_latest and last is not None:
        chosen_day:date=st.date_input("giorno", value=last.date())
        chosen_hour:time=st.time_input("ora", value=time(12, 0), step=3600)
        chosen=pd.Timestamp(datetime.combine(chosen_day, chosen_hour))

forecast, problem=get_forecast(chosen.isoformat() if chosen is not None else None)

if problem:
    st.warning(
        f"Finestra di storico incompleta: {problem['found_hours']} ore su "
        f"{problem['expected_hours']}. Il servizio rifiuta di prevedere su dati mancanti."
    )
    with st.expander("ore mancanti"):
        st.write(problem["missing"])
    st.stop()

issued_at=pd.Timestamp(forecast["issued_at"])
points=pd.DataFrame(forecast["points"])
points["timestamp"]=pd.to_datetime(points["timestamp"])

history=get_measurements(
    (issued_at-pd.Timedelta(hours=HISTORY_HOURS)).isoformat(), issued_at.isoformat()
)

left, middle, right=st.columns(3)
left.metric("emessa alle", issued_at.strftime("%Y-%m-%d %H:%M"))
middle.metric("picco previsto", f"{points['kw'].max():.2f} kW")
right.metric("ora del picco", points.loc[points["kw"].idxmax(), "timestamp"].strftime("%H:%M"))

st.altair_chart(consumption_chart(history, points, issued_at), width="stretch")

with st.expander("dettaglio per orizzonte"):
    table=points.rename(
        columns={
            "timestamp":"ora prevista",
            "horizon":"orizzonte (h)",
            "kw":"kW",
            "model":"modello",
            "model_version":"versione",
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)

with st.expander("modelli in servizio"):
    st.caption(f"modello registrato: {get_models()['registered_model']}")
    st.dataframe(pd.DataFrame(get_models()["loaded"]), width="stretch", hide_index=True)