#src/energy/report.py

"""
Tabella e figura di sintesi a partire dai file di metriche.
 
Legge reports/baseline_metrics.json e reports/model_metrics.json e produce:
 
  reports/mae_by_horizon.csv            dati in forma tabellare
  reports/summary.md                    tabella markdown, pronta per il README
  reports/figures/mae_by_horizon.png    MAE e miglioramento per orizzonte
 
Sostituisce `dvc metrics show`, che con 24 orizzonti stampa una tabella di
centinaia di colonne.
"""
 
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd 
from energy.config import load_params  
from energy.evaluation import reference_baseline  
 
REPORTS_DIR=Path("reports")
BASELINE_PATH=REPORTS_DIR/"baseline_metrics.json"
MODELS_PATH=REPORTS_DIR/"model_metrics.json"
TABLE_PATH=REPORTS_DIR/"mae_by_horizon.csv"
SUMMARY_PATH=REPORTS_DIR/"summary.md"
FIGURE_PATH=REPORTS_DIR/"figures"/"mae_by_horizon.png"
 
SURFACE="#fcfcfb"
INK="#1a1a19"
MUTED="#6b6a63"
GRID="#e5e4df"
PERSISTENCE="#1baf7a"
MODEL_STYLE={
    "xgboost":("XGBoost", "#2a78d6"),
    "random_forest":("Random Forest", "#eb6834"),
}
 
 
def build_table(baseline:dict, models:dict, reference_choice:str)->pd.DataFrame:
    """
    Una riga per orizzonte: baseline di riferimento, modelli, miglioramento.
    """
    rows=[]
    for key, entry in sorted(baseline["horizons"].items(), key=lambda item: int(item[0])):
        reference_name, reference_mae=reference_baseline(entry, reference_choice)
        row={
            "horizon":int(key),
            "test_hours":entry["test_hours_evaluated"],
            "coverage":entry["test_coverage"],
            "reference_baseline":reference_name,
            "reference_mae":reference_mae,
        }
        for name, payload in entry.items():
            if name.startswith("naive_") and isinstance(payload, dict):
                row[name]=payload["mae"]
 
        scores={
            model:per_horizon[key]["test"]["mae"]
            for model, per_horizon in models.items()
            if key in per_horizon
        }
        row.update(scores)
        if scores:
            best=min(scores, key=scores.get)
            row["best_model"]=best
            row["best_mae"]=scores[best]
            row["improvement"]=round(1-scores[best]/reference_mae, 4)
        rows.append(row)
 
    return pd.DataFrame(rows).set_index("horizon")
 
 
def write_summary(table: pd.DataFrame, baseline: dict) -> None:
    lines=[
        "## Risultati per orizzonte",
        "",
        f"Test: ultimi 12 mesi, da {baseline['split_date']}. "
        f"Ore valutate da {int(table['test_hours'].min())} a {int(table['test_hours'].max())} "
        f"su {baseline['test_hours_total']} "
        f"(copertura {table['coverage'].min():.1%}-{table['coverage'].max():.1%}; "
        "le ore scartate sono quelle con finestra di 168 ore incompleta).",
        "",
        "| h | riferimento | MAE naive | Random Forest | XGBoost | migliore | miglioramento |",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for horizon, row in table.iterrows():
        forest=f"{row['random_forest']:.4f}" if pd.notna(row.get("random_forest")) else "—"
        boosted=f"{row['xgboost']:.4f}" if pd.notna(row.get("xgboost")) else "—"
        label=MODEL_STYLE[row["best_model"]][0]
        lines.append(
            f"| {horizon} | {row['reference_baseline']} | {row['reference_mae']:.4f} | "
            f"{forest} | {boosted} | {label} | {row['improvement']:+.1%} |"
        )
 
    lines+=[
        "",
        "MAE in kW. Il riferimento e' la baseline naive piu' forte dell'orizzonte: "
        "la persistenza a un'ora, la stagionalita' giornaliera piu' avanti, dove la "
        "persistenza non e' piu' disponibile al momento della previsione.",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
 
 
def plot_figure(table: pd.DataFrame)->None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, (top, bottom)=plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        height_ratios=[2, 1],
        facecolor=SURFACE,
    )
 
    horizons=table.index.to_numpy()
 
    seasonal=table["naive_24h"].dropna()
    top.plot(
        seasonal.index,
        seasonal.to_numpy(),
        color=MUTED,
        linestyle="--",
        linewidth=2,
        label="naive giornaliera (stessa ora di ieri)",
        zorder=2,
    )
    if "naive_1h" in table and pd.notna(table["naive_1h"].iloc[0]):
        persistence=table["naive_1h"].dropna()
        top.plot(
            persistence.index,
            persistence.to_numpy(),
            color=PERSISTENCE,
            linestyle="none",
            marker="D",
            markersize=8,
            label="persistenza (disponibile solo a h=1)",
            zorder=3,
        )
 
    for model, (label, color) in MODEL_STYLE.items():
        if model not in table:
            continue
        series=table[model].dropna()
        # pochi punti: solo marcatori, altrimenti una linea suggerirebbe
        # orizzonti che non sono stati addestrati
        style=(
            {"linestyle":"none", "marker":"o", "markersize":8}
            if len(series)<5
            else {"linewidth":2}
        )
        top.plot(series.index, series.to_numpy(), color=color, label=label, zorder=4, **style)
 
    top.set_ylim(0, float(seasonal.max())*1.3)
    top.set_ylabel("MAE (kW)", color=INK)
    top.set_title(
        "Errore per orizzonte: dopo 3-4 ore l'errore smette di crescere",
        color=INK,
        fontsize=13,
        loc="left",
        pad=18,
    )
    top.text(
        0.0,
        1.02,
        "l'informazione dei lag recenti si esaurisce presto; oltre, il modello vive di stagionalità",
        transform=top.transAxes,
        color=MUTED,
        fontsize=10,
    )
    top.legend(loc="lower right", frameon=False, labelcolor=INK, fontsize=9)
 
    first,last=table.index[0], table.index[-1]
    top.annotate(
        f"{table.loc[first, 'best_mae']:.3f}",
        (first, table.loc[first, "best_mae"]),
        textcoords="offset points",
        xytext=(6, -14),
        color=INK,
        fontsize=9,
    )
    top.annotate(
        f"{table.loc[last, 'best_mae']:.3f}",
        (last, table.loc[last, "best_mae"]),
        textcoords="offset points",
        xytext=(-10, 10),
        color=INK,
        fontsize=9,
    )
 
    # ogni barra prende il colore del modello che vince a quell'orizzonte
    bottom.bar(
        horizons,
        table["improvement"]*100,
        color=[MODEL_STYLE[model][1] for model in table["best_model"]],
        width=0.6,
        zorder=3,
    )
    bottom.set_ylabel("miglioramento (%)", color=INK)
    bottom.text(
        0.0,
        1.14,
        "rispetto alla baseline di riferimento dell'orizzonte: persistenza a h=1, naive giornaliera da h=2",
        transform=bottom.transAxes,
        color=MUTED,
        fontsize=9,
    )
    bottom.set_xlabel("ore di anticipo della previsione", color=INK)
    bottom.set_xticks(horizons[::2])
    peak=table["improvement"].idxmax()
    for horizon in dict.fromkeys((first, peak, last)):
        value=table.loc[horizon, "improvement"] * 100
        bottom.annotate(
            f"{value:.1f}%",
            (horizon, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            color=INK,
            fontsize=9,
        )
 
    for axis in (top, bottom):
        axis.set_facecolor(SURFACE)
        axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=1)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
 
    figure.tight_layout()
    figure.subplots_adjust(hspace=0.38)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
 
 
def main()->None:
    params=load_params()
    baseline=json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    models=json.loads(MODELS_PATH.read_text(encoding="utf-8"))
 
    table=build_table(baseline, models, params["evaluation"]["reference_baseline"])
    table.to_csv(TABLE_PATH)
    write_summary(table, baseline)
    plot_figure(table)
 
    columns=[
        column
        for column in (
            "reference_baseline",
            "reference_mae",
            "random_forest",
            "xgboost",
            "improvement",
        )
        if column in table
    ]
    print(table[columns].to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nScritti {TABLE_PATH}, {SUMMARY_PATH}, {FIGURE_PATH}")
 
 
if __name__=="__main__":
    main()