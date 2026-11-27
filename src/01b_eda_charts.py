#!/usr/bin/env python3
"""Análisis exploratorio visual (EDA) de socavones en la ZMG (2020-2026).

Genera gráficas limpias y de alta resolución en outputs/eda/:
  - 01_socavones_por_anio.png
  - 02_socavones_por_zona.png
  - 03_socavones_por_temporada.png
  - 04_tipo_y_tamano_socavon.png
  - 00_dashboard_socavones_zmg.png (resumen 2x2)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
MESES_LLUVIA = {6, 7, 8, 9, 10}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#cccccc",
            "axes.grid": True,
            "grid.color": "#e5e5e5",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )


def load_data(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
    df["temporada"] = np.where(df["mes"].isin(MESES_LLUVIA), "Lluvias (Jun–Oct)", "Secas (Nov–May)")
    df["colonia_clean"] = df["colonia"].fillna("").astype(str).str.strip()
    df["colonia_clean"] = df["colonia_clean"].replace({"": "Sin colonia registrada"})
    df["causa_clean"] = df["causa_raw"].fillna("Sin especificar").astype(str).str.strip()
    df["tamano_label"] = df["tamano"].map(
        {
            "grande": "Grande (> 3 m)",
            "mediano": "Mediano (1–3 m)",
            "pequeno": "Pequeño (< 1 m)",
            "desconocido": "Sin dato",
        }
    ).fillna("Sin dato")
    return df


def plot_por_anio(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["anio"].value_counts().sort_index()
    years = [int(y) for y in counts.index]
    vals = counts.values

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    colors = ["#4c78a8" if y < 2026 else "#e45756" for y in years]
    bars = ax.bar([str(y) for y in years], vals, color=colors, width=0.6, edgecolor="#333333", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.9, f"{int(v)}", ha="center", va="bottom", fontweight="bold")

    ax.set_title(f"Socavones registrados por año en la ZMG (n = {len(df)})")
    ax.set_xlabel("Año")
    ax.set_ylabel("Número de socavones")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    fig.savefig(out_dir / "01_socavones_por_anio.png", bbox_inches="tight")
    plt.close(fig)


def plot_por_zona(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=160)

    # Panel 1: Municipio
    mun_counts = df["municipio"].value_counts().sort_values()
    bars = axes[0].barh(mun_counts.index, mun_counts.values, color="#2a9d8f", edgecolor="#333333", linewidth=0.5)
    for bar, v in zip(bars, mun_counts.values):
        pct = 100.0 * v / len(df)
        axes[0].text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{int(v)} ({pct:.1f}%)", va="center", fontsize=9)
    axes[0].set_title("Socavones por municipio")
    axes[0].set_xlabel("Número de socavones")
    axes[0].set_xlim(0, mun_counts.max() * 1.25)
    axes[0].grid(axis="y", visible=False)

    # Panel 2: Top 12 colonias (excluyendo vacías)
    col_valid = df[df["colonia_clean"] != "Sin colonia registrada"]["colonia_clean"]
    top_cols = col_valid.value_counts().head(12).sort_values()
    bars2 = axes[1].barh(top_cols.index, top_cols.values, color="#e76f51", edgecolor="#333333", linewidth=0.5)
    for bar, v in zip(bars2, top_cols.values):
        axes[1].text(v + 0.15, bar.get_y() + bar.get_height() / 2, f"{int(v)}", va="center", fontsize=9, fontweight="bold")
    axes[1].set_title("Top 12 colonias con más socavones")
    axes[1].set_xlabel("Número de socavones")
    axes[1].set_xlim(0, top_cols.max() * 1.2)
    axes[1].grid(axis="y", visible=False)

    fig.suptitle(f"Distribución espacial de socavones en la ZMG (n = {len(df)})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "02_socavones_por_zona.png", bbox_inches="tight")
    plt.close(fig)


def plot_por_temporada(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160, gridspec_kw={"width_ratios": [2.2, 1]})

    # Panel 1: Mes a mes
    m_counts = df["mes"].value_counts().reindex(range(1, 13), fill_value=0)
    bar_colors = ["#1d70b8" if m in MESES_LLUVIA else "#e9c46a" for m in range(1, 13)]
    bars = axes[0].bar(MESES_ES, m_counts.values, color=bar_colors, edgecolor="#333333", linewidth=0.5)
    for bar, v in zip(bars, m_counts.values):
        if v > 0:
            axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.7, f"{int(v)}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_title("Socavones por mes (Azul = Temporada de lluvias Jun–Oct)")
    axes[0].set_xlabel("Mes")
    axes[0].set_ylabel("Número de socavones")
    axes[0].set_ylim(0, m_counts.max() * 1.18)
    axes[0].grid(axis="x", visible=False)

    # Panel 2: Lluvias vs Secas
    temp_counts = df["temporada"].value_counts().reindex(["Lluvias (Jun–Oct)", "Secas (Nov–May)"], fill_value=0)
    wedges, texts, autotexts = axes[1].pie(
        temp_counts.values,
        labels=temp_counts.index,
        colors=["#1d70b8", "#e9c46a"],
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * len(df) / 100))})",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    axes[1].set_title("Proporción por temporada")

    fig.suptitle(f"Estacionalidad de socavones en la ZMG (2020–2026, n = {len(df)})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "03_socavones_por_temporada.png", bbox_inches="tight")
    plt.close(fig)


def plot_tipo_y_tamano(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=160)

    # Panel 1: Causa / tipo de socavón
    causa_counts = df["causa_clean"].value_counts().sort_values()
    palette_causa = ["#6c757d", "#f4a261", "#2a9d8f", "#1d70b8", "#e63946"][-len(causa_counts):]
    bars1 = axes[0].barh(causa_counts.index, causa_counts.values, color=palette_causa, edgecolor="#333333", linewidth=0.5)
    for bar, v in zip(bars1, causa_counts.values):
        pct = 100.0 * v / len(df)
        axes[0].text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{int(v)} ({pct:.1f}%)", va="center", fontsize=9)
    axes[0].set_title("Tipo / causa aparente del socavón")
    axes[0].set_xlabel("Número de socavones")
    axes[0].set_xlim(0, causa_counts.max() * 1.28)
    axes[0].grid(axis="y", visible=False)

    # Panel 2: Tamaño apilado por causa
    order_size = ["Pequeño (< 1 m)", "Mediano (1–3 m)", "Grande (> 3 m)", "Sin dato"]
    ct = pd.crosstab(df["tamano_label"], df["causa_clean"]).reindex([s for s in order_size if s in df["tamano_label"].values])
    ct.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2", edgecolor="#333333", linewidth=0.5)
    for i, total in enumerate(ct.sum(axis=1).values):
        axes[1].text(i, total + 1.0, f"{int(total)}", ha="center", va="bottom", fontweight="bold")
    axes[1].set_title("Tamaño del socavón desglosado por causa")
    axes[1].set_xlabel("Categoría de tamaño")
    axes[1].set_ylabel("Número de socavones")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylim(0, ct.sum(axis=1).max() * 1.2)
    axes[1].grid(axis="x", visible=False)
    axes[1].legend(title="Causa", fontsize=8, title_fontsize=8.5)

    fig.suptitle(f"Tipos y tamaños de socavones en la ZMG (n = {len(df)})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "04_tipo_y_tamano_socavon.png", bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=160)

    # (0,0) Año
    counts = df["anio"].value_counts().sort_index()
    years = [int(y) for y in counts.index]
    bars = axes[0, 0].bar([str(y) for y in years], counts.values, color=["#4c78a8" if y < 2026 else "#e45756" for y in years], edgecolor="#333", lw=0.5)
    for b, v in zip(bars, counts.values):
        axes[0, 0].text(b.get_x() + b.get_width() / 2, v + 0.9, f"{int(v)}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    axes[0, 0].set_title("1. Por año (2020–2026)")
    axes[0, 0].set_ylabel("Socavones")
    axes[0, 0].set_ylim(0, counts.max() * 1.18)
    axes[0, 0].grid(axis="x", visible=False)

    # (0,1) Zona (Municipio)
    mun_counts = df["municipio"].value_counts().sort_values()
    bars = axes[0, 1].barh(mun_counts.index, mun_counts.values, color="#2a9d8f", edgecolor="#333", lw=0.5)
    for b, v in zip(bars, mun_counts.values):
        axes[0, 1].text(v + 0.8, b.get_y() + b.get_height() / 2, f"{int(v)} ({100*v/len(df):.0f}%)", va="center", fontsize=9)
    axes[0, 1].set_title("2. Por zona / municipio")
    axes[0, 1].set_xlim(0, mun_counts.max() * 1.25)
    axes[0, 1].grid(axis="y", visible=False)

    # (1,0) Temporada (mes)
    m_counts = df["mes"].value_counts().reindex(range(1, 13), fill_value=0)
    bar_colors = ["#1d70b8" if m in MESES_LLUVIA else "#e9c46a" for m in range(1, 13)]
    bars = axes[1, 0].bar(MESES_ES, m_counts.values, color=bar_colors, edgecolor="#333", lw=0.5)
    for b, v in zip(bars, m_counts.values):
        if v > 0:
            axes[1, 0].text(b.get_x() + b.get_width() / 2, v + 0.6, f"{int(v)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    lluvia_pct = 100.0 * df["mes"].isin(MESES_LLUVIA).mean()
    axes[1, 0].set_title(f"3. Por temporada ({lluvia_pct:.1f}% en lluvias Jun–Oct)")
    axes[1, 0].set_ylabel("Socavones")
    axes[1, 0].set_ylim(0, m_counts.max() * 1.18)
    axes[1, 0].grid(axis="x", visible=False)

    # (1,1) Tipo de socavón (causa)
    causa_counts = df["causa_clean"].value_counts().sort_values()
    bars = axes[1, 1].barh(causa_counts.index, causa_counts.values, color="#e76f51", edgecolor="#333", lw=0.5)
    for b, v in zip(bars, causa_counts.values):
        axes[1, 1].text(v + 0.8, b.get_y() + b.get_height() / 2, f"{int(v)} ({100*v/len(df):.1f}%)", va="center", fontsize=9)
    axes[1, 1].set_title("4. Por tipo / causa aparente")
    axes[1, 1].set_xlim(0, causa_counts.max() * 1.28)
    axes[1, 1].grid(axis="y", visible=False)

    fig.suptitle(f"Resumen Exploratorio — SocavonesGDL (n = {len(df)} socavones mapeados)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "00_dashboard_socavones_zmg.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera gráficas EDA de socavones por año, zona, temporada y tipo.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/sinkholes_clean.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eda"))
    args = parser.parse_args()

    setup_style()
    df = load_data(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_por_anio(df, args.out_dir)
    plot_por_zona(df, args.out_dir)
    plot_por_temporada(df, args.out_dir)
    plot_tipo_y_tamano(df, args.out_dir)
    plot_dashboard(df, args.out_dir)

    print(f"5 gráficas guardadas en: {args.out_dir.resolve()}")
    lluvia_n = int(df["mes"].isin(MESES_LLUVIA).sum())
    print(f"Lluvias (Jun-Oct): {lluvia_n}/{len(df)} ({100*lluvia_n/len(df):.1f}%)")
    print("Tamaños:\n", df["tamano_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
