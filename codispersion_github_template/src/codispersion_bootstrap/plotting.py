from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def lag_norm_from_frame(df: pd.DataFrame) -> pd.Series:
    """Calcula ||h|| desde columnas h1/h2."""
    return np.sqrt(df["h1"].astype(float) ** 2 + df["h2"].astype(float) ** 2)


def plot_variance_ratio(
    df: pd.DataFrame,
    *,
    b_values: Sequence[int] | None = None,
    ratio_cols: Sequence[str] = ("ratio_BB", "ratio_CBB"),
    title: str = "Calibración de varianza bootstrap por lag",
):
    """Gráfico tipo paper: Var*/Var_MC vs ||h|| para BB y CBB."""
    import matplotlib.pyplot as plt

    data = df.copy()
    data["h_norm"] = lag_norm_from_frame(data)
    if b_values is None:
        b_values = sorted(data["b"].dropna().unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for b in b_values:
        sub = data[data["b"] == b].sort_values("h_norm")
        for col in ratio_cols:
            if col not in sub:
                continue
            label = f"{col.replace('ratio_', '')} b={b}"
            linestyle = "-" if "CBB" in col else "--"
            marker = "s" if "CBB" in col else "o"
            ax.plot(sub["h_norm"], sub[col], marker=marker, linestyle=linestyle, label=label)
    ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel(r"Distancia del lag $\|h\|$")
    ax.set_ylabel(r"Ratio Var*/Var$_{MC}$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig, ax


def plot_coverage(
    df: pd.DataFrame,
    *,
    b_values: Sequence[int] | None = None,
    coverage_cols: Sequence[str] = ("coverage_BB", "coverage_CBB"),
    title: str = "Cobertura de intervalos bootstrap por lag",
):
    """Gráfico tipo paper: cobertura vs ||h|| para BB y CBB."""
    import matplotlib.pyplot as plt

    data = df.copy()
    data["h_norm"] = lag_norm_from_frame(data)
    if b_values is None:
        b_values = sorted(data["b"].dropna().unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for b in b_values:
        sub = data[data["b"] == b].sort_values("h_norm")
        for col in coverage_cols:
            if col not in sub:
                continue
            label = f"{col.replace('coverage_', '')} b={b}"
            linestyle = "-" if "CBB" in col else "--"
            marker = "s" if "CBB" in col else "o"
            ax.plot(sub["h_norm"], sub[col], marker=marker, linestyle=linestyle, label=label)
    ax.axhline(0.95, color="black", linewidth=1, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel(r"Distancia del lag $\|h\|$")
    ax.set_ylabel("Cobertura")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig, ax
