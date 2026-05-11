from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .diagnostics import radial_variogram, radial_cross_variogram, variogram_toroidal
from .codispersion import codispersion_empirica


def plot_radial_pair(X, Y, label="pair", max_r=8, outpath: str | Path | None = None, show=True):
    dsX, gX = radial_variogram(X, max_r)
    dsY, gY = radial_variogram(Y, max_r)
    dsC, gC = radial_cross_variogram(X, Y, max_r)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dsX, gX, label="gamma_X")
    ax.plot(dsY, gY, label="gamma_Y")
    ax.plot(dsC, gC, label="gamma_XY")
    ax.set_xlabel("distance (px)")
    ax.set_ylabel("variogram / cross")
    ax.set_title(f"Radial variograms — {label}")
    ax.legend()
    fig.tight_layout()
    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_directionals(X, title="Directional variograms", L=4, outpath=None, show=True):
    fig, ax = plt.subplots(figsize=(7, 4))
    for base in [(1,0),(0,1),(1,1),(-1,1)]:
        xs, ys = [], []
        for k in range(1, L + 1):
            h = (base[0]*k, base[1]*k)
            xs.append(k)
            ys.append(variogram_toroidal(X, h))
        ax.plot(xs, ys, marker="o", label=f"dir {base}")
    ax.set_xlabel("k")
    ax.set_ylabel("variogram")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_contributions(X, Y, h=(1,0), title=None, outpath=None, show=True):
    res = codispersion_empirica(X, Y, h, return_contrib=True)
    S = np.full(X.shape, np.nan)
    # En toroidal + omit sin NaN, S tiene shape plano. Reconstruimos solo si tamaño coincide.
    if res.get("S") is not None and np.asarray(res["S"]).size == X.size:
        S = np.asarray(res["S"]).reshape(X.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(S)
    ax.set_title(title or f"s(i), h={h}; rho={res['rho']:.3f}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_block_selection_table(df: pd.DataFrame, outpath=None, show=True):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["b"], df["score"], marker="o", label="score")
    if "bias" in df.columns:
        ax.plot(df["b"], df["bias"], marker="s", label="bias")
    if "vario" in df.columns:
        ax.plot(df["b"], df["vario"], marker="^", label="vario")
    ax.set_xlabel("b")
    ax.set_ylabel("métrica")
    ax.set_title("Selección de tamaño de bloque")
    ax.legend()
    fig.tight_layout()
    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
