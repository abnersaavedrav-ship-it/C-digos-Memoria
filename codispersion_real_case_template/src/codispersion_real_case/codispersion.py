from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .diagnostics import roll_diff


def _increments_toroidal(Z: np.ndarray, h: Tuple[int, int]) -> np.ndarray:
    return roll_diff(Z, int(h[0]), int(h[1])).astype(np.float64)


def _slices_interior(n1: int, n2: int, h1: int, h2: int):
    i0 = max(0, h1)
    i1 = n1 + min(0, h1)
    j0 = max(0, h2)
    j1 = n2 + min(0, h2)
    S0 = np.s_[i0:i1, j0:j1]
    S1 = np.s_[i0 - h1:i1 - h1, j0 - h2:j1 - h2]
    return S0, S1


def codispersion_empirica(
    X: np.ndarray,
    Y: np.ndarray,
    h: Tuple[int, int],
    mode: str = "toroidal",
    min_pairs: int = 1,
    nan_policy: str = "omit",
    return_contrib: bool = True,
) -> Dict[str, object]:
    """Estimador rho_hat(h) como promedio de contribuciones locales."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape != Y.shape:
        raise ValueError("X e Y deben tener el mismo tamaño")
    h1, h2 = int(h[0]), int(h[1])

    if mode == "toroidal":
        DX = _increments_toroidal(X, (h1, h2))
        DY = _increments_toroidal(Y, (h1, h2))
    elif mode == "interior":
        S0, S1 = _slices_interior(*X.shape, h1, h2)
        DX = X[S1] - X[S0]
        DY = Y[S1] - Y[S0]
    else:
        raise ValueError("mode debe ser 'toroidal' o 'interior'")

    if nan_policy == "omit":
        mask = np.isfinite(DX) & np.isfinite(DY)
        DX = DX[mask]
        DY = DY[mask]
    elif nan_policy == "propagate":
        if not (np.isfinite(DX).all() and np.isfinite(DY).all()):
            return {"rho": np.nan, "Abar": np.nan, "Bbar": np.nan, "m": 0, "S": None, "status": "nan_present"}
    else:
        raise ValueError("nan_policy debe ser 'omit' o 'propagate'")

    m = DX.size
    if m < min_pairs:
        return {"rho": np.nan, "Abar": np.nan, "Bbar": np.nan, "m": int(m), "S": None, "status": "too_few_pairs"}

    Abar = float(np.mean(DX * DX))
    Bbar = float(np.mean(DY * DY))
    if not (Abar > 0.0 and Bbar > 0.0):
        return {"rho": np.nan, "Abar": Abar, "Bbar": Bbar, "m": int(m), "S": None, "status": "zero_variance"}

    S = (DX * DY) / math.sqrt(Abar * Bbar)
    rho = float(np.clip(np.mean(S), -1.0, 1.0))
    out = {"rho": rho, "Abar": Abar, "Bbar": Bbar, "m": int(m), "status": "ok"}
    if return_contrib:
        out["S"] = S
    return out


def codispersion_sobre_H(X: np.ndarray, Y: np.ndarray, H: Iterable[Tuple[int, int]], **kwargs) -> List[Dict[str, object]]:
    return [dict(h=tuple(h), **codispersion_empirica(X, Y, h, **kwargs)) for h in H]


def codispersion_xy(
    X: np.ndarray,
    Y: np.ndarray,
    H: List[Tuple[int, int]],
    mode: str = "toroidal",
    min_pairs: int = 1,
    nan_policy: str = "omit",
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """API congelada: devuelve estimación por lag."""
    return {tuple(h): codispersion_empirica(X, Y, h, mode=mode, min_pairs=min_pairs, nan_policy=nan_policy, return_contrib=True) for h in H}
