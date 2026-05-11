from __future__ import annotations

from typing import Dict, Iterable, Tuple
import numpy as np

Lag = Tuple[int, int]


def delta_h_toroidal(Z: np.ndarray, h: Lag) -> np.ndarray:
    """
    Incremento toroidal Delta_h Z(s) = Z(s ⊕ h) - Z(s).

    Se conserva la convención usada en el primer estudio: `np.roll(..., shift=h)`.
    """
    Z = np.asarray(Z, dtype=float)
    h1, h2 = h
    Z_shift = np.roll(np.roll(Z, shift=h1, axis=0), shift=h2, axis=1)
    return Z_shift - Z


def _delta_h_interior(Z: np.ndarray, h: Lag) -> np.ndarray:
    """Incrementos usando solo pares interiores."""
    Z = np.asarray(Z, dtype=float)
    n1, n2 = Z.shape
    h1, h2 = h

    if abs(h1) >= n1 or abs(h2) >= n2:
        return np.empty((0, 0), dtype=float)

    i0 = slice(0, n1 - h1) if h1 >= 0 else slice(-h1, n1)
    j0 = slice(0, n2 - h2) if h2 >= 0 else slice(-h2, n2)
    i1 = slice(h1, n1) if h1 >= 0 else slice(0, n1 + h1)
    j1 = slice(h2, n2) if h2 >= 0 else slice(0, n2 + h2)
    return Z[i1, j1] - Z[i0, j0]


def codisp_rho_hat(
    X: np.ndarray,
    Y: np.ndarray,
    h: Lag,
    *,
    mode: str = "toroidal",
    return_contrib: bool = False,
) -> tuple[float, np.ndarray | None]:
    """
    Estima rho_hat(h) como promedio de contribuciones locales.

    Parameters
    ----------
    X, Y:
        Campos 2D del mismo tamaño.
    h:
        Lag `(h1, h2)`.
    mode:
        "toroidal" o "interior".
    return_contrib:
        Si es True, retorna también el mapa de contribuciones locales.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("X e Y deben tener el mismo tamaño.")

    if mode == "toroidal":
        dX = delta_h_toroidal(X, h)
        dY = delta_h_toroidal(Y, h)
    elif mode == "interior":
        dX = _delta_h_interior(X, h)
        dY = _delta_h_interior(Y, h)
    else:
        raise ValueError("mode debe ser 'toroidal' o 'interior'.")

    if dX.size == 0 or dY.size == 0:
        S = np.full_like(dX, np.nan, dtype=float)
        return (np.nan, S) if return_contrib else (np.nan, None)

    Abar = float(np.mean(dX ** 2))
    Bbar = float(np.mean(dY ** 2))
    if Abar <= 0 or Bbar <= 0:
        S = np.full_like(dX, np.nan, dtype=float)
        return (np.nan, S) if return_contrib else (np.nan, None)

    S = (dX * dY) / np.sqrt(Abar * Bbar)
    rho = float(np.mean(S))
    return (rho, S) if return_contrib else (rho, None)


def codispersion_by_lag(
    X: np.ndarray,
    Y: np.ndarray,
    H: Iterable[Lag],
    *,
    mode: str = "toroidal",
) -> Dict[str, float]:
    """Calcula rho_hat(h) para todos los lags en H."""
    out: Dict[str, float] = {}
    for h in H:
        rho, _ = codisp_rho_hat(X, Y, h, mode=mode, return_contrib=False)
        out[str(tuple(h))] = float(rho)
    return out
