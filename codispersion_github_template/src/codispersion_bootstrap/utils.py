from __future__ import annotations

from typing import List, Tuple
import numpy as np

Lag = Tuple[int, int]


def build_H(hmax: int = 2) -> List[Lag]:
    """
    Construye el conjunto direccional corto de lags:

        H = { k(1,0), k(0,1), k(1,1), k(-1,1) : k = 1, ..., hmax }.
    """
    if hmax < 1:
        raise ValueError("hmax debe ser >= 1.")
    dirs = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    return [(k * d1, k * d2) for k in range(1, hmax + 1) for d1, d2 in dirs]


def tukey_1d(n: int, alpha: float = 0.5) -> np.ndarray:
    """Ventana Tukey 1D compatible con NumPy puro."""
    n = int(n)
    if n < 1:
        raise ValueError("n debe ser positivo.")
    if n == 1:
        return np.ones(1, dtype=float)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0:
        return np.ones(n, dtype=float)
    if alpha >= 1:
        return np.hanning(n).astype(float)

    w = np.ones(n, dtype=float)
    edge = int(alpha * (n - 1) / 2.0)
    if edge > 0:
        t = np.arange(edge, dtype=float) / edge
        taper = 0.5 * (1.0 - np.cos(np.pi * t))
        w[:edge] = taper
        w[-edge:] = taper[::-1]
    return w


def tukey2d(n1: int, n2: int, k_px: int = 8) -> np.ndarray:
    """
    Ventana Tukey 2D separable. `k_px` controla aproximadamente la rampa de borde.
    """
    if k_px < 0:
        raise ValueError("k_px debe ser no negativo.")
    a1 = min(1.0, max(0.0, 2.0 * k_px / max(n1 - 1, 1)))
    a2 = min(1.0, max(0.0, 2.0 * k_px / max(n2 - 1, 1)))
    return np.outer(tukey_1d(n1, a1), tukey_1d(n2, a2))


def standardize(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Estandariza un campo: media 0 y desviación estándar 1."""
    Z = np.asarray(Z, dtype=float)
    return (Z - np.mean(Z)) / (np.std(Z) + eps)


def standardize_pair(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Estandariza dos campos por separado."""
    return standardize(X, eps=eps), standardize(Y, eps=eps)


def apply_tukey(X: np.ndarray, Y: np.ndarray, k_px: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Aplica la misma ventana Tukey 2D a ambos campos."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("X e Y deben tener el mismo tamaño.")
    W = tukey2d(*X.shape, k_px=k_px)
    return X * W, Y * W
