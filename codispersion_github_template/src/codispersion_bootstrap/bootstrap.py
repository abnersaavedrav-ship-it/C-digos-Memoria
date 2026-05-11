from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from .codispersion import codisp_rho_hat

Lag = Tuple[int, int]


def _validate_pair(X: np.ndarray, Y: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("X e Y deben tener el mismo tamaño.")
    if not isinstance(b, int) or b < 1:
        raise ValueError("b debe ser entero positivo.")
    if b > max(X.shape):
        raise ValueError("b no debería exceder el tamaño máximo de la grilla.")
    return X, Y


def cbb_sample_pair(X: np.ndarray, Y: np.ndarray, b: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Circular Block Bootstrap 2D pareado con wrap toroidal.

    Usa los mismos orígenes de bloque para X e Y, preservando dependencia local conjunta.
    """
    X, Y = _validate_pair(X, Y, b)
    n1, n2 = X.shape
    g1 = int(np.ceil(n1 / b))
    g2 = int(np.ceil(n2 / b))

    Xs = np.empty((g1 * b, g2 * b), dtype=float)
    Ys = np.empty_like(Xs)

    for u in range(g1):
        for v in range(g2):
            i0 = int(rng.integers(0, n1))
            j0 = int(rng.integers(0, n2))
            I = (i0 + np.arange(b)) % n1
            J = (j0 + np.arange(b)) % n2
            r0, c0 = u * b, v * b
            Xs[r0:r0 + b, c0:c0 + b] = X[np.ix_(I, J)]
            Ys[r0:r0 + b, c0:c0 + b] = Y[np.ix_(I, J)]

    return Xs[:n1, :n2], Ys[:n1, :n2]


def _list_overlapping_origins(n1: int, n2: int, b: int) -> List[Lag]:
    if b > n1 or b > n2:
        raise ValueError("Para BB solapado sin wrap, b debe ser <= min(n1, n2).")
    return [(i, j) for i in range(n1 - b + 1) for j in range(n2 - b + 1)]


def bb_lahiri_sample_pair(X: np.ndarray, Y: np.ndarray, b: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap de bloques solapados 2D sin wrap, estilo Lahiri.

    Devuelve una grilla recortada a múltiplos de b: (floor(n1/b)b, floor(n2/b)b).
    """
    X, Y = _validate_pair(X, Y, b)
    n1, n2 = X.shape
    M1 = (n1 // b) * b
    M2 = (n2 // b) * b
    if M1 == 0 or M2 == 0:
        raise ValueError("b es demasiado grande para la grilla.")

    origins = _list_overlapping_origins(n1, n2, b)
    Xs = np.empty((M1, M2), dtype=float)
    Ys = np.empty_like(Xs)

    for u in range(M1 // b):
        for v in range(M2 // b):
            i0, j0 = origins[int(rng.integers(0, len(origins)))]
            r0, c0 = u * b, v * b
            Xs[r0:r0 + b, c0:c0 + b] = X[i0:i0 + b, j0:j0 + b]
            Ys[r0:r0 + b, c0:c0 + b] = Y[i0:i0 + b, j0:j0 + b]

    return Xs, Ys


def bootstrap_codispersion(
    X: np.ndarray,
    Y: np.ndarray,
    H: List[Lag],
    b: int,
    B: int = 800,
    seed: int | None = 12345,
    *,
    sampler: str = "CBB",
    mode_rho: str = "toroidal",
    return_samples: bool = False,
) -> Dict[str, Dict[str, float | np.ndarray]]:
    """
    Bootstrap de rho_hat(h) para cada h en H.

    `sampler` puede ser "CBB" o "BB".
    """
    if B < 2:
        raise ValueError("B debe ser al menos 2 para estimar varianza bootstrap.")

    rng = np.random.default_rng(seed)
    rho_star = {tuple(h): np.empty(B, dtype=float) for h in H}

    for k in range(B):
        if sampler.upper() == "CBB":
            Xs, Ys = cbb_sample_pair(X, Y, b, rng)
        elif sampler.upper() == "BB":
            Xs, Ys = bb_lahiri_sample_pair(X, Y, b, rng)
        else:
            raise ValueError("sampler debe ser 'CBB' o 'BB'.")

        for h in H:
            rho, _ = codisp_rho_hat(Xs, Ys, tuple(h), mode=mode_rho, return_contrib=False)
            rho_star[tuple(h)][k] = rho

    results: Dict[str, Dict[str, float | np.ndarray]] = {}
    for h, arr in rho_star.items():
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            entry: Dict[str, float | np.ndarray] = {"var_boot": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
        else:
            lo, hi = np.percentile(finite, [2.5, 97.5])
            entry = {
                "var_boot": float(np.var(finite, ddof=1)),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
            }
        if return_samples:
            entry["samples"] = arr.copy()
        results[str(h)] = entry
    return results
