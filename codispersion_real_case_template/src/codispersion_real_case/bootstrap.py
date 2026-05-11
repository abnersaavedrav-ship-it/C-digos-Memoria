from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .codispersion import codispersion_xy
from .diagnostics import roll_diff


def cbb_sample_pair(
    X: np.ndarray,
    Y: np.ndarray,
    b: int,
    scheme: str = "CBB",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remuestreo pareado por bloques 2D: CBB toroidal o MBB interior."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape != Y.shape:
        raise ValueError("X e Y deben tener el mismo tamaño")
    n1, n2 = X.shape
    b = int(b)
    if b <= 0:
        raise ValueError("b debe ser positivo")
    if rng is None:
        rng = np.random.default_rng()

    G1 = int(np.ceil(n1 / b))
    G2 = int(np.ceil(n2 / b))
    H1 = G1 * b
    H2 = G2 * b
    Xd = np.empty((H1, H2), dtype=np.float64)
    Yd = np.empty((H1, H2), dtype=np.float64)

    scheme = scheme.upper()
    if scheme == "CBB":
        for u in range(G1):
            for v in range(G2):
                i0 = int(rng.integers(0, n1))
                j0 = int(rng.integers(0, n2))
                ridx = (i0 + np.arange(b)) % n1
                cidx = (j0 + np.arange(b)) % n2
                Xd[u*b:(u+1)*b, v*b:(v+1)*b] = X[np.ix_(ridx, cidx)]
                Yd[u*b:(u+1)*b, v*b:(v+1)*b] = Y[np.ix_(ridx, cidx)]
    elif scheme == "MBB":
        if b > n1 or b > n2:
            raise ValueError("En MBB se requiere b <= min(n1, n2)")
        for u in range(G1):
            for v in range(G2):
                i0 = int(rng.integers(0, n1 - b + 1))
                j0 = int(rng.integers(0, n2 - b + 1))
                Xd[u*b:(u+1)*b, v*b:(v+1)*b] = X[i0:i0+b, j0:j0+b]
                Yd[u*b:(u+1)*b, v*b:(v+1)*b] = Y[i0:i0+b, j0:j0+b]
    else:
        raise ValueError("scheme debe ser 'CBB' o 'MBB'")
    return Xd[:n1, :n2], Yd[:n1, :n2]


def boot_codispersion(
    X: np.ndarray,
    Y: np.ndarray,
    H: List[Tuple[int, int]],
    B: int,
    b: int,
    scheme: str = "CBB",
    alpha: float = 0.05,
    seed: Optional[int] = None,
    mode_estimator: str = "toroidal",
    min_pairs: int = 1,
    nan_policy: str = "omit",
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Bootstrap espacial pareado para rho_hat(h)."""
    rng = np.random.default_rng(seed)
    H = [tuple(h) for h in H]
    est0 = codispersion_xy(X, Y, H, mode=mode_estimator, min_pairs=min_pairs, nan_policy=nan_policy)
    rho_star = {h: np.empty(int(B), dtype=np.float64) for h in H}

    for k in range(int(B)):
        Xs, Ys = cbb_sample_pair(X, Y, b=b, scheme=scheme, rng=rng)
        estk = codispersion_xy(Xs, Ys, H, mode=mode_estimator, min_pairs=min_pairs, nan_policy=nan_policy)
        for h in H:
            if estk[h]["status"] == "ok" and np.isfinite(estk[h]["rho"]):
                rho_star[h][k] = float(estk[h]["rho"])
            else:
                rho_star[h][k] = np.nan

    out: Dict[Tuple[int, int], Dict[str, object]] = {}
    for h in H:
        mask = np.isfinite(rho_star[h])
        B_eff = int(np.count_nonzero(mask))
        if est0[h]["status"] != "ok":
            out[h] = dict(rho=np.nan, rho_star=rho_star[h], var_star=np.nan, se_star=np.nan,
                          ci=(np.nan, np.nan), Abar=est0[h]["Abar"], Bbar=est0[h]["Bbar"],
                          m=est0[h]["m"], status=est0[h]["status"], b=int(b), scheme=scheme)
            continue
        rho0 = float(est0[h]["rho"])
        if B_eff < max(10, int(0.5 * int(B))):
            var_star = np.nan
            se_star = np.nan
            ci = (np.nan, np.nan)
        else:
            boots = rho_star[h][mask]
            var_star = float(np.var(boots, ddof=1))
            se_star = float(np.sqrt(var_star))
            lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
            ci = (float(lo), float(hi))
        out[h] = dict(rho=rho0, rho_star=rho_star[h], var_star=var_star, se_star=se_star, ci=ci,
                      Abar=est0[h]["Abar"], Bbar=est0[h]["Bbar"], m=est0[h]["m"], status="ok",
                      b=int(b), scheme=scheme)
    return out


def _semi_variogram_radial(Z: np.ndarray, maxlag: int) -> np.ndarray:
    gam = np.zeros(maxlag + 1, dtype=np.float64)
    for d in range(1, maxlag + 1):
        vals = []
        for h in [(d, 0), (0, d), (d, d), (d, -d)]:
            diff = roll_diff(Z, *h)
            vals.append(0.5 * np.mean(diff * diff))
        gam[d] = np.mean(vals)
    return gam


def validate_mosaic(
    X: np.ndarray,
    Y: np.ndarray,
    b: int,
    sampler: str = "CBB",
    R: int = 20,
    H_small: Optional[List[Tuple[int, int]]] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Diagnóstico de preservación de estructura local del mosaico."""
    rng = np.random.default_rng(seed)
    n1, n2 = X.shape
    maxlag = max(3, min(10, min(n1, n2) // 8))
    gamX = _semi_variogram_radial(X, maxlag)
    gamY = _semi_variogram_radial(Y, maxlag)
    gamX_star = np.zeros_like(gamX)
    gamY_star = np.zeros_like(gamY)
    for _ in range(int(R)):
        Xs, Ys = cbb_sample_pair(X, Y, b=b, scheme=sampler, rng=rng)
        gamX_star += _semi_variogram_radial(Xs, maxlag)
        gamY_star += _semi_variogram_radial(Ys, maxlag)
    gamX_star /= max(1, int(R))
    gamY_star /= max(1, int(R))

    def rel_L2(a, b):
        num = np.sqrt(np.mean((a - b) ** 2))
        den = np.sqrt(np.mean(a ** 2) + 1e-12)
        return float(num / (den + 1e-12))

    diag = {
        "maxlag": maxlag,
        "gammaX_orig": gamX,
        "gammaY_orig": gamY,
        "gammaX_star_mean": gamX_star,
        "gammaY_star_mean": gamY_star,
        "relL2_gammaX": rel_L2(gamX, gamX_star),
        "relL2_gammaY": rel_L2(gamY, gamY_star),
        "b": int(b),
        "scheme": sampler,
    }
    if H_small:
        est0 = codispersion_xy(X, Y, H_small, mode="toroidal", min_pairs=1, nan_policy="omit")
        rho0 = np.array([est0[h]["rho"] if est0[h]["status"] == "ok" else np.nan for h in H_small], dtype=np.float64)
        rho_star_mean = np.zeros_like(rho0)
        for _ in range(int(R)):
            Xs, Ys = cbb_sample_pair(X, Y, b=b, scheme=sampler, rng=rng)
            estk = codispersion_xy(Xs, Ys, H_small, mode="toroidal", min_pairs=1, nan_policy="omit")
            rho_k = np.array([estk[h]["rho"] if estk[h]["status"] == "ok" else np.nan for h in H_small], dtype=np.float64)
            rho_star_mean += np.nan_to_num(rho_k, nan=0.0)
        rho_star_mean /= max(1, int(R))
        diag["rho_orig_Hsmall"] = rho0
        diag["rho_star_mean_Hsmall"] = rho_star_mean
    return diag


def results_to_table(res: Dict[Tuple[int, int], Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for h, r in res.items():
        lo, hi = r["ci"]
        rows.append({
            "h1": h[0], "h2": h[1], "rho": r["rho"], "SE_boot": r["se_star"],
            "IC95_lo": lo, "IC95_hi": hi, "m": r["m"], "b": r["b"], "scheme": r["scheme"],
        })
    return pd.DataFrame(rows).sort_values(["h1", "h2"]).reset_index(drop=True)
