from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def roll_diff(X: np.ndarray, h1: int, h2: int) -> np.ndarray:
    """Incremento toroidal: Z(s+h)-Z(s)."""
    return np.roll(X, shift=(int(h1), int(h2)), axis=(0, 1)) - X


def border_center_stats(X: np.ndarray, k: int = 10) -> Dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    n1, n2 = X.shape
    border = np.zeros_like(X, dtype=bool)
    border[:k, :] = True
    border[-k:, :] = True
    border[:, :k] = True
    border[:, -k:] = True
    center = ~border
    eps = 1e-12
    return {
        "border_mean": float(X[border].mean()),
        "center_mean": float(X[center].mean()),
        "rel_diff_mean_%": float(100 * (X[border].mean() - X[center].mean()) / (X[center].mean() + eps)),
        "border_std": float(X[border].std()),
        "center_std": float(X[center].std()),
    }


def tile_stats(X: np.ndarray, T: int = 8) -> Dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    n1, n2 = X.shape
    t1, t2 = n1 // T, n2 // T
    means, vars_ = [], []
    for i in range(T):
        for j in range(T):
            block = X[i*t1:(i+1)*t1, j*t2:(j+1)*t2]
            means.append(block.mean())
            vars_.append(block.var())
    means = np.asarray(means)
    vars_ = np.asarray(vars_)
    return {
        "tile_mean_mean": float(means.mean()),
        "tile_mean_std": float(means.std()),
        "tile_var_mean": float(vars_.mean()),
        "tile_var_min": float(vars_.min()),
        "tile_var_max": float(vars_.max()),
        "tile_var_ratio_max_min": float(vars_.max() / (vars_.min() + 1e-12)),
    }


def tile_var_of_increments(X: np.ndarray, h: Tuple[int, int] = (1, 0), T: int = 8) -> Dict[str, object]:
    D = roll_diff(X, *h)
    n1, n2 = D.shape
    t1, t2 = n1 // T, n2 // T
    vals = []
    for i in range(T):
        for j in range(T):
            vals.append(D[i*t1:(i+1)*t1, j*t2:(j+1)*t2].var())
    vals = np.asarray(vals)
    return {"h": tuple(h), "mean_var": float(vals.mean()), "ratio_max_min": float(vals.max() / (vals.min() + 1e-12))}


def variogram_toroidal(X: np.ndarray, h: Tuple[int, int]) -> float:
    D = roll_diff(X, *h)
    return float((D * D).mean())


def cross_variogram_toroidal(X: np.ndarray, Y: np.ndarray, h: Tuple[int, int]) -> float:
    DX = roll_diff(X, *h)
    DY = roll_diff(Y, *h)
    return float((DX * DY).mean())


def radial_variogram(X: np.ndarray, max_r: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    vals: dict[float, list[float]] = {}
    for h1 in range(-max_r, max_r + 1):
        for h2 in range(-max_r, max_r + 1):
            if h1 == 0 and h2 == 0:
                continue
            d = math.hypot(h1, h2)
            if d > max_r:
                continue
            vals.setdefault(round(d, 5), []).append(variogram_toroidal(X, (h1, h2)))
    ds = sorted(vals)
    gammas = [float(np.mean(vals[d])) for d in ds]
    return np.array(ds), np.array(gammas)


def radial_cross_variogram(X: np.ndarray, Y: np.ndarray, max_r: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    vals: dict[float, list[float]] = {}
    for h1 in range(-max_r, max_r + 1):
        for h2 in range(-max_r, max_r + 1):
            if h1 == 0 and h2 == 0:
                continue
            d = math.hypot(h1, h2)
            if d > max_r:
                continue
            vals.setdefault(round(d, 5), []).append(cross_variogram_toroidal(X, Y, (h1, h2)))
    ds = sorted(vals)
    gammas = [float(np.mean(vals[d])) for d in ds]
    return np.array(ds), np.array(gammas)


def effective_range(ds: np.ndarray, gammas: np.ndarray, pct: float = 0.95) -> float:
    k = min(3, len(gammas))
    sill = float(np.mean(np.sort(gammas)[-k:]))
    thr = pct * sill
    for d, g in zip(ds, gammas):
        if g >= thr:
            return float(d)
    return float(ds[-1])


def propose_H(r_eff: float, cap: int = 2) -> Tuple[List[Tuple[int, int]], int]:
    hmax = min(max(1, int(np.floor(0.5 * r_eff))), cap)
    H = set()
    for base in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
        for k in range(1, hmax + 1):
            H.add((base[0] * k, base[1] * k))
    return sorted(H), hmax


def codisp_table(X: np.ndarray, Y: np.ndarray, H: Iterable[Tuple[int, int]]) -> pd.DataFrame:
    from .codispersion import codispersion_empirica
    rows = []
    for h in H:
        res = codispersion_empirica(X, Y, h, mode="toroidal", return_contrib=False)
        rows.append({"h": tuple(h), "rho_hat": res["rho"], "m": res["m"], "status": res["status"]})
    return pd.DataFrame(rows)


def _basic_stats_table(images: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, X in images.items():
        rows.append({
            "image": name,
            "shape": f"{X.shape[0]}x{X.shape[1]}",
            "min": float(np.nanmin(X)),
            "max": float(np.nanmax(X)),
            "mean": float(np.nanmean(X)),
            "std": float(np.nanstd(X)),
        })
    return pd.DataFrame(rows)


def real_diagnostics(images: Dict[str, np.ndarray], *, max_r: int = 8, tile_T: int = 8) -> Dict[str, object]:
    """Diagnósticos para imágenes reales ya cargadas/preprocesadas."""
    basic = _basic_stats_table(images)
    bvc_rows, tile_rows, inc_rows = [], [], []
    r_eff = {}
    for name, X in images.items():
        s = border_center_stats(X, k=10); s["image"] = name; bvc_rows.append(s)
        t = tile_stats(X, T=tile_T); t["image"] = name; tile_rows.append(t)
        for h in [(1, 0), (0, 1)]:
            r = tile_var_of_increments(X, h, T=tile_T); r["image"] = name; inc_rows.append(r)
        ds, g = radial_variogram(X, max_r=max_r)
        r_eff[name] = effective_range(ds, g, pct=0.95)
    pairs = {}
    if "a" in images and "b" in images:
        r_ab = min(r_eff["a"], r_eff["b"])
        H_ab, hmax_ab = propose_H(r_ab, cap=2)
        pairs["ab"] = {"H": H_ab, "hmax": hmax_ab, "codisp": codisp_table(images["a"], images["b"], H_ab)}
    if "c" in images and "d" in images:
        r_cd = min(r_eff["c"], r_eff["d"])
        H_cd, hmax_cd = propose_H(r_cd, cap=2)
        pairs["cd"] = {"H": H_cd, "hmax": hmax_cd, "codisp": codisp_table(images["c"], images["d"], H_cd)}
    return {
        "basic": basic,
        "border_center": pd.DataFrame(bvc_rows),
        "tiles": pd.DataFrame(tile_rows),
        "increment_tiles": pd.DataFrame(inc_rows),
        "r_eff": r_eff,
        "pairs": pairs,
    }


def synthetic_diagnostics(models: Dict[str, Tuple[np.ndarray, np.ndarray]], *, max_r: int = 8) -> Dict[str, object]:
    rows = []
    codisp = {}
    for name, (X, Y) in models.items():
        dsX, gX = radial_variogram(X, max_r)
        dsY, gY = radial_variogram(Y, max_r)
        r_eff_X = effective_range(dsX, gX, 0.95)
        r_eff_Y = effective_range(dsY, gY, 0.95)
        H, hmax = propose_H(min(r_eff_X, r_eff_Y), cap=2)
        tab = codisp_table(X, Y, H)
        codisp[name] = tab
        rows.append({
            "model": name,
            "X_std": float(X.std()),
            "Y_std": float(Y.std()),
            "tileVarRatio_X": tile_stats(X)["tile_var_ratio_max_min"],
            "tileVarRatio_Y": tile_stats(Y)["tile_var_ratio_max_min"],
            "r_eff_X(95%)": r_eff_X,
            "r_eff_Y(95%)": r_eff_Y,
            "hmax": hmax,
            "H": H,
            "rho(1,0)": float(tab.loc[tab["h"].astype(str) == str((1,0)), "rho_hat"].iloc[0]) if (tab["h"].astype(str) == str((1,0))).any() else np.nan,
        })
    return {"summary": pd.DataFrame(rows), "codisp": codisp}


def rank1_diagnostics(X_base: np.ndarray, *, taper: np.ndarray | None = None, max_r: int = 8) -> Dict[str, object]:
    cases = {"Rank-1 (Y = X)": (X_base.copy(), X_base.copy()), "Rank-1 (Y = -X)": (X_base.copy(), -X_base.copy())}
    rows = []
    per_case_tables = {}
    for name, (X, Y) in cases.items():
        dsX, gX = radial_variogram(X, max_r)
        dsY, gY = radial_variogram(Y, max_r)
        r_eff = min(effective_range(dsX, gX, 0.95), effective_range(dsY, gY, 0.95))
        H, hmax = propose_H(r_eff, cap=2)
        tab_pre = codisp_table(X, Y, H)
        if taper is not None:
            Xt, Yt = X * taper, Y * taper
            tab_post = codisp_table(Xt, Yt, H)
        else:
            tab_post = tab_pre.copy()
        target = 1.0 if "Y = X" in name else -1.0
        rows.append({
            "case": name,
            "r_eff_pre": r_eff,
            "hmax_pre": hmax,
            "max|rho-±1|_pre": float(np.max(np.abs(tab_pre["rho_hat"].to_numpy() - target))),
            "max|rho-±1|_post": float(np.max(np.abs(tab_post["rho_hat"].to_numpy() - target))),
            "max|Δrho|_taper": float(np.max(np.abs(tab_pre["rho_hat"].to_numpy() - tab_post["rho_hat"].to_numpy()))),
        })
        per_case_tables[name] = {"pre": tab_pre, "post": tab_post}
    return {"summary": pd.DataFrame(rows), "tables": per_case_tables}
