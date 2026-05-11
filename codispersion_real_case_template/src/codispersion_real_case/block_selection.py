from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .bootstrap import _semi_variogram_radial, boot_codispersion, cbb_sample_pair, validate_mosaic
from .codispersion import codispersion_xy


def _estimate_reff(Z: np.ndarray, maxlag: Optional[int] = None) -> int:
    n1, n2 = Z.shape
    if maxlag is None:
        maxlag = int(min(64, max(5, min(n1, n2) // 3)))
    gam = _semi_variogram_radial(Z, maxlag)
    tail = gam[int(0.7 * maxlag):] if maxlag >= 10 else gam[max(1, maxlag - 3):]
    sill = float(np.mean(tail))
    thr = 0.95 * sill
    idx = np.where(gam >= thr)[0]
    r_eff = int(idx[0]) if len(idx) else int(maxlag)
    return max(r_eff, 5)


def _make_odd(x: int | float) -> int:
    x = int(max(3, round(float(x))))
    return x if (x % 2 == 1) else x + 1


def grid_from_reff(r_eff: int, widen: float = 0.2) -> List[int]:
    lo = max(3, int(round((0.5 - widen) * r_eff)))
    hi = int(round((1.0 + widen) * r_eff))
    cand = {_make_odd(b) for b in range(lo, hi + 1)}
    cand |= {_make_odd(t) for t in [0.4*r_eff, 0.5*r_eff, 0.6*r_eff, 0.8*r_eff, 1.0*r_eff, 1.2*r_eff]}
    return sorted({b for b in cand if b >= 3})


def score_b(
    X: np.ndarray,
    Y: np.ndarray,
    H: List[Tuple[int, int]],
    b: int,
    scheme: str = "CBB",
    B_small: int = 120,
    R_diag: int = 12,
    alpha: float = 0.05,
    seed: Optional[int] = 1234,
    w_bias: float = 3.0,
    w_vario: float = 1.0,
) -> Dict[str, float]:
    diag = validate_mosaic(X, Y, b=b, sampler=scheme, R=R_diag, H_small=None, seed=seed)
    vario_err = 0.5 * (float(diag["relL2_gammaX"]) + float(diag["relL2_gammaY"]))
    if not np.isfinite(vario_err):
        vario_err = float("inf")
    est0 = codispersion_xy(X, Y, H, mode="toroidal", min_pairs=1, nan_policy="omit")
    res = boot_codispersion(X, Y, H, B=B_small, b=b, scheme=scheme, alpha=alpha, seed=seed,
                            mode_estimator="toroidal", min_pairs=1, nan_policy="omit")
    diffs = []
    for h in H:
        r0 = est0[h]["rho"] if est0[h]["status"] == "ok" else np.nan
        boots = res[h]["rho_star"]
        msk = np.isfinite(boots)
        if np.isfinite(r0) and np.any(msk):
            diffs.append(abs(np.nanmean(boots[msk]) - float(r0)))
    bias = float(np.nanmean(diffs)) if diffs else float("inf")
    score = w_bias * bias + w_vario * vario_err
    return {"b": int(b), "bias": bias, "vario": vario_err, "score": score}


def select_block_size(
    X: np.ndarray,
    Y: np.ndarray,
    H: List[Tuple[int, int]],
    scheme: str = "CBB",
    B_small: int = 120,
    R_diag: int = 12,
    alpha: float = 0.05,
    seed: Optional[int] = 1234,
    grid: Optional[List[int]] = None,
) -> Tuple[int, pd.DataFrame]:
    if grid is None:
        r_eff = int(round(min(_estimate_reff(X), _estimate_reff(Y))))
        grid = grid_from_reff(r_eff)
    h_max = max(max(abs(h1), abs(h2)) for h1, h2 in H)
    grid = [int(b) for b in grid if int(b) >= max(7, 3*h_max)]
    if not grid:
        base_b = _make_odd(max(7, 3*h_max))
        grid = [base_b, base_b + 4, base_b + 8]
    rows = [score_b(X, Y, H, b=b, scheme=scheme, B_small=B_small, R_diag=R_diag,
                    alpha=alpha, seed=seed, w_bias=3.0, w_vario=1.0) for b in grid]
    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    return int(df.loc[0, "b"]), df


def _double_bootstrap_coverage_score(
    X: np.ndarray,
    Y: np.ndarray,
    H: List[Tuple[int, int]],
    b: int,
    *,
    scheme_outer: str = "CBB",
    scheme_inner: str = "CBB",
    R_outer: int = 40,
    B_inner: int = 200,
    alpha: float = 0.05,
    seed: Optional[int] = 1234,
    penalize_variogram: bool = True,
    w_vario: float = 0.2,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    b_outer = _make_odd(max(b, int(1.1 * b)))
    cover = {h: 0 for h in H}
    widths = {h: [] for h in H}
    se_list = {h: [] for h in H}
    for _ in range(int(R_outer)):
        Xo, Yo = cbb_sample_pair(X, Y, b=b_outer, scheme=scheme_outer, rng=rng)
        est_out = codispersion_xy(Xo, Yo, H, mode="toroidal", min_pairs=1, nan_policy="omit")
        res_in = boot_codispersion(Xo, Yo, H, B=B_inner, b=b, scheme=scheme_inner, alpha=alpha, seed=int(rng.integers(1_000_000)))
        for h in H:
            if est_out[h]["status"] != "ok":
                continue
            rho_true = float(est_out[h]["rho"])
            lo, hi = res_in[h]["ci"]
            if np.isfinite(lo) and np.isfinite(hi) and lo <= rho_true <= hi:
                cover[h] += 1
            if np.isfinite(lo) and np.isfinite(hi):
                widths[h].append(float(hi - lo))
            if np.isfinite(res_in[h]["se_star"]):
                se_list[h].append(float(res_in[h]["se_star"]))
    cov_rates = [cover[h] / max(1, int(R_outer)) for h in H]
    cov_mean = float(np.nanmean(cov_rates))
    abs_err = abs(cov_mean - (1.0 - alpha))
    width = float(np.nanmean([np.nanmean(widths[h]) if widths[h] else np.nan for h in H]))
    se_mean = float(np.nanmean([np.nanmean(se_list[h]) if se_list[h] else np.nan for h in H]))
    vario_err = 0.0
    if penalize_variogram:
        diag = validate_mosaic(X, Y, b=b, sampler=scheme_inner, R=12, H_small=None, seed=seed)
        vario_err = 0.5 * (float(diag["relL2_gammaX"]) + float(diag["relL2_gammaY"]))
        if not np.isfinite(vario_err):
            vario_err = 1e3
    score = abs_err + w_vario * vario_err
    return {"b": int(b), "cov_mean": cov_mean, "abs_err": abs_err, "width": width, "se_mean": se_mean, "vario": vario_err, "score": score}


def select_block_size_double_bootstrap(
    X: np.ndarray,
    Y: np.ndarray,
    H: List[Tuple[int, int]],
    *,
    grid: Optional[List[int]] = None,
    scheme_outer: str = "CBB",
    scheme_inner: str = "CBB",
    R_outer: int = 40,
    B_inner: int = 200,
    alpha: float = 0.05,
    seed: Optional[int] = 1234,
    penalize_variogram: bool = True,
    w_vario: float = 0.2,
) -> Tuple[int, pd.DataFrame]:
    if grid is None:
        r_eff = int(round(min(_estimate_reff(X), _estimate_reff(Y))))
        grid = grid_from_reff(r_eff)
    h_max = max(max(abs(h1), abs(h2)) for h1, h2 in H)
    grid = [int(b) for b in grid if int(b) >= max(7, 3*h_max)]
    if not grid:
        base_b = _make_odd(max(7, 3*h_max))
        grid = [base_b, base_b + 4, base_b + 8]
    rows = []
    for b in grid:
        rows.append(_double_bootstrap_coverage_score(X, Y, H, b, scheme_outer=scheme_outer, scheme_inner=scheme_inner,
                                                     R_outer=R_outer, B_inner=B_inner, alpha=alpha, seed=seed,
                                                     penalize_variogram=penalize_variogram, w_vario=w_vario))
    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    return int(df.loc[0, "b"]), df
