from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_codispersion
from .codispersion import codisp_rho_hat
from .simulation import MaternParams, lmc_pair_matern, simulate_pair_matern_LMC
from .utils import apply_tukey, build_H, standardize_pair, tukey2d

Lag = Tuple[int, int]


@dataclass(frozen=True)
class Scenario:
    """Escenario para estudios Monte Carlo Matérn-LMC y comparación BB vs CBB."""

    rho0: float = 0.5
    range_pix: float = 3.5
    anis_mode: str = "aniso"  # "iso" o "aniso"
    anis_ratio: float = 2.0
    angle_deg: float = 45.0
    nu: float = 1.0
    n1: int = 256
    n2: int = 256
    B_boot: int = 800
    b_block: int = 32
    R_mc: int = 100


def scenario_grid(
    rhos: Sequence[float],
    ranges: Sequence[float],
    anis_modes: Sequence[str],
) -> List[dict[str, object]]:
    """Construye la grilla cartesiana de escenarios básicos."""
    return [
        {"rho0": float(rho0), "range_pix": float(rg), "anis_mode": str(am)}
        for rho0 in rhos
        for rg in ranges
        for am in anis_modes
    ]


def _preprocess_pair(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    use_taper: bool = True,
    tukey_alpha: float | None = None,
    k_px: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Estandariza y aplica taper común, compatible con los estudios previos."""
    X, Y = standardize_pair(X, Y)
    if use_taper:
        if tukey_alpha is None:
            X, Y = apply_tukey(X, Y, k_px=k_px)
        else:
            W = tukey2d(X.shape[0], X.shape[1], k_px=max(0, int(round(tukey_alpha * min(X.shape) / 2))))
            X, Y = X * W, Y * W
    X, Y = standardize_pair(X, Y)
    return X, Y


def summarize_montecarlo_per_lag(
    rho_hat_rep: np.ndarray,
    var_boot_rep: np.ndarray,
    ci_lo_rep: np.ndarray,
    ci_hi_rep: np.ndarray,
    rho_true: float,
) -> Dict[str, float]:
    """Métricas por lag para el tercer estudio: media, Var_MC, Var*, cobertura y ancho."""
    rho_hat_rep = np.asarray(rho_hat_rep, dtype=float)
    var_boot_rep = np.asarray(var_boot_rep, dtype=float)
    ci_lo_rep = np.asarray(ci_lo_rep, dtype=float)
    ci_hi_rep = np.asarray(ci_hi_rep, dtype=float)
    finite = np.isfinite(rho_hat_rep)
    if finite.sum() < 2:
        return {
            "rho_hat_mean": np.nan,
            "rho_hat_var_mc": np.nan,
            "var_boot_mean": np.nan,
            "coverage": np.nan,
            "width_mean": np.nan,
            "R": int(rho_hat_rep.size),
        }
    coverage = np.mean((rho_true >= ci_lo_rep) & (rho_true <= ci_hi_rep))
    width = np.mean(ci_hi_rep - ci_lo_rep)
    return {
        "rho_hat_mean": float(np.mean(rho_hat_rep[finite])),
        "rho_hat_var_mc": float(np.var(rho_hat_rep[finite], ddof=1)),
        "var_boot_mean": float(np.nanmean(var_boot_rep)),
        "coverage": float(coverage),
        "width_mean": float(width),
        "R": int(rho_hat_rep.size),
    }


def run_monte_carlo_matern_lmc(
    n1: int = 256,
    n2: int = 256,
    H: List[Lag] | None = None,
    rhos: Sequence[float] = (0.2, 0.5, 0.8),
    ranges: Sequence[float] = (3.0, 7.0, 12.0),
    anis_modes: Sequence[str] = ("iso", "aniso"),
    b: int = 57,
    B: int = 400,
    R: int = 100,
    nu: float = 1.0,
    angle_deg: float = 45.0,
    anis_ratio: float = 2.0,
    base_seed: int = 2025,
    sampler: str = "CBB",
    use_taper: bool = True,
    k_px: int = 8,
) -> Dict[str, object]:
    """
    Tercer estudio: loop Monte Carlo Matérn-LMC con un tamaño de bloque.

    Para cada escenario (rho0, rango, anisotropía) repite R veces:
    simula (X,Y), preprocesa, calcula rho_hat(h), ejecuta bootstrap y resume
    cobertura contra la verdad rho_true = rho0 por construcción LMC.
    """
    if H is None:
        H = build_H(hmax=2)
    grid = scenario_grid(rhos, ranges, anis_modes)
    rng_global = np.random.default_rng(base_seed)

    out: Dict[str, object] = {"H": [tuple(h) for h in H], "scenarios": []}

    for sc in grid:
        rho0 = float(sc["rho0"])
        rg = float(sc["range_pix"])
        am = str(sc["anis_mode"])
        per_h_buffers = {
            tuple(h): {
                "rho_hat_rep": np.empty(R, dtype=float),
                "var_boot_rep": np.empty(R, dtype=float),
                "ci_lo_rep": np.empty(R, dtype=float),
                "ci_hi_rep": np.empty(R, dtype=float),
            }
            for h in H
        }

        for r in range(R):
            seed_r = int(rng_global.integers(0, 2**31 - 1))
            rng_r = np.random.default_rng(seed_r)
            X, Y, _ = simulate_pair_matern_LMC(
                n1,
                n2,
                rho0=rho0,
                range_pix=rg,
                nu=nu,
                anis_mode=am,
                angle_deg=angle_deg,
                anis_ratio=anis_ratio,
                rng=rng_r,
            )
            X, Y = _preprocess_pair(X, Y, use_taper=use_taper, k_px=k_px)

            for h in H:
                rho, _ = codisp_rho_hat(X, Y, tuple(h), mode="toroidal")
                per_h_buffers[tuple(h)]["rho_hat_rep"][r] = rho

            boot = bootstrap_codispersion(
                X,
                Y,
                H,
                b=b,
                B=B,
                seed=seed_r,
                sampler=sampler,
                mode_rho="toroidal",
                return_samples=False,
            )
            for h in H:
                hs = str(tuple(h))
                per_h_buffers[tuple(h)]["var_boot_rep"][r] = float(boot[hs]["var_boot"])
                per_h_buffers[tuple(h)]["ci_lo_rep"][r] = float(boot[hs]["ci_lo"])
                per_h_buffers[tuple(h)]["ci_hi_rep"][r] = float(boot[hs]["ci_hi"])

        per_h_summary = {}
        for h in H:
            buf = per_h_buffers[tuple(h)]
            per_h_summary[str(tuple(h))] = summarize_montecarlo_per_lag(
                buf["rho_hat_rep"],
                buf["var_boot_rep"],
                buf["ci_lo_rep"],
                buf["ci_hi_rep"],
                rho_true=rho0,
            )

        out["scenarios"].append(
            {
                "params": {
                    "rho0": rho0,
                    "range_pix": rg,
                    "anis_mode": am,
                    "n": (n1, n2),
                    "nu": nu,
                    "angle_deg": angle_deg,
                    "anis_ratio": anis_ratio,
                    "b": b,
                    "B": B,
                    "R": R,
                    "sampler": sampler,
                },
                "per_h": per_h_summary,
            }
        )
    return out


def run_mc_study_lmc(
    sc: Scenario,
    H: List[Lag],
    *,
    use_taper: bool = True,
    tukey_alpha: float | None = None,
    k_px: int = 8,
    seed: int = 12345,
    samplers: Sequence[str] = ("BB", "CBB"),
) -> Dict[Lag, Dict[str, float]]:
    """
    Quinto/Sexto estudio: compara BB vs CBB en un escenario y un b.

    Se guarda Var* e IC en la misma pasada. La cobertura se calcula al final contra
    la media Monte Carlo plug-in rho_mean_MC por lag, tal como en el sexto estudio.
    """
    rng = np.random.default_rng(seed)
    anis_ratio = sc.anis_ratio if sc.anis_mode == "aniso" else 1.0
    angle_deg = sc.angle_deg if sc.anis_mode == "aniso" else 0.0

    p1 = MaternParams(sc.nu, sc.range_pix, 1.0, anis_ratio, angle_deg)
    p2 = MaternParams(sc.nu, sc.range_pix, 1.0, anis_ratio, angle_deg)

    R = int(sc.R_mc)
    B = int(sc.B_boot)
    b = int(sc.b_block)
    sampler_names = [s.upper() for s in samplers]

    rho_hat = {tuple(h): np.empty(R, dtype=float) for h in H}
    var_star = {s: {tuple(h): np.empty(R, dtype=float) for h in H} for s in sampler_names}
    ci_lo = {s: {tuple(h): np.empty(R, dtype=float) for h in H} for s in sampler_names}
    ci_hi = {s: {tuple(h): np.empty(R, dtype=float) for h in H} for s in sampler_names}

    for r in range(R):
        X, Y = lmc_pair_matern(sc.n1, sc.n2, p1, p2, sc.rho0, rng)
        X, Y = _preprocess_pair(X, Y, use_taper=use_taper, tukey_alpha=tukey_alpha, k_px=k_px)

        for h in H:
            rho_hat[tuple(h)][r] = codisp_rho_hat(X, Y, tuple(h), mode="toroidal")[0]

        for sampler in sampler_names:
            boot = bootstrap_codispersion(
                X,
                Y,
                H,
                b=b,
                B=B,
                seed=int(rng.integers(0, 2**31 - 1)),
                sampler=sampler,
                mode_rho="toroidal",
                return_samples=False,
            )
            for h in H:
                hs = str(tuple(h))
                var_star[sampler][tuple(h)][r] = float(boot[hs]["var_boot"])
                ci_lo[sampler][tuple(h)][r] = float(boot[hs]["ci_lo"])
                ci_hi[sampler][tuple(h)][r] = float(boot[hs]["ci_hi"])

    results: Dict[Lag, Dict[str, float]] = {}
    for h in H:
        h = tuple(h)
        mu = float(np.nanmean(rho_hat[h]))
        var_mc = float(np.nanvar(rho_hat[h], ddof=1))
        row: Dict[str, float] = {"rho_hat_mean": mu, "Var_MC": var_mc}
        for sampler in sampler_names:
            short = sampler
            v = float(np.nanmean(var_star[sampler][h]))
            row[f"Var*_{short}_mean"] = v
            row[f"ratio_{short}"] = float(v / var_mc) if var_mc > 0 else np.nan
            row[f"coverage_{short}"] = float(np.nanmean((ci_lo[sampler][h] <= mu) & (mu <= ci_hi[sampler][h])))
        results[h] = row
    return results


def results_to_dataframe(results: Dict[Lag, Dict[str, float]], *, extra: dict[str, object] | None = None) -> pd.DataFrame:
    """Convierte resultados por lag a DataFrame."""
    rows = []
    for h, values in results.items():
        row = {"h": str(tuple(h)), "h1": int(h[0]), "h2": int(h[1]), **values}
        if extra:
            row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def monte_carlo_to_dataframe(mc_results: Dict[str, object]) -> pd.DataFrame:
    """Convierte la salida de run_monte_carlo_matern_lmc a formato largo."""
    rows = []
    for scenario in mc_results.get("scenarios", []):
        params = dict(scenario["params"])
        for h, metrics in scenario["per_h"].items():
            row = {"h": h, **params, **metrics}
            rows.append(row)
    return pd.DataFrame(rows)


def run_grid_study(
    n1: int,
    n2: int,
    nu: float,
    range_pix_list: Sequence[float],
    rho0_list: Sequence[float],
    anis_mode_list: Sequence[str],
    b_list: Sequence[int],
    H: List[Lag],
    *,
    B_boot: int = 800,
    R_mc: int = 100,
    anis_ratio: float = 2.0,
    angle_deg: float = 45.0,
    use_taper: bool = True,
    tukey_alpha: float | None = None,
    seed_base: int = 2025,
) -> pd.DataFrame:
    """Sexto estudio: grilla de escenarios y tamaños de bloque. Devuelve DataFrame largo."""
    frames: list[pd.DataFrame] = []
    counter = 0
    for range_pix in range_pix_list:
        for rho0 in rho0_list:
            for anis_mode in anis_mode_list:
                for b in b_list:
                    counter += 1
                    sc = Scenario(
                        rho0=float(rho0),
                        range_pix=float(range_pix),
                        anis_mode=str(anis_mode),
                        anis_ratio=float(anis_ratio),
                        angle_deg=float(angle_deg),
                        nu=float(nu),
                        n1=int(n1),
                        n2=int(n2),
                        B_boot=int(B_boot),
                        b_block=int(b),
                        R_mc=int(R_mc),
                    )
                    res = run_mc_study_lmc(
                        sc,
                        H,
                        use_taper=use_taper,
                        tukey_alpha=tukey_alpha,
                        seed=seed_base + counter,
                    )
                    frames.append(
                        results_to_dataframe(
                            res,
                            extra={
                                "range_pix": float(range_pix),
                                "rho0": float(rho0),
                                "anis_mode": str(anis_mode),
                                "b": int(b),
                                "B_boot": int(B_boot),
                                "R_mc": int(R_mc),
                                "nu": float(nu),
                            },
                        )
                    )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_best_b(
    df: pd.DataFrame,
    *,
    ratio_col: str = "ratio_CBB",
    group_cols: Sequence[str] = ("range_pix", "rho0", "anis_mode"),
) -> pd.DataFrame:
    """Selecciona b con ratio promedio más cercano a 1 dentro de cada escenario."""
    if df.empty:
        return pd.DataFrame()
    grp_cols = list(group_cols) + ["b"]
    score = (
        df.groupby(grp_cols, dropna=False)[ratio_col]
        .mean()
        .reset_index(name="ratio_mean")
    )
    score["score_abs_ratio_minus_1"] = (score["ratio_mean"] - 1.0).abs()
    idx = score.groupby(list(group_cols), dropna=False)["score_abs_ratio_minus_1"].idxmin()
    return score.loc[idx].reset_index(drop=True)


def _one_replica_package(
    seed_r: int,
    n1: int,
    n2: int,
    rho0: float,
    range_pix: float,
    anis_mode: str,
    nu: float,
    angle_deg: float,
    anis_ratio: float,
    H: List[Lag],
    b: int,
    B: int,
) -> Dict[str, tuple[float, float, float, float]]:
    rng_r = np.random.default_rng(seed_r)
    X, Y, _ = simulate_pair_matern_LMC(
        n1,
        n2,
        rho0=rho0,
        range_pix=range_pix,
        nu=nu,
        anis_mode=anis_mode,
        angle_deg=angle_deg,
        anis_ratio=anis_ratio,
        rng=rng_r,
    )
    X, Y = _preprocess_pair(X, Y, use_taper=True, k_px=8)
    rho_hat = {tuple(h): codisp_rho_hat(X, Y, tuple(h), mode="toroidal")[0] for h in H}
    boot = bootstrap_codispersion(
        X,
        Y,
        H,
        b=b,
        B=B,
        seed=seed_r,
        sampler="CBB",
        mode_rho="toroidal",
        return_samples=False,
    )
    return {
        str(tuple(h)): (
            float(rho_hat[tuple(h)]),
            float(boot[str(tuple(h))]["var_boot"]),
            float(boot[str(tuple(h))]["ci_lo"]),
            float(boot[str(tuple(h))]["ci_hi"]),
        )
        for h in H
    }


def run_study_multi_b_parallel_to_csv(
    n1: int = 256,
    n2: int = 256,
    H: List[Lag] | None = None,
    rhos: Sequence[float] = (0.2, 0.5, 0.8),
    ranges: Sequence[float] = (3.0, 7.0, 12.0),
    anis_modes: Sequence[str] = ("iso", "aniso"),
    Bsizes: Sequence[int] = (45, 51, 55, 57, 59),
    B: int = 400,
    R: int = 120,
    nu: float = 1.0,
    angle_deg: float = 45.0,
    anis_ratio: float = 2.0,
    base_seed: int = 2025,
    n_jobs: int = -1,
    batch_size: int = 4,
    path_detailed_csv: str | Path = "results/detailed_rows_parallel.csv",
    path_summary_csv: str | Path = "results/summary_scenario_parallel.csv",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Cuarto estudio: versión paralela CBB multi-b, guarda CSV detallado y resumen."""
    if H is None:
        H = build_H(hmax=2)
    try:
        from joblib import Parallel, delayed
    except Exception as exc:  # pragma: no cover
        raise ImportError("Este estudio requiere joblib. Instala con: pip install joblib") from exc

    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(var, "1")

    grid = scenario_grid(rhos, ranges, anis_modes)
    rng_global = np.random.default_rng(base_seed)
    detailed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    total_tasks = len(grid) * len(Bsizes)
    task_id = 0

    for sc_idx, sc in enumerate(grid):
        rho0 = float(sc["rho0"])
        rg = float(sc["range_pix"])
        am = str(sc["anis_mode"])
        for b in Bsizes:
            task_id += 1
            if verbose:
                print(f"[{task_id}/{total_tasks}] rho0={rho0}, range={rg}, anis={am}, b={b}, R={R}, B={B}")
            seeds = rng_global.integers(0, 2**31 - 1, size=R, dtype=np.int64)
            rep_outputs = Parallel(n_jobs=n_jobs, batch_size=batch_size, prefer="threads")(
                delayed(_one_replica_package)(
                    int(seeds[r]), n1, n2, rho0, rg, am, nu, angle_deg, anis_ratio, H, int(b), B
                )
                for r in range(R)
            )

            for r, rep in enumerate(rep_outputs):
                for h in H:
                    hs = str(tuple(h))
                    rh, vb, lo, hi = rep[hs]
                    detailed_rows.append(
                        {
                            "scenario_id": sc_idx,
                            "rep": r,
                            "rho0": rho0,
                            "range_pix": rg,
                            "anis_mode": am,
                            "b": int(b),
                            "h": hs,
                            "h1": int(h[0]),
                            "h2": int(h[1]),
                            "rho_hat": rh,
                            "var_boot": vb,
                            "ci_lo": lo,
                            "ci_hi": hi,
                        }
                    )

            # resumen por lag para este escenario-b
            for h in H:
                hs = str(tuple(h))
                arr = np.array([rep[hs][0] for rep in rep_outputs], dtype=float)
                varb = np.array([rep[hs][1] for rep in rep_outputs], dtype=float)
                lo = np.array([rep[hs][2] for rep in rep_outputs], dtype=float)
                hi = np.array([rep[hs][3] for rep in rep_outputs], dtype=float)
                summ = summarize_montecarlo_per_lag(arr, varb, lo, hi, rho_true=rho0)
                summary_rows.append(
                    {
                        "scenario_id": sc_idx,
                        "rho0": rho0,
                        "range_pix": rg,
                        "anis_mode": am,
                        "b": int(b),
                        "h": hs,
                        "h1": int(h[0]),
                        "h2": int(h[1]),
                        **summ,
                    }
                )

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = pd.DataFrame(summary_rows)
    path_detailed_csv = Path(path_detailed_csv)
    path_summary_csv = Path(path_summary_csv)
    path_detailed_csv.parent.mkdir(parents=True, exist_ok=True)
    path_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    detailed_df.to_csv(path_detailed_csv, index=False)
    summary_df.to_csv(path_summary_csv, index=False)
    return {"detailed": detailed_df, "summary": summary_df}
