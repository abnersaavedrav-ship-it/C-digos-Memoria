from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .preprocessing import load_real_images, preprocess_image, crop_pair, tukey2d
from .diagnostics import real_diagnostics, synthetic_diagnostics, rank1_diagnostics, propose_H
from .synthetic import model1_alignment, model2_diff_ranges, model3_anisotropy_corr, model_rank1, unit_tests_rank1, behavior_tests_B
from .codispersion import codispersion_sobre_H
from .block_selection import select_block_size, select_block_size_double_bootstrap
from .bootstrap import boot_codispersion, results_to_table, validate_mosaic


def run_real_preprocessing_A(
    data_dir: str | Path = "data",
    *,
    test: bool = True,
    crop_n: int = 128,
    use_taper: bool = True,
    border_px: int = 8,
) -> Dict[str, object]:
    imgs0 = load_real_images(data_dir, normalize01=True)
    if test:
        imgs0 = {k: v[:crop_n, :crop_n].copy() for k, v in imgs0.items()}
    pre = {k: preprocess_image(v, use_zscore=False, use_taper=use_taper, border_px=border_px) for k, v in imgs0.items()}
    raw_diag = real_diagnostics(imgs0)
    pre_diag = real_diagnostics(pre)
    return {"raw_images": imgs0, "preprocessed_images": pre, "raw_diag": raw_diag, "pre_diag": pre_diag}


def run_synthetic_preprocessing_B(*, test: bool = True, n: int = 96) -> Dict[str, object]:
    if not test:
        n = 256
    models = {
        "M1: alignment + noise": model1_alignment(n=n),
        "M2: different ranges": model2_diff_ranges(n=n),
        "M3: anisotropy": model3_anisotropy_corr(n=n),
    }
    diag = synthetic_diagnostics(models)
    return {"models": models, "diagnostics": diag}


def run_rank1_preprocessing_C(*, test: bool = True, n: int = 96) -> Dict[str, object]:
    if not test:
        n = 256
    X, _, _ = model_rank1(n=n, sigma=3.0, seed=7)
    W = tukey2d(n, n, 8)
    diag = rank1_diagnostics(X, taper=W)
    return {"X_base": X, "diagnostics": diag}


def run_formal_implementation_checks(
    data_dir: str | Path = "data",
    *,
    test: bool = True,
    crop_n: int = 128,
) -> Dict[str, object]:
    all_ok = unit_tests_rank1(verbose=True)
    all_ok &= behavior_tests_B(verbose=True)
    A = run_real_preprocessing_A(data_dir, test=test, crop_n=crop_n, use_taper=True)
    pre = A["preprocessed_images"]
    pairs = A["pre_diag"]["pairs"]
    out = {"tests_ok": bool(all_ok), "real_pairs": pairs}
    # smoke: pair AB and CD with selected H
    if "ab" in pairs:
        H_ab = pairs["ab"]["H"]
        out["ab_codisp_api"] = codispersion_sobre_H(pre["a"], pre["b"], H_ab, mode="toroidal", return_contrib=False)
    if "cd" in pairs:
        H_cd = pairs["cd"]["H"]
        out["cd_codisp_api"] = codispersion_sobre_H(pre["c"], pre["d"], H_cd, mode="toroidal", return_contrib=False)
    return out


def _default_H():
    base = [(1,0), (0,1), (1,1), (-1,1)]
    return [(k*h1, k*h2) for k in (1,2) for (h1,h2) in base]


def run_select_b_real(
    data_dir: str | Path = "data",
    *,
    test: bool = True,
    crop_n: int = 128,
    B_small: int = 30,
    R_diag: int = 4,
    grid: Optional[List[int]] = None,
    double_bootstrap: bool = False,
) -> Dict[str, object]:
    A = run_real_preprocessing_A(data_dir, test=test, crop_n=crop_n, use_taper=True)
    imgs = A["preprocessed_images"]
    H = _default_H()
    if grid is None and test:
        grid = [7, 9, 11, 13]
    if double_bootstrap:
        b_ab, tabla_ab = select_block_size_double_bootstrap(imgs["a"], imgs["b"], H, grid=grid, R_outer=4 if test else 40, B_inner=B_small, seed=777)
        b_cd, tabla_cd = select_block_size_double_bootstrap(imgs["c"], imgs["d"], H, grid=grid, R_outer=4 if test else 40, B_inner=B_small, seed=778)
    else:
        b_ab, tabla_ab = select_block_size(imgs["a"], imgs["b"], H, B_small=B_small, R_diag=R_diag, seed=909, grid=grid)
        b_cd, tabla_cd = select_block_size(imgs["c"], imgs["d"], H, B_small=B_small, R_diag=R_diag, seed=910, grid=grid)
    return {"H": H, "b_ab": b_ab, "b_cd": b_cd, "tabla_ab": tabla_ab, "tabla_cd": tabla_cd, "images": imgs}


def run_bootstrap_real(
    data_dir: str | Path = "data",
    *,
    test: bool = True,
    crop_n: int = 128,
    B: int = 50,
    b: int | None = None,
    scheme: str = "CBB",
    outdir: str | Path = "results",
) -> Dict[str, object]:
    A = run_real_preprocessing_A(data_dir, test=test, crop_n=crop_n, use_taper=True)
    imgs = A["preprocessed_images"]
    H = _default_H()
    if b is None:
        b = 7 if test else 57
    res_ab = boot_codispersion(imgs["a"], imgs["b"], H, B=B, b=b, scheme=scheme, alpha=0.05, seed=20250903)
    res_cd = boot_codispersion(imgs["c"], imgs["d"], H, B=B, b=b, scheme=scheme, alpha=0.05, seed=20250904)
    df_ab = results_to_table(res_ab)
    df_cd = results_to_table(res_cd)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "test" if test else "full"
    df_ab.to_csv(outdir / f"boot_codisp_AB_{scheme}_{suffix}.csv", index=False)
    df_cd.to_csv(outdir / f"boot_codisp_CD_{scheme}_{suffix}.csv", index=False)
    diag_ab = validate_mosaic(imgs["a"], imgs["b"], b=b, sampler=scheme, R=4 if test else 20, H_small=[(1,0), (0,1)], seed=2025)
    diag_cd = validate_mosaic(imgs["c"], imgs["d"], b=b, sampler=scheme, R=4 if test else 20, H_small=[(1,0), (0,1)], seed=2026)
    return {"H": H, "res_ab": res_ab, "res_cd": res_cd, "df_ab": df_ab, "df_cd": df_cd, "diag_ab": diag_ab, "diag_cd": diag_cd, "b": b, "scheme": scheme}
