from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple
import json
from pathlib import Path

import numpy as np

from .bootstrap import bootstrap_codispersion
from .codispersion import codispersion_by_lag
from .simulation import simulate_pair_basic
from .utils import apply_tukey, build_H, standardize_pair


@dataclass(frozen=True)
class ExperimentConfig:
    n1: int = 256
    n2: int = 256
    rho0: float = 0.8
    range_pix: int = 7
    mode: str = "aniso"
    angle_deg: float = 45.0
    hmax: int = 2
    b: int = 57
    B: int = 800
    seed: int = 2025
    use_tukey: bool = True
    k_px: int = 8
    sampler: str = "CBB"
    mode_rho: str = "toroidal"


def run_min_experiment(config: ExperimentConfig | None = None) -> Dict[str, object]:
    """
    Flujo mínimo del primer estudio:
    simulación -> preprocesamiento -> codispersión -> bootstrap.
    """
    cfg = ExperimentConfig() if config is None else config
    rng = np.random.default_rng(cfg.seed)

    X, Y = simulate_pair_basic(
        cfg.n1,
        cfg.n2,
        rho0=cfg.rho0,
        range_pix=cfg.range_pix,
        mode=cfg.mode,
        angle_deg=cfg.angle_deg,
        rng=rng,
    )

    if cfg.use_tukey:
        X, Y = apply_tukey(X, Y, k_px=cfg.k_px)
    X, Y = standardize_pair(X, Y)

    H = build_H(hmax=cfg.hmax)
    rho_hat = codispersion_by_lag(X, Y, H, mode=cfg.mode_rho)
    boot = bootstrap_codispersion(
        X,
        Y,
        H,
        b=cfg.b,
        B=cfg.B,
        seed=cfg.seed,
        sampler=cfg.sampler,
        mode_rho=cfg.mode_rho,
        return_samples=False,
    )

    return {
        "params": asdict(cfg),
        "H": [tuple(h) for h in H],
        "rho_hat": rho_hat,
        "boot": boot,
    }


def save_results_json(results: Dict[str, object], path: str | Path) -> None:
    """Guarda resultados en JSON serializable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Objeto no serializable: {type(obj)!r}")

    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


@dataclass(frozen=True)
class MaternExperimentConfig:
    """Configuración del segundo estudio: Matérn bivariado vía LMC."""

    n1: int = 256
    n2: int = 256
    rho0: float = 0.75
    range_pix: float = 7.0
    nu: float = 1.0
    anis_mode: str = "aniso"
    angle_deg: float = 45.0
    anis_ratio: float = 2.0
    hmax: int = 2
    b: int = 57
    B: int = 400
    seed: int = 2025
    use_tukey: bool = True
    k_px: int = 8
    sampler: str = "CBB"
    mode_rho: str = "toroidal"


def run_min_experiment_matern(config: MaternExperimentConfig | None = None) -> Dict[str, object]:
    """
    Flujo mínimo del segundo estudio:
    Matérn-LMC -> preprocesamiento -> codispersión -> CBB pareado.
    """
    from .simulation import simulate_pair_matern_LMC

    cfg = MaternExperimentConfig() if config is None else config
    rng = np.random.default_rng(cfg.seed)

    X, Y, sim_params = simulate_pair_matern_LMC(
        cfg.n1,
        cfg.n2,
        rho0=cfg.rho0,
        range_pix=cfg.range_pix,
        nu=cfg.nu,
        anis_mode=cfg.anis_mode,
        angle_deg=cfg.angle_deg,
        anis_ratio=cfg.anis_ratio,
        rng=rng,
    )

    if cfg.use_tukey:
        X, Y = apply_tukey(X, Y, k_px=cfg.k_px)
    X, Y = standardize_pair(X, Y)

    H = build_H(hmax=cfg.hmax)
    rho_hat = codispersion_by_lag(X, Y, H, mode=cfg.mode_rho)
    boot = bootstrap_codispersion(
        X,
        Y,
        H,
        b=cfg.b,
        B=cfg.B,
        seed=cfg.seed,
        sampler=cfg.sampler,
        mode_rho=cfg.mode_rho,
        return_samples=False,
    )

    params = asdict(cfg)
    params.update({f"sim_{k}": v for k, v in sim_params.items()})

    return {
        "params": params,
        "H": [tuple(h) for h in H],
        "rho_hat": rho_hat,
        "boot": boot,
    }
