from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .utils import standardize


@dataclass(frozen=True)
class MaternParams:
    """Parámetros para simulación Matérn espectral aproximada en 2D."""

    nu: float = 1.0
    ell: float = 3.5
    sill: float = 1.0
    anis_ratio: float = 1.0
    angle_deg: float = 0.0


def _freq_grids(n1: int, n2: int) -> tuple[np.ndarray, np.ndarray]:
    """Mallas de frecuencias FFT en radianes/píxel."""
    k1 = np.fft.fftfreq(n1) * 2.0 * np.pi
    k2 = np.fft.fftfreq(n2) * 2.0 * np.pi
    K2, K1 = np.meshgrid(k2, k1)
    return K1, K2


def _rotate(kx: np.ndarray, ky: np.ndarray, theta_rad: float) -> tuple[np.ndarray, np.ndarray]:
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    return ct * kx + st * ky, -st * kx + ct * ky


def grf_gaussian_spectral(
    n1: int,
    n2: int,
    sig_x: float,
    sig_y: float,
    theta_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Campo Gaussiano suave mediante filtro Gaussiano anisotrópico en Fourier."""
    rng = np.random.default_rng() if rng is None else rng
    K1, K2 = _freq_grids(n1, n2)
    K1p, K2p = _rotate(K1, K2, np.deg2rad(theta_deg))
    F = np.exp(-0.5 * ((sig_x * K1p) ** 2 + (sig_y * K2p) ** 2))
    W = rng.normal(size=(n1, n2)) + 1j * rng.normal(size=(n1, n2))
    return np.fft.ifft2(F * W).real


def simulate_pair_basic(
    n1: int,
    n2: int,
    rho0: float = 0.5,
    range_pix: int = 7,
    mode: str = "iso",
    angle_deg: float = 45.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera un par (X, Y) de campos suaves correlacionados.

    mode:
        "iso"        : ambos isotrópicos.
        "aniso"      : Y incorpora textura anisotrópica 2:1.
        "diff_range" : Y incorpora textura con rango distinto.
    """
    rng = np.random.default_rng() if rng is None else rng
    sig_base = max(1.0, range_pix / 2.0)

    X0 = standardize(grf_gaussian_spectral(n1, n2, sig_base, sig_base, 0.0, rng))
    eps = standardize(grf_gaussian_spectral(n1, n2, sig_base, sig_base, 0.0, rng))

    if mode == "iso":
        Y_base = X0.copy()
    elif mode == "aniso":
        Y_base = standardize(grf_gaussian_spectral(n1, n2, 2.0 * sig_base, sig_base, angle_deg, rng))
    elif mode == "diff_range":
        Y_base = standardize(grf_gaussian_spectral(n1, n2, 1.5 * sig_base, 1.5 * sig_base, 0.0, rng))
    else:
        raise ValueError("mode debe ser 'iso', 'aniso' o 'diff_range'.")

    c = float(np.clip(rho0, -0.999, 0.999))
    eta = float(np.sqrt(max(0.0, 1.0 - c**2)))
    Y0 = c * X0 + eta * eps

    if mode != "iso":
        Y0 = 0.7 * standardize(Y0) + 0.3 * Y_base

    return standardize(X0), standardize(Y0)


def _anisotropic_quadform(K1: np.ndarray, K2: np.ndarray, anis_ratio: float, angle_deg: float) -> np.ndarray:
    a = float(max(anis_ratio, 1.0))
    U, V = _rotate(K1, K2, np.deg2rad(angle_deg))
    return (U / a) ** 2 + (V * a) ** 2


def _matern_spectral_density(
    K1: np.ndarray,
    K2: np.ndarray,
    nu: float,
    ell: float,
    anis_ratio: float,
    angle_deg: float,
) -> np.ndarray:
    if nu <= 0 or ell <= 0:
        raise ValueError("nu y ell deben ser positivos.")
    kappa2 = 2.0 * nu / (ell**2)
    quad = _anisotropic_quadform(K1, K2, anis_ratio, angle_deg)
    S = (kappa2 + quad) ** (-(nu + 1.0))  # d/2 = 1 en 2D
    return np.maximum(S, 0.0)


def simulate_matern_fft(
    n1: int,
    n2: int,
    params: MaternParams,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simula un campo Matérn aproximado por densidad espectral en FFT."""
    rng = np.random.default_rng() if rng is None else rng
    K1, K2 = _freq_grids(n1, n2)
    S = _matern_spectral_density(K1, K2, params.nu, params.ell, params.anis_ratio, params.angle_deg)
    W = rng.normal(size=(n1, n2)) + 1j * rng.normal(size=(n1, n2))
    Z = np.fft.ifft2(np.sqrt(S) * W).real
    return standardize(Z) * np.sqrt(params.sill)


def lmc_pair_matern(
    n1: int,
    n2: int,
    p1: MaternParams,
    p2: MaternParams,
    rho0: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Par bivariado por coregionalización lineal simple:

        X = sqrt(sill_1) Z1,
        Y = sqrt(sill_2) (rho0 Z1 + sqrt(1-rho0^2) Z2).
    """
    rng = np.random.default_rng() if rng is None else rng
    rho0 = float(np.clip(rho0, -0.999, 0.999))

    Z1 = standardize(simulate_matern_fft(n1, n2, MaternParams(p1.nu, p1.ell, 1.0, p1.anis_ratio, p1.angle_deg), rng))
    Z2 = standardize(simulate_matern_fft(n1, n2, MaternParams(p2.nu, p2.ell, 1.0, p2.anis_ratio, p2.angle_deg), rng))

    X = Z1 * np.sqrt(p1.sill)
    Y = (rho0 * Z1 + np.sqrt(max(0.0, 1.0 - rho0**2)) * Z2) * np.sqrt(p2.sill)
    return X, Y


def _elliptical_norm(
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    stretch_x: float = 1.0,
    stretch_y: float = 1.0,
    theta_deg: float = 0.0,
) -> np.ndarray:
    """
    Norma elíptica usada por el segundo estudio Matérn-LMC.

    La rotación se aplica en el plano de frecuencias y luego se ponderan los ejes
    con `stretch_x` y `stretch_y`.
    """
    kxp, kyp = _rotate(kx, ky, np.deg2rad(theta_deg))
    return np.sqrt((stretch_x * kxp) ** 2 + (stretch_y * kyp) ** 2)


def matern_spectral_filter(
    nu: float,
    kappa: float,
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    stretch_x: float = 1.0,
    stretch_y: float = 1.0,
    theta_deg: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Filtro de amplitud Matérn en 2D para síntesis espectral.

    Usa la forma

        sqrt(S(omega)) proporcional a (kappa^2 + ||A_theta omega||^2)^(-(nu + d/2)/2),

    con d=2. Se normaliza por su máximo para evitar depender de la escala absoluta;
    el campo se estandariza después.
    """
    if nu <= 0:
        raise ValueError("nu debe ser positivo.")
    if kappa <= 0:
        raise ValueError("kappa debe ser positivo.")
    d = 2.0
    r_ell = _elliptical_norm(
        kx,
        ky,
        stretch_x=stretch_x,
        stretch_y=stretch_y,
        theta_deg=theta_deg,
    )
    alpha = nu + d / 2.0
    amp = (kappa**2 + r_ell**2 + eps) ** (-alpha / 2.0)
    max_amp = float(np.max(amp))
    if max_amp > 0:
        amp = amp / max_amp
    return amp


def grf_matern_2d(
    n1: int,
    n2: int,
    nu: float,
    kappa: float,
    *,
    stretch_x: float = 1.0,
    stretch_y: float = 1.0,
    theta_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Genera un GRF Matérn 2D aproximado mediante FFT e IFFT."""
    rng = np.random.default_rng() if rng is None else rng
    K1, K2 = _freq_grids(n1, n2)
    amp = matern_spectral_filter(
        nu,
        kappa,
        K1,
        K2,
        stretch_x=stretch_x,
        stretch_y=stretch_y,
        theta_deg=theta_deg,
    )
    W = rng.normal(size=(n1, n2)) + 1j * rng.normal(size=(n1, n2))
    Z = np.fft.ifft2(amp * W).real
    return Z


def kappa_from_range_pix(range_pix: float, nu: float) -> float:
    """
    Convierte rango efectivo aproximado en píxeles al parámetro kappa del Matérn.

    Usa la regla práctica del segundo estudio:

        r_eff ≈ sqrt(8 nu) / kappa,
        kappa ≈ sqrt(8 nu) / r_eff.
    """
    if nu <= 0:
        raise ValueError("nu debe ser positivo.")
    range_pix = max(1.0, float(range_pix))
    return float(np.sqrt(8.0 * nu) / range_pix)


def simulate_pair_matern_LMC(
    n1: int,
    n2: int,
    rho0: float = 0.6,
    range_pix: float = 7.0,
    nu: float = 1.0,
    anis_mode: str = "iso",
    angle_deg: float = 45.0,
    anis_ratio: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """
    Simula un par bivariado Matérn mediante LMC simple.

    Modelo latente:

        G, H independientes ~ Matérn(nu, kappa, anisotropía),
        X = G,
        Y = rho0 * G + sqrt(1-rho0^2) * H.

    Si G y H comparten estructura espacial, la correlación cruzada objetivo queda
    controlada por `rho0` y la codispersión esperada de incrementos queda cerca de
    `rho0` para lags cortos.
    """
    rng = np.random.default_rng() if rng is None else rng
    kappa = kappa_from_range_pix(range_pix, nu)

    if anis_mode == "iso":
        sx, sy, th = 1.0, 1.0, 0.0
    elif anis_mode == "aniso":
        sx, sy, th = float(anis_ratio), 1.0, float(angle_deg)
    else:
        raise ValueError("anis_mode debe ser 'iso' o 'aniso'.")

    G = standardize(
        grf_matern_2d(
            n1,
            n2,
            nu,
            kappa,
            stretch_x=sx,
            stretch_y=sy,
            theta_deg=th,
            rng=rng,
        )
    )
    H = standardize(
        grf_matern_2d(
            n1,
            n2,
            nu,
            kappa,
            stretch_x=sx,
            stretch_y=sy,
            theta_deg=th,
            rng=rng,
        )
    )

    c = float(np.clip(rho0, -0.999, 0.999))
    eta = float(np.sqrt(max(0.0, 1.0 - c**2)))
    X = standardize(G)
    Y = standardize(c * G + eta * H)

    params: dict[str, object] = {
        "rho0": float(rho0),
        "nu": float(nu),
        "kappa": float(kappa),
        "range_pix": float(range_pix),
        "anis_mode": anis_mode,
        "anis_ratio": float(anis_ratio),
        "angle_deg": float(angle_deg),
        "stretch_x": float(sx),
        "stretch_y": float(sy),
    }
    return X, Y, params


# Alias en estilo snake_case para código nuevo.
simulate_pair_matern_lmc = simulate_pair_matern_LMC
