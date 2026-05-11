from __future__ import annotations

from typing import Tuple

import numpy as np

from .preprocessing import tukey2d
from .diagnostics import radial_variogram, effective_range, propose_H, variogram_toroidal, cross_variogram_toroidal
from .codispersion import codispersion_empirica


def gaussian_kernel_2d(sx: float, sy: float | None = None, theta_deg: float = 0.0, tol: float = 1e-3) -> np.ndarray:
    if sy is None:
        sy = sx
    theta = np.deg2rad(theta_deg)
    R = int(np.ceil(1.2 * max(3, int(np.ceil(3.5 * sx)), int(np.ceil(3.5 * sy)))))
    xs = np.arange(-R, R + 1, dtype=np.float64)
    ys = np.arange(-R, R + 1, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    c, s = np.cos(theta), np.sin(theta)
    Xp = c * X + s * Y
    Yp = -s * X + c * Y
    K = np.exp(-0.5 * ((Xp / sx) ** 2 + (Yp / sy) ** 2))
    K /= K.max()
    K *= (K >= tol)
    K /= K.sum() + 1e-12
    return K.astype(np.float64)


def fftconv2(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    pad = np.zeros_like(a, dtype=np.float64)
    kh, kw = k.shape
    pad[:kh, :kw] = k
    pad = np.roll(np.roll(pad, -kh // 2, axis=0), -kw // 2, axis=1)
    A = np.fft.rfft2(a)
    K = np.fft.rfft2(pad)
    return np.fft.irfft2(A * K, s=a.shape).astype(np.float64)


def standardize(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - Z.mean()
    s = Z.std()
    return (Z / s if s > 0 else Z).astype(np.float64)


def smooth_field(white: np.ndarray, sx: float, sy: float | None = None, theta: float = 0.0) -> np.ndarray:
    K = gaussian_kernel_2d(sx, sy if sy is not None else sx, theta_deg=theta, tol=1e-3)
    return fftconv2(white, K)


def model_rank1(n: int = 160, sigma: float = 3.0, seed: int = 2025):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    X = standardize(fftconv2(W, gaussian_kernel_2d(sigma, sigma, 0)))
    return X, X.copy(), -X.copy()


def model1_alignment(n: int = 192, c: float = 0.8, sx: float = 3.0, seps: float = 2.0, noise_ratio: float = 0.25, taper_k: int = 8, seed: int = 123):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    E = rng.standard_normal((n, n))
    X0 = standardize(fftconv2(W, gaussian_kernel_2d(sx, sx, 0)))
    eps = standardize(fftconv2(E, gaussian_kernel_2d(seps, seps, 0))) * noise_ratio
    X = X0
    Y = c * X0 + eps
    Wt = tukey2d(n, n, taper_k)
    return X * Wt, Y * Wt


def model2_diff_ranges(n: int = 192, sx_x: float = 1.4, sx_y: float = 7.0, noise_ratio: float = 0.12, taper_k: int = 8, seed: int = 124):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    X = standardize(fftconv2(G, gaussian_kernel_2d(sx_x, sx_x, 0)))
    Y0 = standardize(fftconv2(G, gaussian_kernel_2d(sx_y, sx_y, 0)))
    N = standardize(fftconv2(rng.standard_normal((n, n)), gaussian_kernel_2d(2, 2, 0))) * noise_ratio
    Y = Y0 + N
    Wt = tukey2d(n, n, taper_k)
    return X * Wt, Y * Wt


def model3_anisotropy_corr(n: int = 192, sx_iso: float = 8.0, sx_an: float = 7.0, sy_an: float = 2.0, theta: float = 45.0, noise_ratio: float = 0.08, taper_k: int = 8, seed: int = 125):
    rng = np.random.default_rng(seed)
    Gc = rng.standard_normal((n, n))
    X = standardize(fftconv2(Gc, gaussian_kernel_2d(sx_iso, sx_iso, 0)))
    Y0 = standardize(fftconv2(Gc, gaussian_kernel_2d(sx_an, sy_an, theta)))
    N = standardize(fftconv2(rng.standard_normal((n, n)), gaussian_kernel_2d(2, 2, 0))) * noise_ratio
    Y = Y0 + N
    Wt = tukey2d(n, n, taper_k)
    return X * Wt, Y * Wt


def _assert_close(a, b, tol=1e-12, msg=""):
    if not (abs(a - b) <= tol or (np.isnan(a) and np.isnan(b))):
        raise AssertionError(msg or f"Expected {b}, got {a}")


def unit_tests_rank1(verbose: bool = True) -> bool:
    X, Y_id, Y_neg = model_rank1()
    dsX, gX = radial_variogram(X, 8)
    r_eff = effective_range(dsX, gX, 0.95)
    H, _ = propose_H(r_eff, cap=2)
    for h in H:
        res = codispersion_empirica(X, Y_id, h, return_contrib=True)
        from .diagnostics import roll_diff
        DX = roll_diff(X, *h)
        DY = roll_diff(Y_id, *h)
        C = float((DX * DY).sum())
        A = float((DX * DX).sum())
        B = float((DY * DY).sum())
        rho_classic = C / np.sqrt(A * B)
        _assert_close(res["rho"], rho_classic, 1e-12, "Equivalencia falló")
        _assert_close(codispersion_empirica(X, Y_id, h)["rho"], 1.0, 1e-12, "Rank-1 Y=X")
        _assert_close(codispersion_empirica(X, Y_neg, h)["rho"], -1.0, 1e-12, "Rank-1 Y=-X")
    if verbose:
        print("✓ Unit tests (Rank-1 C): OK")
    return True


def behavior_tests_B(verbose: bool = True) -> bool:
    ok = True
    X1, Y1 = model1_alignment()
    rho10 = codispersion_empirica(X1, Y1, (1, 0))["rho"]
    cond1 = rho10 > 0.70
    ok &= cond1
    if verbose:
        print(f"M1: rho(1,0)={rho10:.3f} ⇒ {'OK' if cond1 else 'FAIL'}")
    X2, Y2 = model2_diff_ranges()
    dsX, gX = radial_variogram(X2, 8)
    dsY, gY = radial_variogram(Y2, 8)
    rX = effective_range(dsX, gX, 0.95)
    rY = effective_range(dsY, gY, 0.95)
    rho10 = codispersion_empirica(X2, Y2, (1, 0))["rho"]
    cond2 = (abs(rX - rY) >= 1.5) and (0.05 <= rho10 <= 0.85)
    ok &= cond2
    if verbose:
        print(f"M2: |Δr_eff|={abs(rX-rY):.2f}, rho(1,0)={rho10:.3f} ⇒ {'OK' if cond2 else 'FAIL'}")
    X3, Y3 = model3_anisotropy_corr()
    ratioY = _dir_anisotropy_ratio_avg(Y3)
    ratioX = _dir_anisotropy_ratio_avg(X3)
    rho10 = codispersion_empirica(X3, Y3, (1, 0))["rho"]
    cond3 = (ratioY > 1.4) and ((ratioY / max(ratioX, 1e-12)) > 1.3) and (rho10 > 0.25)
    ok &= cond3
    if verbose:
        print(f"M3: anisY={ratioY:.2f}, anisX={ratioX:.2f}, rho(1,0)={rho10:.3f} ⇒ {'OK' if cond3 else 'FAIL'}")
        print("Behavior tests (B):" if ok else "Behavior tests (B): FAIL")
    return bool(ok)


def _dir_anisotropy_ratio_avg(Z: np.ndarray, ks=(1, 2, 3)) -> float:
    dirs = [(1, 0), (0, 1), (1, 1), (-1, 1)]
    acc = np.zeros(len(dirs), dtype=float)
    for k in ks:
        vals = [variogram_toroidal(Z, (k * d[0], k * d[1])) for d in dirs]
        acc += np.array(vals, dtype=float)
    acc /= len(ks)
    return float(acc.max() / max(acc.min(), 1e-12))
