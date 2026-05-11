from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convierte una imagen RGB/RGBA a escala de grises; si ya es 2D, la conserva."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        else:
            arr = arr.mean(axis=2)
    return arr.astype(np.float64)


def load_gray(path: str | Path) -> np.ndarray:
    """Carga una imagen como float64 en escala de grises, manteniendo escala original."""
    path = Path(path)
    try:
        import imageio.v3 as iio
        arr = iio.imread(path)
    except Exception:
        from PIL import Image
        arr = np.asarray(Image.open(path))
    return ensure_gray(arr)


def load_gray01(path: str | Path) -> np.ndarray:
    """Carga una imagen como float64 en escala de grises normalizada a [0, 1]."""
    arr = load_gray(path)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr.astype(np.float64)


def zscore(img: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Estandariza una imagen: media 0 y desviación estándar 1."""
    img = np.asarray(img, dtype=np.float64)
    return (img - np.nanmean(img)) / (np.nanstd(img) + eps)


def tukey_1d(N: int, k: int) -> np.ndarray:
    """Ventana Tukey 1D con aproximadamente k pixeles de rampa por borde."""
    N = int(N)
    k = int(k)
    if N <= 1:
        return np.ones(N, dtype=np.float64)
    alpha = min(1.0, max(0.0, 2.0 * k / max(1, N - 1)))
    n = np.arange(N, dtype=np.float64)
    if alpha <= 0.0:
        return np.ones(N, dtype=np.float64)
    if alpha >= 1.0:
        return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    w = np.ones(N, dtype=np.float64)
    edge = int(np.floor(alpha * (N - 1) / 2.0))
    if edge > 0:
        idx = np.arange(edge, dtype=np.float64)
        vals = 0.5 * (1 + np.cos(np.pi * (2 * idx / (alpha * (N - 1)) - 1)))
        w[:edge] = vals
        w[-edge:] = vals[::-1]
    return w


def tukey2d(n1: int, n2: int, k: int = 8) -> np.ndarray:
    """Ventana Tukey 2D separable."""
    return np.outer(tukey_1d(n1, k), tukey_1d(n2, k)).astype(np.float64)


def apply_tukey2d(Z: np.ndarray, border_px: int = 8) -> np.ndarray:
    """Aplica Tukey 2D a una imagen."""
    Z = np.asarray(Z, dtype=np.float64)
    W = tukey2d(Z.shape[0], Z.shape[1], border_px)
    return (Z * W).astype(np.float64)


def preprocess_image(
    Z: np.ndarray,
    *,
    use_zscore: bool = True,
    use_taper: bool = True,
    border_px: int = 8,
) -> np.ndarray:
    """Pipeline estándar: z-score opcional + Tukey opcional."""
    out = np.asarray(Z, dtype=np.float64)
    if use_zscore:
        out = zscore(out)
    if use_taper:
        out = apply_tukey2d(out, border_px=border_px)
    return out.astype(np.float64)


def load_real_images(data_dir: str | Path = "data", normalize01: bool = True) -> Dict[str, np.ndarray]:
    """Carga las imágenes reales a,b,c,d desde una carpeta."""
    data_dir = Path(data_dir)
    loader = load_gray01 if normalize01 else load_gray
    paths = {
        "a": data_dir / "ch05_nano-a.png",
        "b": data_dir / "ch05_nano-b.png",
        "c": data_dir / "ch05_nano-c.png",
        "d": data_dir / "ch05_nano-d.png",
    }
    return {key: loader(path) for key, path in paths.items()}


def crop_pair(X: np.ndarray, Y: np.ndarray, n: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Recorta un par al cuadrado superior izquierdo n x n para pruebas rápidas."""
    if n is None:
        return X, Y
    return X[:n, :n].copy(), Y[:n, :n].copy()
