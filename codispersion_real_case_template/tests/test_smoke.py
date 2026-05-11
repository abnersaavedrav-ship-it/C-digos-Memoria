import numpy as np

from codispersion_real_case import (
    model_rank1,
    codispersion_empirica,
    cbb_sample_pair,
    boot_codispersion,
    run_rank1_preprocessing_C,
    run_synthetic_preprocessing_B,
)


def test_rank1_codispersion_pm_one():
    X, Y, Yn = model_rank1(n=32, sigma=2.0, seed=1)
    H = [(1,0), (0,1), (1,1), (-1,1)]
    for h in H:
        assert abs(codispersion_empirica(X, Y, h)["rho"] - 1.0) < 1e-10
        assert abs(codispersion_empirica(X, Yn, h)["rho"] + 1.0) < 1e-10


def test_cbb_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 32))
    Y = rng.normal(size=(32, 32))
    Xs, Ys = cbb_sample_pair(X, Y, b=7, scheme="CBB", rng=rng)
    assert Xs.shape == X.shape
    assert Ys.shape == Y.shape


def test_bootstrap_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 24))
    Y = X + 0.1 * rng.normal(size=(24, 24))
    out = boot_codispersion(X, Y, [(1,0), (0,1)], B=12, b=5, seed=1)
    assert set(out.keys()) == {(1,0), (0,1)}
    assert np.isfinite(out[(1,0)]["rho"])


def test_pipelines_smoke():
    c = run_rank1_preprocessing_C(test=True, n=32)
    assert "summary" in c["diagnostics"]
    b = run_synthetic_preprocessing_B(test=True, n=96)
    assert "summary" in b["diagnostics"]
