import numpy as np

from codispersion_bootstrap import (
    ExperimentConfig,
    MaternExperimentConfig,
    Scenario,
    build_H,
    cbb_sample_pair,
    codisp_rho_hat,
    run_min_experiment,
    run_min_experiment_matern,
    run_mc_study_lmc,
    results_to_dataframe,
    select_best_b,
    run_grid_study,
)


def test_codisp_self_near_one():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(16, 16))
    rho, _ = codisp_rho_hat(X, X, (1, 0))
    assert np.isfinite(rho)
    assert 0.99 <= rho <= 1.01


def test_cbb_shape():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(17, 19))
    Y = rng.normal(size=(17, 19))
    Xs, Ys = cbb_sample_pair(X, Y, b=5, rng=rng)
    assert Xs.shape == X.shape
    assert Ys.shape == Y.shape


def test_min_experiment_runs():
    cfg = ExperimentConfig(n1=32, n2=32, B=5, b=8, seed=123)
    out = run_min_experiment(cfg)
    assert "rho_hat" in out and "boot" in out
    assert len(out["rho_hat"]) == len(build_H(cfg.hmax))


def test_matern_min_experiment_runs():
    cfg = MaternExperimentConfig(n1=32, n2=32, B=5, b=8, seed=123)
    out = run_min_experiment_matern(cfg)
    assert "rho_hat" in out and "boot" in out
    assert len(out["rho_hat"]) == len(build_H(cfg.hmax))


def test_mc_bb_cbb_runs_small():
    H = [(1, 0), (0, 1)]
    sc = Scenario(n1=24, n2=24, B_boot=4, b_block=6, R_mc=3, rho0=0.5)
    res = run_mc_study_lmc(sc, H, seed=123)
    df = results_to_dataframe(res, extra={"b": sc.b_block, "rho0": sc.rho0, "range_pix": sc.range_pix, "anis_mode": sc.anis_mode})
    assert not df.empty
    assert "ratio_CBB" in df.columns


def test_grid_study_and_best_b_small():
    H = [(1, 0)]
    df = run_grid_study(
        n1=20,
        n2=20,
        nu=1.0,
        range_pix_list=[1.5],
        rho0_list=[0.2],
        anis_mode_list=["iso"],
        b_list=[4, 5],
        H=H,
        B_boot=3,
        R_mc=2,
        seed_base=123,
    )
    best = select_best_b(df)
    assert not df.empty
    assert not best.empty
