"""Herramientas para codispersión y bootstrap espacial en campos 2D."""

from .utils import build_H, tukey_1d, tukey2d, standardize, standardize_pair, apply_tukey
from .codispersion import delta_h_toroidal, codisp_rho_hat, codispersion_by_lag
from .bootstrap import cbb_sample_pair, bb_lahiri_sample_pair, bootstrap_codispersion
from .simulation import (
    MaternParams,
    grf_gaussian_spectral,
    simulate_pair_basic,
    simulate_matern_fft,
    lmc_pair_matern,
    matern_spectral_filter,
    grf_matern_2d,
    kappa_from_range_pix,
    simulate_pair_matern_LMC,
    simulate_pair_matern_lmc,
)
from .experiments import (
    ExperimentConfig,
    MaternExperimentConfig,
    run_min_experiment,
    run_min_experiment_matern,
    save_results_json,
)
from .monte_carlo import (
    Scenario,
    scenario_grid,
    summarize_montecarlo_per_lag,
    run_monte_carlo_matern_lmc,
    run_mc_study_lmc,
    run_grid_study,
    run_study_multi_b_parallel_to_csv,
    results_to_dataframe,
    monte_carlo_to_dataframe,
    select_best_b,
)

__all__ = [
    "build_H",
    "tukey_1d",
    "tukey2d",
    "standardize",
    "standardize_pair",
    "apply_tukey",
    "delta_h_toroidal",
    "codisp_rho_hat",
    "codispersion_by_lag",
    "cbb_sample_pair",
    "bb_lahiri_sample_pair",
    "bootstrap_codispersion",
    "grf_gaussian_spectral",
    "simulate_pair_basic",
    "MaternParams",
    "simulate_matern_fft",
    "lmc_pair_matern",
    "matern_spectral_filter",
    "grf_matern_2d",
    "kappa_from_range_pix",
    "simulate_pair_matern_LMC",
    "simulate_pair_matern_lmc",
    "ExperimentConfig",
    "MaternExperimentConfig",
    "run_min_experiment",
    "run_min_experiment_matern",
    "save_results_json",
    "Scenario",
    "scenario_grid",
    "summarize_montecarlo_per_lag",
    "run_monte_carlo_matern_lmc",
    "run_mc_study_lmc",
    "run_grid_study",
    "run_study_multi_b_parallel_to_csv",
    "results_to_dataframe",
    "monte_carlo_to_dataframe",
    "select_best_b",
]

from .sexto_plots import (
    standardize_sexto_results,
    summarize_sexto_results,
    generate_all_sexto_figures,
    polynomial_fit_summary,
    generate_coverage_ratio_polynomial_fits,
)