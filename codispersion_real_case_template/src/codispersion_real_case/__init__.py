"""Codispersión y bootstrap espacial para la aplicación real de nanotubos."""

from .preprocessing import (
    ensure_gray,
    load_gray01,
    load_gray,
    zscore,
    tukey_1d,
    tukey2d,
    apply_tukey2d,
    preprocess_image,
    load_real_images,
    crop_pair,
)
from .diagnostics import (
    roll_diff,
    border_center_stats,
    tile_stats,
    tile_var_of_increments,
    variogram_toroidal,
    cross_variogram_toroidal,
    radial_variogram,
    radial_cross_variogram,
    effective_range,
    propose_H,
    codisp_table,
    real_diagnostics,
    synthetic_diagnostics,
    rank1_diagnostics,
)
from .synthetic import (
    gaussian_kernel_2d,
    fftconv2,
    standardize,
    model_rank1,
    model1_alignment,
    model2_diff_ranges,
    model3_anisotropy_corr,
    unit_tests_rank1,
    behavior_tests_B,
)
from .codispersion import (
    codispersion_empirica,
    codispersion_sobre_H,
    codispersion_xy,
)
from .bootstrap import (
    cbb_sample_pair,
    boot_codispersion,
    validate_mosaic,
    results_to_table,
)
from .block_selection import (
    grid_from_reff,
    select_block_size,
    select_block_size_double_bootstrap,
)
from .pipelines import (
    run_real_preprocessing_A,
    run_synthetic_preprocessing_B,
    run_rank1_preprocessing_C,
    run_formal_implementation_checks,
    run_select_b_real,
    run_bootstrap_real,
)

__all__ = [name for name in globals() if not name.startswith('_')]
