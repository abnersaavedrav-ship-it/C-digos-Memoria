from __future__ import annotations

from codispersion_bootstrap import ExperimentConfig, run_min_experiment
from codispersion_bootstrap.experiments import save_results_json


def main() -> None:
    # Usa B=200 para corrida rápida. Para resultados finales, subir a B=800 o más.
    cfg = ExperimentConfig(
        n1=256,
        n2=256,
        rho0=0.8,
        range_pix=7,
        mode="aniso",
        angle_deg=45.0,
        hmax=2,
        b=57,
        B=200,
        seed=2025,
    )
    result = run_min_experiment(cfg)

    print("Parámetros:", result["params"])
    print("Lags H:", result["H"])
    print("\nrho_hat(h) por lag:")
    for h, rho in result["rho_hat"].items():
        print(f"  h={h:>8s}  rho_hat={rho: .6f}")

    print("\nBootstrap Var* e IC 95%:")
    for h, info in result["boot"].items():
        print(
            f"  h={h:>8s}  var*={info['var_boot']:.6e}  "
            f"IC=({info['ci_lo']:.6f}, {info['ci_hi']:.6f})"
        )

    save_results_json(result, "results/primer_estudio_resultados.json")


if __name__ == "__main__":
    main()
