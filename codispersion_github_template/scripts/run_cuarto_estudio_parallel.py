from codispersion_bootstrap import build_H, run_study_multi_b_parallel_to_csv

# Configuración rápida paralela. Para corrida final, aumentar n, R, B y Bsizes.
out = run_study_multi_b_parallel_to_csv(
    n1=64,
    n2=64,
    H=build_H(hmax=2),
    rhos=(0.2, 0.5),
    ranges=(3.0,),
    anis_modes=("iso", "aniso"),
    Bsizes=(8, 12),
    B=30,
    R=8,
    n_jobs=-1,
    path_detailed_csv="results/cuarto_estudio_detailed_quick.csv",
    path_summary_csv="results/cuarto_estudio_summary_quick.csv",
)
print(out["summary"].head())
print("Guardado en results/cuarto_estudio_*_quick.csv")
