from pathlib import Path
from codispersion_bootstrap import build_H, run_monte_carlo_matern_lmc, monte_carlo_to_dataframe

# Configuración rápida. Para corrida final, aumentar n, R y B.
res = run_monte_carlo_matern_lmc(
    n1=64,
    n2=64,
    H=build_H(hmax=2),
    rhos=(0.2, 0.5),
    ranges=(3.0, 7.0),
    anis_modes=("iso", "aniso"),
    b=12,
    B=50,
    R=10,
    base_seed=2025,
)

df = monte_carlo_to_dataframe(res)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/tercer_estudio_mc_quick.csv", index=False)
print(df.head())
print("Guardado en results/tercer_estudio_mc_quick.csv")
