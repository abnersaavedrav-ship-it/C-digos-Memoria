from pathlib import Path
from codispersion_bootstrap import run_grid_study, select_best_b

H = [(1,0), (0,1), (1,1), (-1,1), (2,0), (0,2), (2,2), (-2,2)]

df = run_grid_study(
    n1=64,
    n2=64,
    nu=1.0,
    range_pix_list=[1.5, 3.5],
    rho0_list=[0.2, 0.5],
    anis_mode_list=["iso", "aniso"],
    b_list=[8, 12],
    H=H,
    B_boot=30,
    R_mc=8,
    seed_base=2025,
)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/sexto_estudio_grid_quick.csv", index=False)
print(df.head())
print("\nMejor b por escenario:")
print(select_best_b(df))
print("Guardado en results/sexto_estudio_grid_quick.csv")
