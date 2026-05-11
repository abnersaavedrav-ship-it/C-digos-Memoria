from pathlib import Path
from codispersion_bootstrap import Scenario, run_mc_study_lmc, results_to_dataframe

H = [(1,0), (0,1), (1,1), (-1,1), (2,0), (0,2), (2,2), (-2,2)]

sc = Scenario(
    rho0=0.5,
    range_pix=3.5,
    anis_mode="aniso",
    anis_ratio=2.0,
    angle_deg=45.0,
    nu=1.0,
    n1=64,
    n2=64,
    B_boot=50,
    b_block=12,
    R_mc=10,
)
res = run_mc_study_lmc(sc, H, seed=2025)
df = results_to_dataframe(res, extra={"b": sc.b_block, "rho0": sc.rho0, "range_pix": sc.range_pix, "anis_mode": sc.anis_mode})
Path("results").mkdir(exist_ok=True)
df.to_csv("results/quinto_estudio_bb_vs_cbb_quick.csv", index=False)
print(df)
print("Guardado en results/quinto_estudio_bb_vs_cbb_quick.csv")
