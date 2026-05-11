from codispersion_real_case import run_bootstrap_real

out = run_bootstrap_real(data_dir="data", test=True, crop_n=128, B=50, b=7, scheme="CBB", outdir="results")
print(out["df_ab"])
print(out["df_cd"])
