from codispersion_real_case import run_select_b_real

out = run_select_b_real(data_dir="data", test=True, crop_n=128, B_small=30, R_diag=4, grid=[7, 9, 11, 13])
print("b* AB:", out["b_ab"])
print(out["tabla_ab"])
print("b* CD:", out["b_cd"])
print(out["tabla_cd"])
