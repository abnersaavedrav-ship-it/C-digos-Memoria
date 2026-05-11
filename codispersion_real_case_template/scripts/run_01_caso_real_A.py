from codispersion_real_case import run_real_preprocessing_A

out = run_real_preprocessing_A(data_dir="data", test=True, crop_n=128)
print(out["pre_diag"]["basic"])
print(out["pre_diag"]["pairs"]["ab"]["codisp"])
