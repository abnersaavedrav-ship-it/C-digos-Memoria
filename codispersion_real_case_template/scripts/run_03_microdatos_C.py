from codispersion_real_case import run_rank1_preprocessing_C

out = run_rank1_preprocessing_C(test=True, n=96)
print(out["diagnostics"]["summary"])
