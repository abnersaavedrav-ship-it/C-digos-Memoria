from codispersion_real_case import run_synthetic_preprocessing_B

out = run_synthetic_preprocessing_B(test=True, n=96)
print(out["diagnostics"]["summary"])
