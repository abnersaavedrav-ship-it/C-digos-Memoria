from codispersion_real_case import run_formal_implementation_checks

out = run_formal_implementation_checks(data_dir="data", test=True, crop_n=128)
print("tests_ok:", out["tests_ok"])
