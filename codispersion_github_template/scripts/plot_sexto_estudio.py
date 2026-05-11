from codispersion_bootstrap import run_grid_study, generate_all_sexto_figures


def main():
    H = [
        (1,0), (0,1), (1,1), (-1,1),
        (2,0), (0,2), (2,2), (-2,2),
    ]

    df_full = run_grid_study(
        n1=256,
        n2=256,
        nu=1.0,
        range_pix_list=[1.5, 3.5, 6.0],
        rho0_list=[0.2, 0.5, 0.8],
        anis_mode_list=["iso", "aniso"],
        b_list=[4, 8, 16, 32, 48, 52, 64],
        H=H,
        B_boot=800,
        R_mc=100,
        anis_ratio=2.0,
        angle_deg=45.0,
        use_taper=True,
        tukey_alpha=0.5,
        seed_base=2025,
    )

    generate_all_sexto_figures(
        df_full,
        outdir="results/graficas_codispersion_full",
        summary_csv_path="results/escenarios_codispersion_resumen_126_reconstruido.csv",
        best_blocks_csv_path="results/sexto_estudio_bloques_optimos.csv",
        show=False,
        curve_degree=2,
    )


if __name__ == "__main__":
    main()