from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def standardize_sexto_results(obj: Any) -> pd.DataFrame:
    """
    Acepta:
    1) salida tipo DataFrame del módulo, o
    2) salida tipo dict del run_grid_study original.

    Devuelve un DataFrame largo con columnas:
    Rango, Ell, Rho, Anisotropía, Bloque, h, Método,
    Cobertura, Ratio Varianzas, Var_MC, Var_boot, rho_hat_mean.
    """

    if isinstance(obj, dict):
        rows = []

        for key, res_by_h in obj.items():
            range_pix, rho0, anis_mode, b = key

            for h, vals in res_by_h.items():
                for metodo in ["BB", "CBB"]:
                    rows.append(
                        {
                            "Ell": float(range_pix),
                            "Rango": int(round(2 * float(range_pix))),
                            "Rho": float(rho0),
                            "Anisotropía": str(anis_mode),
                            "Bloque": int(b),
                            "h": str(h),
                            "Método": metodo,
                            "Cobertura": vals[f"coverage_{metodo}"],
                            "Ratio Varianzas": vals[f"ratio_{metodo}"],
                            "Var_MC": vals["Var_MC"],
                            "Var_boot": vals[f"Var*_{metodo}_mean"],
                            "rho_hat_mean": vals["rho_hat_mean"],
                        }
                    )

        return pd.DataFrame(rows)

    df0 = obj.copy()

    rename_map = {
        "range_pix": "Ell",
        "rho0": "Rho",
        "anis_mode": "Anisotropía",
        "b": "Bloque",
        "b_block": "Bloque",
        "method": "Método",
        "sampler": "Método",
        "coverage": "Cobertura",
        "ratio": "Ratio Varianzas",
        "ratio_var": "Ratio Varianzas",
        "ratio_variances": "Ratio Varianzas",
        "var_mc": "Var_MC",
        "rho_mean": "rho_hat_mean",
    }

    df0 = df0.rename(columns={k: v for k, v in rename_map.items() if k in df0.columns})

    required_long = {
        "Bloque",
        "Rho",
        "Anisotropía",
        "Método",
        "Cobertura",
        "Ratio Varianzas",
    }

    if required_long.issubset(df0.columns):
        if "Ell" not in df0.columns:
            if "Rango" in df0.columns:
                df0["Ell"] = df0["Rango"] / 2
            else:
                df0["Ell"] = np.nan

        if "Rango" not in df0.columns:
            df0["Rango"] = np.round(2 * df0["Ell"]).astype(int)

        return df0

    rows = []

    if "Ell" not in df0.columns:
        if "Rango" in df0.columns:
            df0["Ell"] = df0["Rango"] / 2
        else:
            raise ValueError("No encuentro columna range_pix/Ell/Rango.")

    if "Rango" not in df0.columns:
        df0["Rango"] = np.round(2 * df0["Ell"]).astype(int)

    for _, row in df0.iterrows():
        for metodo in ["BB", "CBB"]:
            cov_col = f"coverage_{metodo}"
            ratio_col = f"ratio_{metodo}"
            var_col = f"Var*_{metodo}_mean"

            if cov_col not in df0.columns:
                cov_col = f"coverage_{metodo.lower()}"

            if ratio_col not in df0.columns:
                ratio_col = f"ratio_{metodo.lower()}"

            rows.append(
                {
                    "Ell": float(row["Ell"]),
                    "Rango": int(row["Rango"]),
                    "Rho": float(row["Rho"]),
                    "Anisotropía": str(row["Anisotropía"]),
                    "Bloque": int(row["Bloque"]),
                    "h": str(row["h"]) if "h" in df0.columns else "",
                    "Método": metodo,
                    "Cobertura": row[cov_col],
                    "Ratio Varianzas": row[ratio_col],
                    "Var_MC": row["Var_MC"] if "Var_MC" in df0.columns else np.nan,
                    "Var_boot": row[var_col] if var_col in df0.columns else np.nan,
                    "rho_hat_mean": row["rho_hat_mean"]
                    if "rho_hat_mean" in df0.columns
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def summarize_sexto_results(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Promedia por escenario base, bloque y método.
    Reproduce el resumen tipo escenarios_codispersion_resumen_126.csv.
    """

    df = (
        df_long.groupby(
            ["Rho", "Rango", "Ell", "Anisotropía", "Bloque", "Método"],
            as_index=False,
        )
        .agg(
            {
                "Cobertura": "mean",
                "Ratio Varianzas": "mean",
                "Var_MC": "mean",
                "Var_boot": "mean",
                "rho_hat_mean": "mean",
            }
        )
    )

    df["Abs Error Cobertura"] = (df["Cobertura"] - 0.95).abs()
    df["Abs Error Ratio"] = (df["Ratio Varianzas"] - 1.0).abs()

    return df


def _savefig(outdir: Path, name: str, show: bool = False) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return path


def _heat_matrix(
    dfm: pd.DataFrame,
    metric: str,
    orden_rangos: list[int],
    orden_bloques: list[int],
) -> np.ndarray:
    M = []

    for r in orden_rangos:
        fila = []

        for b in orden_bloques:
            dd = dfm[(dfm["Rango"] == r) & (dfm["Bloque"] == b)][metric]
            fila.append(dd.mean() if len(dd) > 0 else np.nan)

        M.append(fila)

    return np.array(M)


def _polyfit_curve(x, y, deg: int = 2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(np.unique(x)) <= deg:
        deg = max(1, len(np.unique(x)) - 1)

    if deg < 1 or len(x) < 2:
        return None

    coefs = np.polyfit(x, y, deg)
    yhat = np.polyval(coefs, x)

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    xgrid = np.linspace(x.min(), x.max(), 200)
    ygrid = np.polyval(coefs, xgrid)

    return xgrid, ygrid, coefs, r2


def _plot_curve_fits_by_group(
    df: pd.DataFrame,
    outdir: Path,
    metric: str,
    group_col: str,
    ref_value: float | None,
    deg: int,
    title: str,
    filename: str,
    show: bool,
) -> Path:
    plt.figure(figsize=(11, 6))

    for group_value in sorted(df[group_col].dropna().unique()):
        for metodo in ["BB", "CBB"]:
            sub = df[(df[group_col] == group_value) & (df["Método"] == metodo)]

            if len(sub) == 0:
                continue

            g = sub.groupby("Bloque")[metric]
            x = g.mean().index.values.astype(float)
            y = g.mean().values.astype(float)

            plt.scatter(x, y, s=45, alpha=0.75)

            fit = _polyfit_curve(x, y, deg=deg)

            if fit is None:
                continue

            xgrid, ygrid, _, r2 = fit
            label = f"{metodo} | {group_col}={group_value} | R²={r2:.3f}"
            plt.plot(xgrid, ygrid, linewidth=2, label=label)

    if ref_value is not None:
        plt.axhline(ref_value, linestyle="--", linewidth=1)

    plt.xlabel("Tamaño de bloque (b)")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)

    return _savefig(outdir, filename, show=show)


def generate_all_sexto_figures(
    df_full_or_dict: Any,
    outdir: str | Path = "../results/graficas_codispersion_full",
    summary_csv_path: str | Path = "../results/escenarios_codispersion_resumen_126_reconstruido.csv",
    best_blocks_csv_path: str | Path = "../results/sexto_estudio_bloques_optimos.csv",
    show: bool = False,
    curve_degree: int = 2,
) -> dict[str, Any]:
    """
    Genera todas las gráficas originales del sexto estudio y los ajustes de curvas.

    Retorna:
    {
        "df_long": DataFrame largo,
        "df_summary": DataFrame resumen,
        "best_blocks": DataFrame de bloques óptimos,
        "figure_paths": lista de rutas PNG,
        "outdir": carpeta de salida
    }
    """

    outdir = Path(outdir)
    summary_csv_path = Path(summary_csv_path)
    best_blocks_csv_path = Path(best_blocks_csv_path)

    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    best_blocks_csv_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set(style="whitegrid")

    df_long = standardize_sexto_results(df_full_or_dict)
    df = summarize_sexto_results(df_long)
    df.to_csv(summary_csv_path, index=False)

    orden_bloques = sorted(df["Bloque"].unique())
    orden_rangos = sorted(df["Rango"].unique())
    orden_rho = sorted(df["Rho"].unique())
    orden_anis = sorted(df["Anisotropía"].unique())

    figure_paths: list[Path] = []

    # 1) Relplot cobertura original
    g1 = sns.relplot(
        data=df,
        x="Bloque",
        y="Cobertura",
        hue="Método",
        col="Rho",
        row="Rango",
        kind="line",
        marker="o",
        facet_kws={"sharey": True, "sharex": True},
        height=3.2,
        aspect=1.15,
    )
    g1.fig.subplots_adjust(top=0.9)
    g1.fig.suptitle("Cobertura del IC al 95% según Método, Rho y Rango")
    path = outdir / "00_original_relplot_cobertura_rho_rango.png"
    g1.fig.savefig(path, dpi=300, bbox_inches="tight")
    figure_paths.append(path)
    if show:
        plt.show()
    else:
        plt.close(g1.fig)

    # 2) Relplot ratio original
    g2 = sns.relplot(
        data=df,
        x="Bloque",
        y="Ratio Varianzas",
        hue="Método",
        col="Rho",
        row="Rango",
        kind="line",
        marker="o",
        facet_kws={"sharey": True, "sharex": True},
        height=3.2,
        aspect=1.15,
    )
    g2.fig.subplots_adjust(top=0.9)
    g2.fig.suptitle("Ratio de Varianza Bootstrap / Monte Carlo")
    path = outdir / "00_original_relplot_ratio_rho_rango.png"
    g2.fig.savefig(path, dpi=300, bbox_inches="tight")
    figure_paths.append(path)
    if show:
        plt.show()
    else:
        plt.close(g2.fig)

    # 3) Diferencia cobertura CBB-BB
    df_pivot = df.pivot_table(
        index=["Rho", "Rango", "Ell", "Anisotropía", "Bloque"],
        columns="Método",
        values="Cobertura",
    ).reset_index()

    df_pivot["Dif Cobertura (CBB - BB)"] = df_pivot["CBB"] - df_pivot["BB"]

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_pivot,
        x="Rango",
        y="Dif Cobertura (CBB - BB)",
        hue="Anisotropía",
    )
    plt.title("Diferencia de Cobertura entre CBB y BB por Rango y Anisotropía")
    plt.axhline(0, linestyle="--", color="gray")
    figure_paths.append(_savefig(outdir, "00_original_boxplot_dif_cobertura_cbb_bb", show=show))

    # 4) Cobertura vs Bloque por método y rango
    plt.figure(figsize=(10, 6))
    for r in orden_rangos:
        for metodo in ["BB", "CBB"]:
            d = df[(df["Rango"] == r) & (df["Método"] == metodo)]
            g = d.groupby("Bloque")["Cobertura"]
            x = g.mean().index.values
            y = g.mean().values
            se = g.std(ddof=1).values / np.sqrt(g.count().values)

            plt.plot(x, y, marker="o", label=f"{metodo} | rango={r}")
            plt.fill_between(x, y - se, y + se, alpha=0.15)

    plt.axhline(0.95, linestyle="--", linewidth=1)
    plt.xlabel("Tamaño de bloque (b)")
    plt.ylabel("Cobertura IC 95%")
    plt.title("Cobertura vs Bloque por Método y Rango")
    plt.legend(ncol=2)
    figure_paths.append(_savefig(outdir, "01_cobertura_vs_bloque_metodo_rango", show=show))

    # 5) Cobertura vs Bloque por anisotropía
    plt.figure(figsize=(10, 6))
    for an in orden_anis:
        for metodo in ["BB", "CBB"]:
            d = df[(df["Anisotropía"] == an) & (df["Método"] == metodo)]
            g = d.groupby("Bloque")["Cobertura"]
            x = g.mean().index.values
            y = g.mean().values
            se = g.std(ddof=1).values / np.sqrt(g.count().values)

            plt.plot(x, y, marker="o", label=f"{metodo} | {an}")
            plt.fill_between(x, y - se, y + se, alpha=0.15)

    plt.axhline(0.95, linestyle="--", linewidth=1)
    plt.xlabel("Tamaño de bloque (b)")
    plt.ylabel("Cobertura IC 95%")
    plt.title("Cobertura vs Bloque por Anisotropía y Método")
    plt.legend(ncol=2)
    figure_paths.append(_savefig(outdir, "02_cobertura_vs_bloque_anisotropia", show=show))

    # 6) Ratio vs Bloque por método y rango
    plt.figure(figsize=(10, 6))
    for r in orden_rangos:
        for metodo in ["BB", "CBB"]:
            d = df[(df["Rango"] == r) & (df["Método"] == metodo)]
            g = d.groupby("Bloque")["Ratio Varianzas"]
            x = g.mean().index.values
            y = g.mean().values
            se = g.std(ddof=1).values / np.sqrt(g.count().values)

            plt.plot(x, y, marker="o", label=f"{metodo} | rango={r}")
            plt.fill_between(x, y - se, y + se, alpha=0.15)

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Tamaño de bloque (b)")
    plt.ylabel("Ratio Varianzas (Var*/Var_MC)")
    plt.title("Ratio de Varianzas vs Bloque por Método y Rango")
    plt.legend(ncol=2)
    figure_paths.append(_savefig(outdir, "03_ratio_var_vs_bloque_metodo_rango", show=show))

    # 7) Heatmaps cobertura
    for metodo in ["BB", "CBB"]:
        plt.figure(figsize=(10, 5))
        M = _heat_matrix(df[df["Método"] == metodo], "Cobertura", orden_rangos, orden_bloques)
        im = plt.imshow(M, aspect="auto", interpolation="nearest")
        plt.colorbar(im, label="Cobertura IC 95%")
        plt.xticks(range(len(orden_bloques)), orden_bloques)
        plt.yticks(range(len(orden_rangos)), orden_rangos)
        plt.xlabel("Bloque")
        plt.ylabel("Rango")
        plt.title(f"Mapa de calor de Cobertura — {metodo}")
        figure_paths.append(_savefig(outdir, f"04_heatmap_cobertura_{metodo}", show=show))

    # 8) Heatmaps ratio
    for metodo in ["BB", "CBB"]:
        plt.figure(figsize=(10, 5))
        M = _heat_matrix(df[df["Método"] == metodo], "Ratio Varianzas", orden_rangos, orden_bloques)
        im = plt.imshow(M, aspect="auto", interpolation="nearest")
        plt.colorbar(im, label="Ratio Varianzas")
        plt.xticks(range(len(orden_bloques)), orden_bloques)
        plt.yticks(range(len(orden_rangos)), orden_rangos)
        plt.xlabel("Bloque")
        plt.ylabel("Rango")
        plt.title(f"Mapa de calor de Ratio Varianzas — {metodo}")
        figure_paths.append(_savefig(outdir, f"05_heatmap_ratio_var_{metodo}", show=show))

    # 9) Cobertura vs Rho
    plt.figure(figsize=(10, 6))
    for r in orden_rangos:
        for metodo in ["BB", "CBB"]:
            d = df[(df["Rango"] == r) & (df["Método"] == metodo)]
            g = d.groupby("Rho")["Cobertura"]
            x = g.mean().index.values
            y = g.mean().values
            se = g.std(ddof=1).values / np.sqrt(g.count().values)

            plt.plot(x, y, marker="o", label=f"{metodo} | rango={r}")
            plt.fill_between(x, y - se, y + se, alpha=0.15)

    plt.axhline(0.95, linestyle="--", linewidth=1)
    plt.xlabel(r"Correlación verdadera $\rho_0$")
    plt.ylabel("Cobertura IC 95%")
    plt.title("Cobertura vs Rho por Método y Rango")
    plt.legend(ncol=2)
    figure_paths.append(_savefig(outdir, "06_cobertura_vs_rho_metodo_rango", show=show))

    # 10) Ratio vs Rho
    plt.figure(figsize=(10, 6))
    for r in orden_rangos:
        for metodo in ["BB", "CBB"]:
            d = df[(df["Rango"] == r) & (df["Método"] == metodo)]
            g = d.groupby("Rho")["Ratio Varianzas"]
            x = g.mean().index.values
            y = g.mean().values
            se = g.std(ddof=1).values / np.sqrt(g.count().values)

            plt.plot(x, y, marker="o", label=f"{metodo} | rango={r}")
            plt.fill_between(x, y - se, y + se, alpha=0.15)

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel(r"Correlación verdadera $\rho_0$")
    plt.ylabel("Ratio Varianzas")
    plt.title("Ratio Varianzas vs Rho por Método y Rango")
    plt.legend(ncol=2)
    figure_paths.append(_savefig(outdir, "07_ratio_var_vs_rho_metodo_rango", show=show))

    # 11) Scatter cobertura vs ratio
    plt.figure(figsize=(8, 6))
    for metodo in ["BB", "CBB"]:
        d = df[df["Método"] == metodo]
        plt.scatter(
            d["Ratio Varianzas"],
            d["Cobertura"],
            alpha=0.65,
            label=metodo,
        )

    plt.axhline(0.95, linestyle="--", linewidth=1)
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Ratio Varianzas (Var*/Var_MC)")
    plt.ylabel("Cobertura IC 95%")
    plt.title("Cobertura vs Exactitud de Varianza")
    plt.legend()
    figure_paths.append(_savefig(outdir, "08_scatter_cobertura_vs_ratio", show=show))

    # 12) Bloque óptimo
    def bloque_optimo(gr):
        k = (gr["Cobertura"] - 0.95).abs()
        idx = k.idxmin()
        return gr.loc[idx, "Bloque"]

    optimos = []

    for metodo in ["BB", "CBB"]:
        for rho in orden_rho:
            for r in orden_rangos:
                for an in orden_anis:
                    sub = df[
                        (df["Método"] == metodo)
                        & (df["Rho"] == rho)
                        & (df["Rango"] == r)
                        & (df["Anisotropía"] == an)
                    ]

                    if len(sub) > 0:
                        optimos.append(
                            {
                                "Método": metodo,
                                "Rho": rho,
                                "Rango": r,
                                "Anisotropía": an,
                                "Bloque*": bloque_optimo(sub),
                            }
                        )

    opt_df = pd.DataFrame(optimos)
    opt_df.to_csv(best_blocks_csv_path, index=False)

    plt.figure(figsize=(10, 6))
    bins = np.arange(min(orden_bloques) - 1, max(orden_bloques) + 2, 4)

    for metodo in ["BB", "CBB"]:
        vals = opt_df[opt_df["Método"] == metodo]["Bloque*"].values
        plt.hist(vals, bins=bins, alpha=0.5, density=True, label=metodo)

    plt.xlabel("Bloque óptimo más cercano a cobertura 0.95")
    plt.ylabel("Densidad")
    plt.title("Distribución de tamaños de bloque óptimos por Método")
    plt.legend()
    figure_paths.append(_savefig(outdir, "09_hist_bloque_optimo_por_metodo", show=show))

    # 13) Violines
    plt.figure(figsize=(8, 6))
    data_iso = df[df["Anisotropía"] == "iso"]["Cobertura"].values
    data_aniso = df[df["Anisotropía"] == "aniso"]["Cobertura"].values

    plt.violinplot([data_iso, data_aniso], showmeans=True, showextrema=True)
    plt.axhline(0.95, linestyle="--", linewidth=1)
    plt.xticks([1, 2], ["iso", "aniso"])
    plt.ylabel("Cobertura IC 95%")
    plt.title("Cobertura por Anisotropía — global")
    figure_paths.append(_savefig(outdir, "10a_violin_cobertura_anisotropia_global", show=show))

    plt.figure(figsize=(10, 6))
    grp = [
        df[(df["Anisotropía"] == "iso") & (df["Método"] == "BB")]["Cobertura"].values,
        df[(df["Anisotropía"] == "iso") & (df["Método"] == "CBB")]["Cobertura"].values,
        df[(df["Anisotropía"] == "aniso") & (df["Método"] == "BB")]["Cobertura"].values,
        df[(df["Anisotropía"] == "aniso") & (df["Método"] == "CBB")]["Cobertura"].values,
    ]

    plt.violinplot(grp, showmeans=True, showextrema=True)
    plt.axhline(0.95, linestyle="--", linewidth=1)
    plt.xticks([1, 2, 3, 4], ["iso-BB", "iso-CBB", "aniso-BB", "aniso-CBB"])
    plt.ylabel("Cobertura IC 95%")
    plt.title("Cobertura por Anisotropía y Método")
    figure_paths.append(_savefig(outdir, "10b_violin_cobertura_anisotropia_metodo", show=show))

    # 14) Ajustes agregados: Cobertura
    figure_paths.append(
        _plot_curve_fits_by_group(
            df,
            outdir,
            metric="Cobertura",
            group_col="Rango",
            ref_value=0.95,
            deg=curve_degree,
            title="Ajuste de curvas: Cobertura vs Bloque por Método y Rango",
            filename="11_fit_cobertura_vs_bloque_por_rango",
            show=show,
        )
    )

    figure_paths.append(
        _plot_curve_fits_by_group(
            df,
            outdir,
            metric="Cobertura",
            group_col="Anisotropía",
            ref_value=0.95,
            deg=curve_degree,
            title="Ajuste de curvas: Cobertura vs Bloque por Método y Anisotropía",
            filename="12_fit_cobertura_vs_bloque_por_anisotropia",
            show=show,
        )
    )

    # 15) Ajustes agregados: Ratio
    figure_paths.append(
        _plot_curve_fits_by_group(
            df,
            outdir,
            metric="Ratio Varianzas",
            group_col="Rango",
            ref_value=1.0,
            deg=curve_degree,
            title="Ajuste de curvas: Ratio Varianzas vs Bloque por Método y Rango",
            filename="13_fit_ratio_var_vs_bloque_por_rango",
            show=show,
        )
    )

    figure_paths.append(
        _plot_curve_fits_by_group(
            df,
            outdir,
            metric="Ratio Varianzas",
            group_col="Anisotropía",
            ref_value=1.0,
            deg=curve_degree,
            title="Ajuste de curvas: Ratio Varianzas vs Bloque por Método y Anisotropía",
            filename="14_fit_ratio_var_vs_bloque_por_anisotropia",
            show=show,
        )
    )

    # 16) Ajustes por escenario individual
    fitdir = outdir / "ajustes_por_escenario"
    fitdir.mkdir(exist_ok=True)

    for (rho, rango, an), sub in df.groupby(["Rho", "Rango", "Anisotropía"]):
        # Cobertura
        plt.figure(figsize=(9, 5))

        for metodo in ["BB", "CBB"]:
            d = sub[sub["Método"] == metodo].sort_values("Bloque")
            x = d["Bloque"].values
            y = d["Cobertura"].values

            plt.scatter(x, y, label=f"{metodo} datos")

            fit = _polyfit_curve(x, y, deg=curve_degree)
            if fit is not None:
                xgrid, ygrid, _, r2 = fit
                plt.plot(xgrid, ygrid, label=f"{metodo} ajuste R²={r2:.3f}")

        plt.axhline(0.95, linestyle="--", linewidth=1)
        plt.xlabel("Tamaño de bloque (b)")
        plt.ylabel("Cobertura IC 95%")
        plt.title(f"Ajuste Cobertura | rho={rho}, rango={rango}, anis={an}")
        plt.legend()
        path = fitdir / f"fit_cobertura_rho_{rho}_rango_{rango}_anis_{an}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        figure_paths.append(path)
        if show:
            plt.show()
        else:
            plt.close()

        # Ratio
        plt.figure(figsize=(9, 5))

        for metodo in ["BB", "CBB"]:
            d = sub[sub["Método"] == metodo].sort_values("Bloque")
            x = d["Bloque"].values
            y = d["Ratio Varianzas"].values

            plt.scatter(x, y, label=f"{metodo} datos")

            fit = _polyfit_curve(x, y, deg=curve_degree)
            if fit is not None:
                xgrid, ygrid, _, r2 = fit
                plt.plot(xgrid, ygrid, label=f"{metodo} ajuste R²={r2:.3f}")

        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xlabel("Tamaño de bloque (b)")
        plt.ylabel("Ratio Varianzas")
        plt.title(f"Ajuste Ratio | rho={rho}, rango={rango}, anis={an}")
        plt.legend()
        path = fitdir / f"fit_ratio_rho_{rho}_rango_{rango}_anis_{an}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        figure_paths.append(path)
        if show:
            plt.show()
        else:
            plt.close()

    return {
        "df_long": df_long,
        "df_summary": df,
        "best_blocks": opt_df,
        "figure_paths": figure_paths,
        "outdir": outdir,
        "fitdir": fitdir,
        "summary_csv_path": summary_csv_path,
        "best_blocks_csv_path": best_blocks_csv_path,
    }

def polynomial_fit_summary(x, y, degree: int = 2):
    """
    Ajusta un polinomio y entrega coeficientes, R^2 y vértice si degree=2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    coef = np.polyfit(x, y, degree)
    poly = np.poly1d(coef)

    y_hat = poly(x)

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out = {
        "coef": coef,
        "poly": poly,
        "r2": r2,
    }

    if degree == 2:
        a, b, c = coef
        x_vertex = -b / (2 * a)
        y_vertex = poly(x_vertex)
        out["vertex"] = (float(x_vertex), float(y_vertex))

    return out


def generate_coverage_ratio_polynomial_fits(
    df_or_path,
    outdir: str | Path = "../results/graficas_codispersion_full",
    degree: int = 2,
    show: bool = False,
):
    """
    Genera los ajustes polinómicos:
      1) Cobertura vs Ratio de Varianzas por método.
      2) Cobertura vs Ratio de Varianzas global.

    Acepta:
      - un DataFrame resumen con columnas:
        'Ratio Varianzas', 'Cobertura', 'Método'
      - o una ruta CSV, por ejemplo:
        '../results/escenarios_codispersion_resumen_126_reconstruido.csv'
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if isinstance(df_or_path, (str, Path)):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    x_col = "Ratio Varianzas"
    y_col = "Cobertura"
    method_col = "Método"

    required = {x_col, y_col, method_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    fits = {}

    for method in df[method_col].unique():
        df_m = df[df[method_col] == method].copy()

        x = df_m[x_col].to_numpy()
        y = df_m[y_col].to_numpy()

        fit = polynomial_fit_summary(x, y, degree=degree)
        fits[method] = fit

    # --------------------------------------------------------
    # Figura 1: ajuste por método
    # --------------------------------------------------------
    plt.figure(figsize=(9, 6))

    colors = {
        "BB": "tab:blue",
        "CBB": "tab:orange",
    }

    x_grid = np.linspace(
        df[x_col].min() * 0.95,
        df[x_col].max() * 1.05,
        400,
    )

    for method in df[method_col].unique():
        df_m = df[df[method_col] == method].copy()

        x = df_m[x_col].to_numpy()
        y = df_m[y_col].to_numpy()

        fit = fits[method]
        poly = fit["poly"]

        color = colors.get(method, None)

        plt.scatter(
            x,
            y,
            alpha=0.55,
            s=35,
            label=f"{method} observado",
            color=color,
        )

        plt.plot(
            x_grid,
            poly(x_grid),
            linewidth=2.5,
            color=color,
            label=f"{method} ajuste polinómico",
        )

    plt.axhline(
        0.95,
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.8,
        label="Cobertura nominal 0.95",
    )

    plt.axvline(
        1.0,
        linestyle=":",
        linewidth=1.2,
        color="black",
        alpha=0.8,
        label="Ratio ideal = 1",
    )

    plt.xlabel(
        r"Ratio de varianzas "
        r"$\overline{\mathrm{Var}}^{*(m)}(h)/\widehat{\mathrm{Var}}_{MC}(h)$"
    )
    plt.ylabel("Cobertura empírica")
    plt.title("Cobertura vs exactitud de la varianza bootstrap")
    plt.ylim(0.50, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()

    path_by_method = outdir / "15_cobertura_vs_ratio_varianzas_ajuste_polinomico.png"
    plt.savefig(path_by_method, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    # --------------------------------------------------------
    # Figura 2: ajuste global
    # --------------------------------------------------------
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    fit_global = polynomial_fit_summary(x, y, degree=degree)
    poly_global = fit_global["poly"]

    x_grid = np.linspace(
        df[x_col].min() * 0.95,
        df[x_col].max() * 1.05,
        400,
    )

    plt.figure(figsize=(9, 6))

    plt.scatter(
        x,
        y,
        alpha=0.5,
        s=35,
        color="gray",
        label="Observaciones",
    )

    plt.plot(
        x_grid,
        poly_global(x_grid),
        color="black",
        linewidth=2.5,
        label="Ajuste polinómico global",
    )

    plt.axhline(
        0.95,
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.8,
    )

    plt.axvline(
        1.0,
        linestyle=":",
        linewidth=1.2,
        color="black",
        alpha=0.8,
    )

    plt.xlabel(
        r"Ratio de varianzas "
        r"$\overline{\mathrm{Var}}^{*}(h)/\widehat{\mathrm{Var}}_{MC}(h)$"
    )
    plt.ylabel("Cobertura empírica")
    plt.title("Cobertura vs exactitud de la varianza bootstrap: ajuste global")
    plt.ylim(0.50, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()

    path_global = outdir / "16_cobertura_vs_ratio_varianzas_ajuste_global.png"
    plt.savefig(path_global, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return {
        "fits_by_method": fits,
        "fit_global": fit_global,
        "path_by_method": path_by_method,
        "path_global": path_global,
    }