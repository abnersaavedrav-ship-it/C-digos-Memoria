# Codispersión y Bootstrap Espacial para Campos 2D

Repositorio para organizar los estudios de simulación asociados al coeficiente de codispersión y su incertidumbre mediante bootstrap espacial.

La idea general es mantener:

- módulos `.py` con funciones reutilizables;
- notebooks `.ipynb` con el flujo principal de cada estudio;
- scripts `.py` para ejecutar estudios desde consola;
- carpeta `results/` para guardar tablas y gráficos.

---

## Estructura del proyecto

```text
codispersion_github_template/
├── src/
│   └── codispersion_bootstrap/
│       ├── __init__.py
│       ├── utils.py
│       ├── codispersion.py
│       ├── bootstrap.py
│       ├── simulation.py
│       ├── experiments.py
│       ├── monte_carlo.py
│       └── plotting.py
├── notebooks/
│   ├── 01_primer_estudio.ipynb
│   ├── 02_segundo_estudio_matern.ipynb
│   ├── 03_tercer_estudio_monte_carlo.ipynb
│   ├── 04_cuarto_estudio_parallel_multi_b.ipynb
│   ├── 05_quinto_estudio_bb_vs_cbb.ipynb
│   └── 06_sexto_estudio_grid.ipynb
├── scripts/
│   ├── run_primer_estudio.py
│   ├── run_segundo_estudio_matern.py
│   ├── run_tercer_estudio_mc.py
│   ├── run_cuarto_estudio_parallel.py
│   ├── run_quinto_estudio_bb_vs_cbb.py
│   └── run_sexto_estudio_grid.py
├── tests/
│   └── test_smoke.py
├── results/
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Qué contiene cada módulo

### `utils.py`

Funciones generales:

- `build_H`: construcción de lags direccionales;
- `tukey2d`: ventana Tukey 2D;
- `standardize_pair`: estandarización de pares de campos;
- `apply_tukey`: aplicación de taper común.

### `codispersion.py`

Funciones del coeficiente de codispersión:

- `delta_h_toroidal`;
- `codisp_rho_hat`;
- `codispersion_by_lag`.

### `bootstrap.py`

Funciones de remuestreo espacial:

- `cbb_sample_pair`: Circular Block Bootstrap 2D pareado con wrap toroidal;
- `bb_lahiri_sample_pair`: bootstrap de bloques solapados sin wrap;
- `bootstrap_codispersion`: bootstrap de `rho_hat(h)` para varios lags.

### `simulation.py`

Funciones de simulación:

- campos gaussianos suaves por FFT;
- simulación Matérn 2D espectral;
- LMC bivariado para generar pares `(X,Y)` con correlación cruzada controlada.

### `experiments.py`

Flujos mínimos:

- `run_min_experiment`: primer estudio;
- `run_min_experiment_matern`: segundo estudio.

### `monte_carlo.py`

Estudios Monte Carlo avanzados:

- `run_monte_carlo_matern_lmc`: tercer estudio;
- `run_study_multi_b_parallel_to_csv`: cuarto estudio paralelo;
- `run_mc_study_lmc`: quinto estudio, comparación BB vs CBB;
- `run_grid_study`: sexto estudio, grilla de escenarios;
- `select_best_b`: selección heurística de tamaño de bloque.

### `plotting.py`

Gráficos tipo paper:

- `plot_variance_ratio`;
- `plot_coverage`.

---

## Cómo correr los notebooks en VS Code

### 1. Abrir el proyecto completo

En VS Code:

```text
File → Open Folder
```

Seleccionar la carpeta completa:

```text
codispersion_github_template
```

No abrir solamente la carpeta `notebooks/`.

---

### 2. Abrir un notebook

Por ejemplo:

```text
notebooks/01_primer_estudio.ipynb
```

---

### 3. Instalar el paquete desde el notebook

Si aparece:

```text
ModuleNotFoundError: No module named 'codispersion_bootstrap'
```

entonces ejecutar esta celda al inicio del notebook:

```python
from pathlib import Path
import sys
import subprocess

cwd = Path.cwd().resolve()

candidates = [
    cwd,
    cwd.parent,
    cwd.parent.parent,
]

ROOT = None
for path in candidates:
    if (path / "pyproject.toml").exists() and (path / "src" / "codispersion_bootstrap").exists():
        ROOT = path
        break

if ROOT is None:
    raise RuntimeError(
        f"No encontré la carpeta raíz del proyecto. Estoy parado en: {cwd}"
    )

print("Carpeta raíz encontrada:", ROOT)

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-e",
    str(ROOT)
])

print("Paquete instalado correctamente.")
```

Después de ejecutar esa celda, reiniciar el kernel:

```text
Restart Kernel
```

Luego correr el notebook desde el inicio.

---

## Orden sugerido de trabajo

### Primer estudio

Notebook:

```text
notebooks/01_primer_estudio.ipynb
```

Script:

```bash
python scripts/run_primer_estudio.py
```

Objetivo:

- simular campos gaussianos suaves;
- aplicar Tukey y estandarización;
- calcular codispersión por lag;
- aplicar CBB pareado.

---

### Segundo estudio

Notebook:

```text
notebooks/02_segundo_estudio_matern.ipynb
```

Script:

```bash
python scripts/run_segundo_estudio_matern.py
```

Objetivo:

- simular campos Matérn bivariados vía LMC;
- usar `rho0` como correlación cruzada objetivo;
- estimar codispersión e intervalos bootstrap.

---

### Tercer estudio

Notebook:

```text
notebooks/03_tercer_estudio_monte_carlo.ipynb
```

Script:

```bash
python scripts/run_tercer_estudio_mc.py
```

Objetivo:

- correr una grilla Monte Carlo de escenarios `(rho0, range_pix, anis_mode)`;
- calcular media de `rho_hat`, varianza Monte Carlo, varianza bootstrap, cobertura y ancho medio del IC;
- usar CBB pareado.

Ejemplo mínimo:

```python
from codispersion_bootstrap import build_H, run_monte_carlo_matern_lmc, monte_carlo_to_dataframe

res = run_monte_carlo_matern_lmc(
    n1=64,
    n2=64,
    H=build_H(hmax=2),
    rhos=(0.2, 0.5),
    ranges=(3.0, 7.0),
    anis_modes=("iso", "aniso"),
    b=12,
    B=50,
    R=10,
)

df = monte_carlo_to_dataframe(res)
df.head()
```

---

### Cuarto estudio

Notebook:

```text
notebooks/04_cuarto_estudio_parallel_multi_b.ipynb
```

Script:

```bash
python scripts/run_cuarto_estudio_parallel.py
```

Objetivo:

- versión paralela del Monte Carlo;
- evaluar varios tamaños de bloque `b`;
- guardar CSV detallado y resumen.

Este estudio usa `joblib`.

Ejemplo mínimo:

```python
from codispersion_bootstrap import build_H, run_study_multi_b_parallel_to_csv

out = run_study_multi_b_parallel_to_csv(
    n1=64,
    n2=64,
    H=build_H(hmax=2),
    rhos=(0.2, 0.5),
    ranges=(3.0,),
    anis_modes=("iso", "aniso"),
    Bsizes=(8, 12),
    B=30,
    R=8,
    n_jobs=-1,
    path_detailed_csv="results/cuarto_estudio_detailed_quick.csv",
    path_summary_csv="results/cuarto_estudio_summary_quick.csv",
)

out["summary"].head()
```

---

### Quinto estudio

Notebook:

```text
notebooks/05_quinto_estudio_bb_vs_cbb.ipynb
```

Script:

```bash
python scripts/run_quinto_estudio_bb_vs_cbb.py
```

Objetivo:

- comparar BB tipo Lahiri contra CBB toroidal pareado;
- calcular `Var_MC`, `Var*_BB`, `Var*_CBB`, ratios y coberturas;
- generar gráficos tipo paper.

Ejemplo mínimo:

```python
from codispersion_bootstrap import Scenario, run_mc_study_lmc, results_to_dataframe

H = [(1,0), (0,1), (1,1), (-1,1), (2,0), (0,2), (2,2), (-2,2)]

sc = Scenario(
    rho0=0.5,
    range_pix=3.5,
    anis_mode="aniso",
    n1=64,
    n2=64,
    B_boot=50,
    b_block=12,
    R_mc=10,
)

res = run_mc_study_lmc(sc, H)
df = results_to_dataframe(res, extra={"b": sc.b_block})
df
```

Gráficos:

```python
from codispersion_bootstrap.plotting import plot_variance_ratio, plot_coverage

plot_variance_ratio(df)
plot_coverage(df)
```

---

### Sexto estudio

Notebook:

```text
notebooks/06_sexto_estudio_grid.ipynb
```

Script:

```bash
python scripts/run_sexto_estudio_grid.py
```

Objetivo:

- correr una grilla completa de escenarios;
- comparar BB y CBB para múltiples `b`;
- seleccionar `b` usando el ratio CBB promedio más cercano a 1.

Ejemplo mínimo:

```python
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
)

best_b = select_best_b(df)
best_b
```

---

## Configuraciones rápidas vs configuraciones finales

Los notebooks y scripts vienen con configuraciones pequeñas para probar que todo funciona.

Ejemplo rápido:

```python
n1 = n2 = 64
B = 30
R = 8
```

Para una corrida más seria, usar algo como:

```python
n1 = n2 = 256
B = 400     # o 800
R = 100     # o más
```

Advertencia: las corridas grandes pueden tardar bastante, especialmente si hay muchos escenarios y varios tamaños de bloque.

---

## Guardar resultados

Ejemplo:

```python
df.to_csv("../results/mi_estudio.csv", index=False)
```

Si se ejecuta desde la raíz del proyecto:

```python
df.to_csv("results/mi_estudio.csv", index=False)
```

---

## Tests

Desde la carpeta raíz del proyecto:

```bash
pip install -e .
pytest
```

---

## Dependencias principales

```text
numpy
pandas
matplotlib
joblib
jupyter
ipykernel
```

---

## Nota metodológica

El flujo general de los estudios es:

1. Simular un par de campos espaciales `(X, Y)`.
2. Preprocesar con estandarización y, cuando corresponda, ventana Tukey.
3. Definir un conjunto de lags direccionales `H`.
4. Calcular `rho_hat(h)` para cada lag.
5. Aplicar bootstrap espacial por bloques.
6. Estimar varianza bootstrap e intervalos de confianza.
7. En estudios Monte Carlo, comparar contra `rho0` o contra una verdad plug-in.
8. Guardar tablas y gráficos para análisis posterior.
