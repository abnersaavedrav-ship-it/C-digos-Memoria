# Codispersión — Aplicación Caso Real

Repositorio para ordenar los códigos de la aplicación real de la memoria: imágenes de nanotubos, diagnóstico, implementación formal del coeficiente, selección de tamaño de bloque y bootstrap espacial pareado CBB/MBB.

La estructura replica la idea usada en `codispersion_github_template`: funciones reutilizables en módulos `.py`, notebooks limpios que ejecutan el flujo principal, scripts para ejecución desde consola, resultados en `results/` y documentos originales en `source_studies/`.

## Estructura

```text
codispersion_real_case_template/
├── data/
│   ├── ch05_nano-a.png
│   ├── ch05_nano-b.png
│   ├── ch05_nano-c.png
│   └── ch05_nano-d.png
├── notebooks/
│   ├── 01_preprocesamiento_diagnostico_caso_real_A.ipynb
│   ├── 02_preprocesamiento_diagnostico_sinteticos_B.ipynb
│   ├── 03_preprocesamiento_diagnostico_microdatos_C.ipynb
│   ├── 04_implementacion_formal_coef.ipynb
│   ├── 05_seleccion_b.ipynb
│   └── 06_bootstrap_espacial_pareado_cbb_mbb.ipynb
├── scripts/
├── source_studies/
├── src/codispersion_real_case/
├── tests/
├── results/
├── requirements.txt
└── pyproject.toml
```

## Instalación rápida en VS Code

Abre la carpeta completa del proyecto en VS Code:

```text
File → Open Folder → codispersion_real_case_template
```

Abre cualquier notebook y ejecuta la primera celda de instalación. Si necesitas hacerlo manualmente desde una celda:

```python
%pip install -e ..
```

Luego reinicia el kernel.

## Orden de notebooks

1. `01_preprocesamiento_diagnostico_caso_real_A.ipynb`
2. `02_preprocesamiento_diagnostico_sinteticos_B.ipynb`
3. `03_preprocesamiento_diagnostico_microdatos_C.ipynb`
4. `04_implementacion_formal_coef.ipynb`
5. `05_seleccion_b.ipynb`
6. `06_bootstrap_espacial_pareado_cbb_mbb.ipynb`

Cada notebook tiene dos configuraciones:

- **TEST / juguete**: usa menos tamaño, menos réplicas o recorte de imágenes para verificar que funciona.
- **FULL / original**: usa imágenes completas 512×512 y parámetros del experimento original.

## Ejemplo de uso rápido

```python
from codispersion_real_case import run_bootstrap_real

out = run_bootstrap_real(
    data_dir="../data",
    test=True,
    crop_n=128,
    B=50,
    b=7,
    scheme="CBB",
    outdir="../results",
)

out["df_ab"]
```

Para la corrida original:

```python
out = run_bootstrap_real(
    data_dir="../data",
    test=False,
    B=800,
    b=57,
    scheme="CBB",
    outdir="../results",
)
```

## Tests

Desde la raíz del proyecto:

```bash
pytest
```

## Notas

- `CBB` usa bloques toroidales con wrap.
- `MBB` usa bloques interiores.
- Los pares principales del caso real son `(a,b)` y `(c,d)`.
- El conjunto de lags base usado en la aplicación completa es:

```python
H = [(1,0), (0,1), (1,1), (-1,1), (2,0), (0,2), (2,2), (-2,2)]
```
