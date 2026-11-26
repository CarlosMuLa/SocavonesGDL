# Procesamiento de imágenes: CEM 4.0

Procesamiento del Continuo de Elevaciones Mexicano 4.0 (INEGI, 15 m) para obtener la topografía de cada colonia de Guadalajara y Zapopan. Usa funciones propias de **convolución** y **normalización** con NumPy, como en los ejercicios de la clase.

## Pasos de `01_pipeline_CEM.ipynb`

| Paso | Operación | Filtro |
|---|---|---|
| 1 | Carga y recorte de la imagen (margen de 3 km) | — |
| 2 | Funciones `convolucion()` y `normalizacion()` | — |
| 3 | Suavizado del ruido del radar | Gaussiano 5x5, σ = 1 |
| 4 | Gradiente y pendiente | Sobel 3x3 en x y en y |
| 5 | Entorno y elevación relativa (z − entorno) | Promedio 67x67 ≈ 1 km (separable: 1x67 y 67x1) |
| 6 | Estadísticas en un círculo de 500 m por colonia | — |
| 7 | Comparación alta contra baja inundación | Prueba de permutación |

## Cómo correrlo

```bash
pip install -r requirements.txt
cd procesamiento_imagenes
jupyter notebook 01_pipeline_CEM.ipynb
```

- **Entradas:** `datos/CEM4_GDL_ZAP_recorte.tif` y `datos/colonias_anexo2.csv`.
- **Salidas:** `resultados/colonias_topografia.csv` (con `elev_media`, `pendiente_media` y `pct_area_baja`) y `resultados/pipeline_cem/paso*.png`.
- Tarda unos 25 segundos en correr.

## Reutilizarlo (issues #7 y #8)

En un notebook nuevo dentro de esta carpeta:

```python
%run ./01_pipeline_CEM.ipynb
for radio in [300, 500, 800]:
    for n in [33, 67, 133]:
        zrel = imagen - filtro_promedio(imagen, n)
        tabla = estadisticas_colonias(pd.read_csv(COL), imagen, pendiente, zrel, radio)
        dif, p = prueba_permutacion(tabla.pendiente_media, tabla.grupo_inundacion == "alta")
        print(radio, n, round(dif, 2), p)
```

## Datos

- `CEM4_GDL_ZAP_recorte.tif`: recorte del archivo `14_Jalisco_r15m_v4.tif` del CEM 4.0 (INEGI, 2024), que se descarga en https://www.inegi.org.mx/app/geo2/elevacionesmex/. El archivo completo de Jalisco no se sube porque pesa demasiado.
- `colonias_anexo2.csv`: 605 colonias (puntos de OSM) con su grupo de inundación según el Mapa Único de Inundaciones 2026 (Imeplan) y sus socavones de 2020 a 2024.
