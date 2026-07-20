# Informe de aceptación

## Estado

La base del proyecto está implementada. El dataset real no se incluye en Git: debe subirse manualmente desde la interfaz o descargarse con Kaggle. Por ello los valores de métricas, tablas y conclusiones se generan al ejecutar el pipeline y no se inventan en este documento.

El archivo real `cropdata_updated.csv` fue validado y cargado durante esta ejecución: contiene 16.411 filas originales; 1.122 tienen `result=2` (exceso de agua) y fueron excluidas explícitamente para mantener el objetivo binario. Quedaron 15.289 filas con clases `0/1`.

El pipeline rápido se ejecutó sobre ese archivo y terminó en aproximadamente 111 segundos en CPU, con las etapas visibles en `/pipeline/status`. Seleccionó el modelo DNN y dejó disponibles los artefactos de modelo, métricas, figuras y reportes.

## Evidencia ejecutada en este entorno

```text
python -m compileall -q backend ml scripts streamlit_app
python -m pytest -q                         # 4 passed
pnpm install                                # frontend
pnpm run build                              # Vite: correcto
pnpm run test                               # 1 test passed
python -m uvicorn backend.app.main:app ...  # /health -> 200
POST /auth/token                            # AUTH_OK
```

La corrida de entrenamiento registrará sus resultados en `artifacts/metrics/`, las figuras en `artifacts/figures/` y los reportes en `artifacts/reports/`. La API publica `/health`, `/data/status`, `/model/metadata`, `/metrics/summary` y `/reports`.

## Limitaciones pendientes de ejecución

- Todavía no se reportan métricas de modelos porque el entrenamiento no se ha lanzado; debe iniciarse explícitamente desde Datos → Iniciar entrenamiento.
- La clase `result=2` se documenta como exceso de agua y se excluye, no se recodifica como riego.
- La instalación de TensorFlow, FastAPI, Streamlit y el resto de dependencias se realizó desde los archivos de requirements.
- La comparación CNN/LSTM es tabular, no temporal.
