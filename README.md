# Smart Irrigation AI

Sistema reproducible para estudiar redes neuronales clásicas e híbridas para la predicción binaria de riego. El dataset **se carga manualmente desde la interfaz** (CSV o XLSX) y también existe una descarga opcional desde Kaggle para quien disponga de credenciales.

## Inicio rápido

Requisitos: Python 3.11+ (3.12 funciona), Node.js LTS, npm y, opcionalmente, Docker.

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
pip install -r ml\requirements.txt
python -m pip install -e .

# Terminal 1: API
$env:PYTHONPATH = "ml\src"
uvicorn backend.app.main:app --reload --port 8000

# Terminal 2: React
cd frontend
npm install
npm run dev

# Opcional: dashboard científico
streamlit run streamlit_app\app.py --server.port 8501
```

Abra `http://localhost:5173`. Inicie sesión con las variables `IRRIGATION_ADMIN_USER` y `IRRIGATION_ADMIN_PASSWORD` del archivo `.env`, entre a **EDA**, suba el conjunto de datos y revise la validación. Después pulse **Continuar a Entrenamiento** y lance el flujo conectado desde el Panel principal o Entrenamiento. El sistema nunca crea datos sintéticos ni entrena al iniciar los paneles.

Para conocer el uso de cada pantalla, el significado de las métricas, las pruebas estadísticas y la solución de problemas, consulte la [Guía de usuario](docs/guia_usuario.md).

## Flujo de trabajo

1. Cargar manualmente el archivo público descargado por el usuario (`.csv`, `.xlsx` o `.xls`).
2. Validar aliases de columnas (`temp`, `Temperature`, `MOI`, `Seedling Stage`, `result`, etc.), tipos y valores del objetivo.
3. Revisar el resumen de calidad y la distribución de clases.
4. Lanzar explícitamente el entrenamiento rápido o completo.
5. Comparar MLP, DNN, RBF, CNN_MLP y LSTM_MLP con validación cruzada estratificada.
6. Persistir el mejor modelo, preprocesador, umbral, métricas, figuras y reportes.
7. Consumir el modelo persistido con `/predict` sin reentrenamiento.

## Flujo conectado y tiempos

Desde **Dashboard** y **Entrenamiento** se ve el avance de cada etapa: carga/validación, limpieza, EDA, preprocesamiento, validación cruzada, tuning, modelos, evaluación, reportes y predicción. El estado se guarda en `artifacts/metrics/pipeline_status.json`, incluye tiempo transcurrido, ETA, historial y errores, y se actualiza cada dos segundos en React.

En CPU, con el dataset de 15.289 filas binarias utilizado en esta ejecución, el modo rápido tardó aproximadamente 2 minutos. Es un smoke test con 2 folds, 2 épocas por modelo y sin Optuna. El modo completo conserva 5 folds y tuning; puede tardar bastante más y debe usarse cuando se necesite la comparación científica final.

La predicción individual y por lote quedan habilitadas cuando el pipeline alcanza `ready`; los reportes, EDA, métricas, validación cruzada y pruebas estadísticas se habilitan con los artefactos reales.

La descarga automática opcional es:

```powershell
python scripts\download_data.py
```

Primero intenta `kagglehub.dataset_download("chaitanyagopidesi/smart-agriculture-dataset")`. Si Kaggle solicita autenticación, configure `KAGGLE_API_TOKEN` o use `kagglehub.login()`. No se incluyen tokens en el repositorio.

El dataset público original contiene además `result=2`, que representa exceso de agua. Como el estudio solicitado es binario, el cargador excluye esa clase de forma explícita y registra el número de filas excluidas en `artifacts/metrics/data_quality.json`; nunca la convierte silenciosamente en `result=1`.

## Comandos

```text
make install       # dependencias Python y frontend
make download-data # descarga opcional desde Kaggle
make eda           # EDA del archivo cargado
make train-fast    # entrenamiento reproducible de smoke test
make train-full    # entrenamiento con configuración completa
make reports       # regenerar PDF, DOCX y XLSX
make test          # pruebas backend y ML
make lint          # Ruff/Black si están instalados
make up            # Docker Compose
make down          # apagar Docker Compose
make bootstrap     # exige dataset real cargado o disponible y ejecuta el flujo
```

## Variables principales

Consulte `.env.example`. Las más importantes son `IRRIGATION_ADMIN_USER`, `IRRIGATION_ADMIN_PASSWORD`, `SECRET_KEY`, `FAST_MODE`, `CV_FOLDS`, `EPOCHS` y `API_URL`.

## Estructura

```text
backend/       API FastAPI, autenticación y endpoints operacionales
ml/            carga, EDA, modelos, evaluación, estadísticas y reportes
frontend/      SPA React + TypeScript + Vite
streamlit_app/ dashboard científico/administrativo
data/raw/      dataset real cargado o descargado por el usuario
artifacts/     modelos, métricas, figuras y reportes generados
docs/          metodología y evidencia de aceptación
```

## Nota científica

CNN y LSTM reciben el vector tabular como una secuencia compacta para reproducir la comparación solicitada; esto no equivale a tener una serie temporal real. Las conclusiones se generan únicamente después de ejecutar el pipeline sobre el dataset cargado. Las pruebas con cinco folds tienen potencia limitada y no permiten declarar superioridad cuando `p >= 0.05`.

## Pruebas estadísticas e interpretación

El módulo **Pruebas estadísticas** calcula los resultados sobre los mismos folds de validación cruzada, para que cada comparación sea pareada. La t de Student pareada trabaja con las diferencias por fold `d_i = A_i - B_i`, calcula `t = media(d) / (sd(d) / sqrt(n))`, sus grados de libertad `n - 1` y el intervalo de confianza del 95 %. El tamaño de efecto mostrado es `d_z = media(d) / sd(d)`.

Friedman contrasta simultáneamente el ranking de tres o más modelos en folds completos. Se muestran `chi2`, `p`, grados de libertad y Kendall W. Wilcoxon contrasta cada par con las diferencias no nulas, reporta suma de rangos, estadístico y rango-biserial. Los valores de las comparaciones múltiples se corrigen con Holm. Cada prueba incluye una interpretación automática: significancia, dirección de la diferencia, magnitud del efecto y advertencia cuando no hay evidencia suficiente.

La implementación sigue la definición de observaciones pareadas de [NIST](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm) y las funciones de referencia de [SciPy para t pareada](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html), [Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) y [Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html). Si se ejecuta el modo rápido con menos de tres folds completos, Friedman queda marcado como no disponible; ejecute el modo completo con 5 folds para obtener la prueba global.

## Credenciales y seguridad

Use únicamente credenciales locales de desarrollo definidas en `.env`. Las contraseñas se almacenan como hashes `scrypt`, el JWT es de corta duración y los endpoints operacionales requieren autenticación. Nunca confirme `.env`, tokens o datasets con información sensible.

## Limitaciones actuales

La ejecución completa de TensorFlow puede tardar en CPU. La disponibilidad del dataset de Kaggle depende de su licencia, consentimiento y autenticación; por eso la interfaz exige carga manual como flujo principal. Consulte `docs/acceptance_report.md` para la evidencia de la ejecución realizada en este entorno.
