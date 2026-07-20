# Datos

El usuario debe descargar el `Smart Agriculture Dataset` de Kaggle y cargarlo manualmente desde la pantalla **Datos** de React o Streamlit. También puede ejecutar `python scripts/download_data.py`.

El cargador busca cualquier CSV/XLSX dentro de `data/raw`, no depende de un nombre fijo y valida estas columnas requeridas: temperatura, humedad, MOI, tipo de suelo, etapa fenológica y `result`. No hay datos sintéticos de respaldo.

El dataset público tiene una tercera etiqueta `result=2` para exceso de agua. Para mantener el objetivo binario del proyecto, esas filas se excluyen explícitamente y quedan registradas en el informe de calidad.
