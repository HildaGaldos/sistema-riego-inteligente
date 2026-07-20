# Guía de uso del Sistema de Riego Inteligente

## 1. ¿Qué hace el sistema?

El sistema analiza un conjunto de datos agrícola y compara cinco modelos de aprendizaje automático para recomendar si una parcela requiere riego. El flujo completo utiliza las variables de temperatura, humedad, MOI, tipo de suelo y etapa del cultivo.

El análisis está organizado por etapas. Cada módulo consume los resultados generados por el módulo anterior y, cuando el flujo termina, el mejor modelo queda guardado para realizar predicciones sin volver a entrenar.

Los modelos comparados son:

- **MLP:** perceptrón multicapa.
- **DNN:** red neuronal profunda.
- **RBF:** función de base radial.
- **CNN-MLP:** modelo híbrido convolucional con MLP.
- **LSTM-MLP:** modelo híbrido de memoria larga con MLP.

## 2. Cómo iniciar el sistema en Visual Studio Code

Abra la carpeta del proyecto y use dos terminales.

En la primera terminal, inicie la API:

```powershell
$env:PYTHONPATH="ml\src"
python -m uvicorn backend.app.main:app --reload --port 8000
```

En la segunda terminal, inicie la interfaz:

```powershell
cd frontend
npm run dev
```

Abra `http://localhost:5173` en el navegador. Ingrese con el usuario y la contraseña definidos en el archivo `.env`.

## 3. Flujo recomendado

Siga siempre este orden:

1. Panel principal: revise el estado general y el flujo disponible.
2. EDA: cargue manualmente el archivo y valide la limpieza.
3. Entrenamiento: ejecute la comparación de los cinco modelos.
4. Validación cruzada: revise la estabilidad por pliegue.
5. Hiperparámetros: ejecute la optimización si necesita mejorar el modelo.
6. Pruebas estadísticas: interprete las diferencias entre modelos.
7. Reportes: descargue el análisis en PDF, Word o Excel.
8. Predicción: use el modelo guardado para una parcela o para un archivo completo.

Los módulos posteriores permanecen bloqueados hasta que el entrenamiento finaliza correctamente. Esto evita mostrar resultados incompletos o inventados.

## 4. Panel principal

El Panel principal es la introducción y el centro de control del proyecto. Muestra:

- el objetivo del sistema;
- el mejor modelo disponible;
- la puntuación F1 y la exactitud global;
- el número de modelos comparados;
- los registros útiles del conjunto de datos;
- el avance del flujo completo;
- el tiempo transcurrido y el tiempo restante estimado.

Desde aquí puede ir directamente a la carga de datos, al entrenamiento o a los resultados.

## 5. Módulo EDA: carga y limpieza

Este módulo solamente carga y prepara los datos. No muestra las figuras de evaluación ni las tablas de modelos; esas salidas pertenecen al módulo Entrenamiento.

### Cómo cargar el archivo

1. Entre en **EDA · Carga**.
2. Seleccione manualmente un archivo `.csv`, `.xlsx` o `.xls`.
3. Pulse **Validar y limpiar**.
4. Revise el nombre del archivo, los registros útiles, los duplicados y las filas excluidas.
5. Pulse **Continuar a Entrenamiento**.

El sistema acepta alias habituales de las columnas. Por ejemplo, `temp` se normaliza como temperatura, `MOI` como índice de humedad del suelo y `Seedling Stage` como etapa del cultivo.

### Regla de la columna result

El estudio es binario y necesita `result=0` y `result=1`. El valor `result=2` representa exceso de agua y se excluye del análisis; no se convierte en riego. Si existen otros valores, el sistema los informa como inválidos para que el archivo pueda corregirse.

La limpieza también informa faltantes, tipos de datos, duplicados, distribución de clases y columnas normalizadas. Los datos originales no se reemplazan silenciosamente: se guarda una copia limpia en los artefactos del proyecto.

## 6. Módulo Entrenamiento

Aquí se concentra la comparación científica de los cinco modelos. El módulo muestra:

- tarjetas individuales de cada modelo;
- exactitud, precisión, sensibilidad, especificidad, F1 y ROC-AUC;
- tiempo de entrenamiento y latencia de predicción;
- estabilidad media y desviación de la validación cruzada;
- tabla comparativa ordenada por rendimiento;
- matriz de confusión del mejor modelo;
- curvas ROC de todos los modelos;
- mapa de calor de métricas;
- mapa de calor de correlaciones;
- histogramas, diagramas de caja y distribuciones categóricas;
- registro de los últimos eventos del proceso;
- descarga del mejor modelo en formato `.h5`.

### Modo rápido y modo completo

- **Modo rápido:** usa 2 pliegues, 2 épocas y omite la optimización de hiperparámetros. Sirve para comprobar que el sistema funciona.
- **Modo completo:** usa normalmente 5 pliegues, entrena durante más épocas y ejecuta la optimización. Es el modo recomendado para el resultado final del artículo.

El modo rápido puede tardar aproximadamente dos minutos en un equipo de CPU similar al utilizado en esta ejecución. El modo completo tarda más porque entrena cinco modelos en varios pliegues y ejecuta más pruebas.

Al finalizar, el sistema guarda el mejor modelo, el preprocesador, el umbral, las métricas, las figuras y los reportes. Si el modelo ya está guardado, la predicción no vuelve a entrenar.

## 7. Módulo Validación cruzada

La validación cruzada estratificada divide los datos de entrenamiento en varios pliegues, conserva la proporción de las clases y repite la evaluación. El módulo muestra:

- F1 medio;
- desviación de F1;
- cantidad de pliegues calculados;
- cantidad de filas evaluadas;
- detalle de exactitud, precisión, sensibilidad, F1, ROC-AUC y tiempo por modelo y pliegue.

Use **Ejecutar 5 pliegues** para la comparación científica final. Los datos del conjunto de prueba permanecen separados de esta evaluación.

## 8. Módulo Hiperparámetros

Este módulo permite configurar:

- el modelo que se desea optimizar;
- la métrica objetivo: F1, ROC-AUC o exactitud balanceada;
- el número de pruebas de optimización.

El sistema usa Optuna cuando está disponible. El historial muestra el motor utilizado, el número de pruebas, el mejor valor encontrado y los parámetros guardados, como tasa de aprendizaje, abandono, filtros, unidades LSTM y centros RBF.

El conjunto de prueba no se utiliza para escoger los hiperparámetros. Para obtener resultados completos, ejecute el flujo en modo completo.

## 9. Módulo Pruebas estadísticas

Las pruebas se calculan usando los mismos pliegues de validación cruzada para que las comparaciones sean pareadas. El módulo incluye el valor p, la interpretación automática, el tamaño de efecto y la corrección de Holm.

### t de Student pareada

Compara la diferencia media de F1 entre dos modelos en los mismos pliegues:

```text
dᵢ = F1 del modelo A − F1 del modelo B
t = media(d) / [desviación estándar(d) / √n]
gl = n − 1
```

También muestra el intervalo de confianza del 95 % y el tamaño de efecto `d` de Cohen. Si el valor p ajustado es menor que 0,05, se interpreta que existe evidencia de una diferencia media significativa. Si es mayor o igual que 0,05, no se declara superioridad estadística.

### Friedman

Compara al mismo tiempo tres o más modelos mediante sus rangos dentro de cada pliegue. Muestra el estadístico chi cuadrado, los grados de libertad, el valor p y W de Kendall.

Friedman necesita al menos tres modelos y tres pliegues completos. Por eso puede aparecer como pendiente después de una ejecución rápida con solo dos pliegues. Ejecute el modo completo con 5 pliegues para obtener el contraste global.

Si Friedman resulta significativo, indica que hay diferencias globales, pero no identifica por sí solo qué pares son diferentes. Para eso se revisan la t pareada y Wilcoxon.

### Wilcoxon

Es una alternativa no paramétrica para comparar dos modelos pareados. El sistema elimina diferencias iguales a cero, ordena los valores absolutos, suma los rangos positivos y negativos y calcula el estadístico de Wilcoxon.

La interpretación considera el valor p ajustado por Holm. Un valor p ajustado menor que 0,05 indica evidencia de diferencia en los rangos pareados; un valor mayor o igual que 0,05 no permite afirmar superioridad.

### Interpretación responsable

Las pruebas estadísticas no sustituyen la revisión agronómica. Una diferencia estadísticamente significativa no garantiza que el modelo sea el más útil en campo. Revise también la estabilidad, el tiempo de respuesta, la matriz de confusión, la ROC y las condiciones del cultivo.

## 10. Módulo Reportes

El módulo genera y descarga tres formatos:

- **PDF:** resumen visual con tablas, figuras y conclusiones.
- **Word:** informe narrativo editable con metodología, resultados e interpretaciones.
- **Excel:** libro de auditoría con muestra de datos, calidad, métricas, validación cruzada, hiperparámetros, pruebas estadísticas, modelo guardado y predicciones.

El informe incluye la calidad del conjunto de datos, las comparaciones, las figuras, la interpretación de t pareada, Friedman y Wilcoxon, las limitaciones y el modelo seleccionado.

## 11. Módulo Predicción

El módulo utiliza el modelo `.h5` y el preprocesador guardado. No vuelve a entrenar.

### Predicción individual

1. Seleccione **Predicción individual**.
2. Complete temperatura, humedad, MOI, tipo de suelo y etapa del cultivo.
3. Pulse **Calcular recomendación**.
4. Revise la probabilidad de requerir riego, el umbral, la clase predicha y la recomendación.

Los tipos de suelo y las etapas deben corresponder a las categorías usadas por el archivo de entrenamiento. La interfaz muestra los nombres en español, pero conserva internamente las categorías originales para que el modelo pueda procesarlas correctamente.

### Predicción por lote

1. Seleccione **Predicción por lote**.
2. Cargue un CSV o Excel con las columnas de entrada; no incluya `result`.
3. Pulse **Procesar lote**.
4. Revise las filas procesadas y la cantidad que requiere riego.
5. Pulse **Descargar resultados CSV** para guardar las predicciones.

## 12. Archivos generados

Los resultados importantes quedan en estas carpetas:

```text
artifacts/models/       modelo H5, modelo Keras, umbral y metadatos
artifacts/metrics/      calidad, métricas, validación y pruebas estadísticas
artifacts/figures/      matrices, curvas ROC y mapas de calor
artifacts/reports/      PDF, Word y Excel
artifacts/preprocessors/ preprocesador usado antes de la predicción
```

## 13. Problemas frecuentes

### El archivo tarda mucho en cargar

La carga y la limpieza normalmente son rápidas. Lo que tarda más es el entrenamiento. Revise la barra de progreso, el tiempo transcurrido y el tiempo restante. Para comprobar el funcionamiento use el modo rápido; para el resultado final use el modo completo.

### La columna result tiene valores inválidos

Abra el archivo y deje únicamente `0` y `1` para el estudio binario. Si existe `2`, el sistema lo registra como exceso de agua y lo excluye. No lo cambie manualmente a `1` porque alteraría la interpretación del experimento.

### Los módulos posteriores aparecen bloqueados

El entrenamiento todavía no terminó o finalizó con error. Revise el registro del módulo Entrenamiento y vuelva a ejecutar el flujo después de corregir el problema.

### Friedman aparece pendiente

La ejecución tuvo menos de tres pliegues completos. Desactive el modo rápido, seleccione 5 pliegues y vuelva a ejecutar el análisis.

### No aparece el botón de predicción

Debe existir un modelo guardado en `artifacts/models` y el flujo debe mostrar el estado **Completado**. Si el modelo fue eliminado, vuelva a ejecutar el entrenamiento.

## 14. Recomendación final de uso

Para una demostración, use el modo rápido. Para presentar resultados científicos, cargue el archivo real, ejecute el modo completo con 5 pliegues, revise todas las métricas, confirme las interpretaciones estadísticas y descargue los tres reportes.

## 15. Despliegue gratuito en Internet

El proyecto incluye el archivo `render.yaml` para publicarlo en Render usando el plan gratuito. Este archivo crea dos servicios conectados:

- `riego-ia-api`: API de FastAPI ejecutada con Docker.
- `riego-ia-interfaz`: interfaz React publicada como sitio estático.

### Publicación

1. Cree un repositorio vacío en GitHub y suba todo el proyecto.
2. Entre en [render.com](https://render.com), regístrese con GitHub y seleccione **New > Blueprint**.
3. Seleccione el repositorio y confirme el archivo `render.yaml`.
4. En el servicio de la API configure `IRRIGATION_ADMIN_USER` y `IRRIGATION_ADMIN_PASSWORD`.
5. Espere a que Render cree la API y copie su URL, por ejemplo `https://riego-ia-api.onrender.com`.
6. En el servicio de la interfaz configure `VITE_API_URL` con esa URL y vuelva a desplegar.
7. Copie la URL de la interfaz, por ejemplo `https://riego-ia-interfaz.onrender.com`, y configúrela en la API como `FRONTEND_ORIGIN`; después vuelva a desplegar la API.

### Consideraciones del plan gratuito

La interfaz queda disponible mediante una URL pública. La API gratuita puede dormirse después de un periodo sin visitas y tardar aproximadamente un minuto en responder al primer acceso. Además, el almacenamiento local de un servicio gratuito puede perderse al reiniciar o redeployar. Por ello, en una demostración gratuita se debe volver a cargar el dataset y regenerar los modelos si Render reinicia la API. No incluya secretos, contraseñas reales ni datasets privados en el repositorio.

Si Render muestra una pantalla de verificación de pago, no agregue una tarjeta: cierre el flujo y use otro proveedor o ejecute la aplicación localmente. La configuración del proyecto no requiere una tarjeta por parte del código.
