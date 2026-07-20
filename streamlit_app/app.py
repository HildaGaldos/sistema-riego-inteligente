"""Panel científico y administrativo conectado al flujo de la API."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_URL = os.getenv("API_URL", "http://localhost:8000")
LABELS = {
    "model": "Modelo", "accuracy": "Exactitud", "precision": "Precisión", "recall": "Sensibilidad", "specificity": "Especificidad",
    "f1": "F1", "roc_auc": "ROC-AUC", "training_seconds": "Entrenamiento (s)", "inference_p95_seconds": "Inferencia P95 (s)",
    "cv_f1_mean": "F1 medio de validación", "cv_f1_std": "Desviación F1 de validación", "fold": "Pliegue", "p_value": "Valor p",
    "p_value_holm": "Valor p (Holm)", "mean_difference": "Diferencia media", "n_folds": "Número de pliegues", "rows": "Registros",
    "rows_excluded": "Filas excluidas", "duplicates": "Duplicados", "warnings": "Advertencias", "backend": "Motor", "trials": "Pruebas",
    "best_value": "Mejor valor", "best_params": "Mejores parámetros", "probability_irrigation": "Probabilidad de riego",
    "predicted_class": "Clase predicha", "recommendation": "Recomendación", "temperature": "Temperatura", "humidity": "Humedad",
    "moi": "MOI", "soil_type": "Tipo de suelo", "crop_stage": "Etapa del cultivo", "interpretation_es": "Interpretación",
    "null_hypothesis_es": "Hipótesis nula", "friedman": "Friedman", "paired_t": "t pareada", "wilcoxon": "Wilcoxon",
    "mcnemar": "McNemar", "available": "Disponible", "reason": "Motivo", "significant": "Significativo",
}


def traducir(value):
    if isinstance(value, dict):
        return {LABELS.get(key, key): traducir(item) for key, item in value.items()}
    if isinstance(value, list):
        return [traducir(item) for item in value]
    return value


def api(method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=120, **kwargs)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", response.text))
    return response


def login():
    st.title("Sistema de Riego Inteligente")
    st.caption("Panel científico y administrativo")
    with st.form("login"):
        username = st.text_input("Usuario", value="admin")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar")
    if submitted:
        try:
            response = requests.post(f"{API_URL}/auth/token", data={"username": username, "password": password}, timeout=30)
            response.raise_for_status()
            st.session_state.token = response.json()["access_token"]
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"No fue posible iniciar sesión: {exc}")


def main():
    st.set_page_config(page_title="Sistema de Riego Inteligente", page_icon="💧", layout="wide")
    if "token" not in st.session_state:
        login()
        return
    with st.sidebar:
        st.header("SIA")
        page = st.radio("Sección", ["Resumen", "Proceso", "Cargar conjunto de datos", "Predicción", "EDA", "Modelos", "Reportes"])
        if st.button("Cerrar sesión"):
            st.session_state.clear()
            st.rerun()
    try:
        if page == "Resumen":
            show_summary()
        elif page == "Proceso":
            show_pipeline()
        elif page == "Cargar conjunto de datos":
            show_upload()
        elif page == "Predicción":
            show_prediction()
        elif page == "EDA":
            show_eda()
        elif page == "Modelos":
            show_models()
        else:
            show_reports()
    except Exception as exc:
        st.error(str(exc))


def show_summary():
    st.title("Resumen operacional")
    status = api("GET", "/data/status").json()
    health = requests.get(f"{API_URL}/health", timeout=15).json()
    training = health.get("training", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Conjunto de datos", "Listo" if status.get("available") else "Pendiente")
    col2.metric("Modelo", "Disponible" if health.get("model_available") else "Pendiente")
    status_labels = {"idle": "En espera", "starting": "Iniciando", "running": "En ejecución", "cancelling": "Cancelando", "completed": "Completado", "failed": "Fallido", "cancelled": "Cancelado"}
    col3.metric("Flujo", status_labels.get(training.get("status", "idle"), training.get("status", "En espera")))
    if training.get("status") in {"running", "cancelling"}:
        st.progress(float(training.get("progress", 0)) / 100, text=f"{training.get('message', 'Procesando')} · {training.get('progress', 0):.0f}%")
        st.caption(f"Tiempo: {training.get('elapsed_seconds', 0)} s · Tiempo restante: {training.get('eta_seconds', '—')} s")
    st.info("Cargue el conjunto de datos y use Proceso para ejecutar todas las etapas conectadas.")


def show_pipeline():
    st.title("Flujo conectado")
    status = api("GET", "/pipeline/status").json()
    st.progress(float(status.get("progress", 0)) / 100, text=f"{status.get('message', 'Listo')} · {status.get('progress', 0):.0f}%")
    status_labels = {"idle": "En espera", "starting": "Iniciando", "running": "En ejecución", "cancelling": "Cancelando", "completed": "Completado", "failed": "Fallido", "cancelled": "Cancelado"}
    stage_labels = {"data_validation": "Carga y validación", "data_cleaning": "Limpieza de datos", "eda": "EDA y calidad", "preprocessing": "Preprocesamiento", "cross_validation": "Validación cruzada", "tuning": "Ajuste de hiperparámetros", "model_training": "Entrenamiento de modelos", "evaluation": "Evaluación y estadísticas", "reports": "Reportes", "ready": "Predicción lista"}
    st.write(f"**Etapa actual:** {stage_labels.get(status.get('stage'), status.get('stage', '—'))} · **Estado:** {status_labels.get(status.get('status'), status.get('status', '—'))}")
    st.caption(f"{status.get('detail', '')} | Tiempo: {status.get('elapsed_seconds', 0)} s | Tiempo restante: {status.get('eta_seconds', '—')} s")
    for stage in status.get("stages", []):
        done = stage.get("id") == "ready" and status.get("status") == "completed"
        st.write(f"{'✅' if done else '•'} {stage.get('label')}")
    col1, col2 = st.columns(2)
    with col1:
        fast = st.checkbox("Modo rápido", value=True, key="pipeline_fast")
        if st.button("Ejecutar flujo completo"):
            st.info(api("POST", f"/pipeline/run?fast={str(fast).lower()}").json())
    with col2:
        if st.button("Cancelar"):
            st.warning(api("POST", "/pipeline/cancel").json())


def show_upload():
    st.title("Cargar conjunto de datos manualmente")
    st.write("Suba el archivo CSV/XLSX. Se validan los alias y las clases sin crear datos sintéticos.")
    uploaded = st.file_uploader("Archivo del experimento agrícola", type=["csv", "xlsx", "xls"])
    if uploaded and st.button("Validar y guardar"):
        response = api("POST", "/data/upload", files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
        st.success(f"Archivo guardado: {response.json()['filename']}")
        st.json(traducir(response.json().get("quality", {})))
    status = api("GET", "/data/status").json()
    if status.get("available"):
        st.success(f"Conjunto de datos actual: {status.get('filename')}")
        st.json(traducir(status.get("quality", {})))
        fast = st.checkbox("Modo rápido", value=True, key="upload_fast")
        if st.button("Ejecutar flujo completo"):
            st.info(api("POST", f"/pipeline/run?fast={str(fast).lower()}").json())


def show_prediction():
    st.title("Predicción individual")
    with st.form("prediction"):
        temperature = st.number_input("Temperatura", value=31.5)
        humidity = st.number_input("Humedad relativa", value=48.0, min_value=0.0, max_value=100.0)
        moi = st.number_input("MOI", value=35.0, min_value=0.0)
        soil_values = {"Suelo negro": "Black Soil", "Suelo aluvial": "Alluvial Soil", "Suelo arenoso": "Sandy Soil", "Suelo rojo": "Red Soil", "Suelo arcilloso": "Clay Soil", "Suelo franco": "Loam Soil", "Suelo calcáreo": "Chalky Soil"}
        stage_values = {"Germinación": "Germination", "Etapa de plántula": "Seedling Stage", "Crecimiento vegetativo / desarrollo de raíz o tubérculo": "Vegetative Growth / Root or Tuber Development", "Floración": "Flowering", "Polinización": "Pollination", "Formación de fruto, grano o bulbo": "Fruit/Grain/Bulb Formation", "Maduración": "Maturation", "Cosecha": "Harvest"}
        soil_type = soil_values[st.selectbox("Tipo de suelo", list(soil_values))]
        crop_stage = stage_values[st.selectbox("Etapa fenológica", list(stage_values))]
        crop_id = st.text_input("Identificador del cultivo (opcional)")
        submitted = st.form_submit_button("Predecir")
    if submitted:
        response = api("POST", "/predict", json={"temperature": temperature, "humidity": humidity, "moi": moi, "soil_type": soil_type, "crop_stage": crop_stage, "crop_id": crop_id or None})
        result = response.json()
        st.metric("Probabilidad de riego", f"{result['probability_irrigation']:.1%}")
        st.success(result["recommendation"])


def show_eda():
    st.title("EDA y calidad de los datos")
    st.json(traducir(api("GET", "/eda/summary").json()))
    figures = api("GET", "/eda/figures").json().get("figures", [])
    st.write("Figuras generadas:", [item["filename"] for item in figures])


def show_models():
    st.title("Comparación de modelos")
    payload = api("GET", "/metrics/summary").json()
    rows = payload.get("models", [])
    if rows:
        st.dataframe(pd.DataFrame(rows).rename(columns={key: LABELS.get(key, key) for key in pd.DataFrame(rows).columns}), use_container_width=True)
        st.subheader("Pruebas estadísticas")
        st.json(traducir(api("GET", "/statistics/summary").json()))
    else:
        st.info("Ejecute el flujo con un conjunto de datos real para generar métricas.")


def show_reports():
    st.title("Reportes descargables")
    items = api("GET", "/reports").json().get("reports", [])
    for item in items:
        response = api("GET", f"/reports/{item['filename']}")
        st.download_button(f"Descargar {item['filename']}", response.content, file_name=item["filename"])
    if not items:
        st.info("Los reportes PDF, DOCX y XLSX se generan después del flujo.")


if __name__ == "__main__":
    main()
