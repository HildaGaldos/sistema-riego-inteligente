"""FastAPI application for uploads, authentication, training and inference."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
ML_SRC = ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from .auth import AuthStore, create_token, decode_token
from .services.dataset_service import dataset_status, save_uploaded_dataset
from .services.predictor import Predictor
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-a-local-env")
DATA_DIR = os.getenv("DATA_DIR", str(ROOT / "data" / "raw"))
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts"))
auth_store = AuthStore(ROOT / "data" / "users.sqlite3")
predictor = Predictor(ARTIFACTS_DIR)
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token")
PIPELINE_STAGES = [
    {"id": "data_validation", "label": "Carga y validación"},
    {"id": "data_cleaning", "label": "Limpieza de datos"},
    {"id": "eda", "label": "EDA y calidad"},
    {"id": "preprocessing", "label": "Preprocesamiento"},
    {"id": "cross_validation", "label": "Validación cruzada"},
    {"id": "tuning", "label": "Ajuste de hiperparámetros"},
    {"id": "model_training", "label": "Entrenamiento de modelos"},
    {"id": "evaluation", "label": "Evaluación y estadísticas"},
    {"id": "reports", "label": "Reportes"},
    {"id": "ready", "label": "Predicción lista"},
]
STATE_PATH = Path(ARTIFACTS_DIR) / "metrics" / "pipeline_status.json"
state_lock = threading.Lock()
pipeline_start_lock = threading.Lock()
cancel_event = threading.Event()


def _initial_pipeline_state() -> dict:
    if predictor.available and (Path(ARTIFACTS_DIR) / "models" / "model_metadata.json").exists():
        return {"status": "completed", "stage": "ready", "progress": 100, "message": "Modelo persistido disponible", "detail": "Puede usar predicción y reportes", "elapsed_seconds": 0, "eta_seconds": 0, "stages": PIPELINE_STAGES, "history": []}
    return {"status": "idle", "stage": "data_validation", "progress": 0, "message": "Listo para iniciar", "detail": "Cargue un dataset real", "elapsed_seconds": 0, "eta_seconds": None, "stages": PIPELINE_STAGES, "history": []}


def _read_pipeline_state() -> dict:
    if STATE_PATH.exists():
        try:
            saved = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            state = {**_initial_pipeline_state(), **saved}
            # A process restart cannot keep a daemon training thread alive.
            # Recover stale runtime states when persisted model artifacts exist.
            if state.get("status") in {"starting", "running", "cancelling"} and predictor.available:
                return {**state, "status": "completed", "stage": "ready", "progress": 100, "message": "Modelo persistido disponible", "detail": "Se recuperó el último resultado guardado; puede iniciar un nuevo análisis.", "eta_seconds": 0}
            return state
        except (OSError, json.JSONDecodeError):
            pass
    return _initial_pipeline_state()


training_state = _read_pipeline_state()


def _save_pipeline_state() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(training_state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _update_pipeline_state(**changes) -> dict:
    global training_state
    with state_lock:
        training_state = {**training_state, **changes}
        _save_pipeline_state()
        return dict(training_state)

app = FastAPI(title="API del Sistema de Riego Inteligente", version="0.1.0", docs_url="/docs" if os.getenv("ALLOW_DOCS", "true").lower() == "true" else None)
app.add_middleware(CORSMiddleware, allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class PredictionInput(BaseModel):
    temperature: float = Field(..., description="Temperatura ambiental")
    humidity: float = Field(..., ge=0, le=100)
    moi: float = Field(..., ge=0)
    soil_type: str
    crop_stage: str
    crop_id: str | None = None


def current_user(token: str = Depends(oauth2)) -> dict:
    try:
        data = decode_token(token, SECRET_KEY)
        return {"username": data["sub"], "is_admin": auth_store.is_admin(data["sub"])}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc


@app.get("/health")
def health():
    return {"status": "ok", "model_available": predictor.available, "training": dict(training_state)}


@app.post("/auth/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not auth_store.authenticate(form_data.username, form_data.password):
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    return {"access_token": create_token(form_data.username, SECRET_KEY, int(os.getenv("JWT_EXPIRE_MINUTES", "60"))), "token_type": "bearer"}


@app.get("/auth/me")
def me(user: dict = Depends(current_user)):
    return user


@app.post("/data/upload")
def upload_dataset(file: UploadFile = File(...), user: dict = Depends(current_user)):
    try:
        return save_uploaded_dataset(file, DATA_DIR)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/data/status")
def data_status(user: dict = Depends(current_user)):
    return dataset_status(DATA_DIR)


@app.post("/predict")
def predict(payload: PredictionInput, user: dict = Depends(current_user)):
    try:
        return predictor.predict_one(payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(file: UploadFile = File(...), user: dict = Depends(current_user)):
    if not predictor.available:
        raise HTTPException(status_code=503, detail="No hay modelo persistido. Entrene primero.")
    from irrigation_ml.data import read_table
    import tempfile

    suffix = Path(file.filename or ".csv").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=422, detail="Suba un CSV/XLSX de entradas sin la columna result.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(file.file.read()); temp_path = Path(temp.name)
    try:
        frame = read_table(temp_path)
        if "result" not in frame.columns:
            frame["result"] = 0
        validation = __import__("irrigation_ml.data", fromlist=["validate_dataframe"]).validate_dataframe(frame)
        values = predictor.preprocessor.transform(validation.dataframe.drop(columns="result")).astype("float32")
        probabilities = predictor.model.predict(values, verbose=0).reshape(-1)
        return {"rows": [{**row, "probability_irrigation": float(prob), "predicted_class": int(prob >= predictor.threshold), "recommendation": "Requiere riego" if prob >= predictor.threshold else "No requiere riego"} for row, prob in zip(validation.dataframe.drop(columns="result").to_dict(orient="records"), probabilities)]}
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/model/metadata")
def model_metadata(user: dict = Depends(current_user)):
    path = Path(ARTIFACTS_DIR) / "models" / "model_metadata.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Todavía no hay modelo entrenado.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/model/artifacts/{filename}")
def model_artifact(filename: str, user: dict = Depends(current_user)):
    """Download the persisted best model without launching training again."""

    allowed = {"best_irrigation_model.h5", "best_irrigation_model.keras", "preprocessor.joblib", "threshold.json"}
    if Path(filename).name not in allowed:
        raise HTTPException(status_code=404, detail="Artefacto de modelo no disponible")
    root = (Path(ARTIFACTS_DIR) / ("preprocessors" if filename == "preprocessor.joblib" else "models")).resolve()
    path = (root / Path(filename).name).resolve()
    if path.parent != root or not path.exists():
        raise HTTPException(status_code=404, detail="Artefacto de modelo no encontrado")
    from fastapi.responses import FileResponse
    return FileResponse(path)


@app.get("/metrics/summary")
def metrics_summary(user: dict = Depends(current_user)):
    path = Path(ARTIFACTS_DIR) / "metrics" / "model_comparison.csv"
    if not path.exists():
        return {"available": False, "models": []}
    import pandas as pd
    return {"available": True, "models": json.loads(pd.read_csv(path).to_json(orient="records"))}


@app.get("/eda/summary")
def eda_summary(user: dict = Depends(current_user)):
    path = Path(ARTIFACTS_DIR) / "metrics" / "data_quality.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"available": False}


@app.get("/eda/figures")
def eda_figures(user: dict = Depends(current_user)):
    root = Path(ARTIFACTS_DIR) / "figures"
    return {"figures": [{"filename": p.name, "size": p.stat().st_size} for p in sorted(root.glob("*.png"))]}


@app.get("/eda/figures/{filename}")
def eda_figure(filename: str, user: dict = Depends(current_user)):
    root = (Path(ARTIFACTS_DIR) / "figures").resolve()
    path = (root / Path(filename).name).resolve()
    if path.parent != root or not path.exists():
        raise HTTPException(status_code=404, detail="Figura no encontrada")
    from fastapi.responses import FileResponse
    return FileResponse(path, media_type="image/png")


@app.get("/metrics/cv")
def cv_summary(user: dict = Depends(current_user)):
    path = Path(ARTIFACTS_DIR) / "metrics" / "cv_results.csv"
    if not path.exists():
        return {"available": False, "rows": []}
    import pandas as pd
    return {"available": True, "rows": json.loads(pd.read_csv(path).to_json(orient="records"))}


@app.get("/tuning/summary")
def tuning_summary(user: dict = Depends(current_user)):
    path = Path(ARTIFACTS_DIR) / "metrics" / "tuning.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"available": False}


@app.get("/statistics/summary")
def statistics_summary(user: dict = Depends(current_user)):
    path = Path(ARTIFACTS_DIR) / "metrics" / "statistical_tests.json"
    if not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Migrate reports produced by older runs so the new paired tests become
    # visible without forcing a full retraining only to rebuild statistics.
    if "paired_t" not in payload or "formulas_es" not in payload:
        cv_path = Path(ARTIFACTS_DIR) / "metrics" / "cv_results.csv"
        if cv_path.exists():
            import sys
            sys.path.insert(0, str(ROOT / "ml" / "src"))
            from irrigation_ml.evaluation import statistical_tests
            import pandas as pd
            payload = statistical_tests(pd.read_csv(cv_path), {})
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            try:
                from irrigation_ml.data import discover_dataset, load_and_validate
                from irrigation_ml.reports import generate_reports
                comparison_path = Path(ARTIFACTS_DIR) / "metrics" / "model_comparison.csv"
                if comparison_path.exists():
                    validation = load_and_validate(discover_dataset(DATA_DIR))
                    generate_reports(Path(ARTIFACTS_DIR), validation.dataframe, pd.read_csv(comparison_path), pd.read_csv(cv_path))
            except Exception:
                # A report refresh must not make the statistics endpoint fail.
                pass
    return payload


@app.get("/reports")
def reports(user: dict = Depends(current_user)):
    root = Path(ARTIFACTS_DIR) / "reports"
    allowed = {".pdf", ".docx", ".xlsx"}
    return {"reports": [{"filename": p.name, "size": p.stat().st_size} for p in sorted(root.glob("*")) if p.is_file() and p.suffix.lower() in allowed]}


@app.get("/reports/{filename}")
def report(filename: str, user: dict = Depends(current_user)):
    root = (Path(ARTIFACTS_DIR) / "reports").resolve()
    path = (root / Path(filename).name).resolve()
    if path.parent != root or not path.exists():
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    from fastapi.responses import FileResponse
    return FileResponse(path)


def _train_background(fast: bool, folds: int, tuning_trials: int | None = None):
    global training_state, predictor
    started = time.time()
    _update_pipeline_state(status="running", stage="data_validation", progress=0, message="Iniciando pipeline", detail="Las etapas se ejecutarán en orden", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"), elapsed_seconds=0, eta_seconds=None, error=None, fast_mode=fast, history=[])
    try:
        import sys
        sys.path.insert(0, str(ROOT / "ml" / "src"))
        from irrigation_ml.training import PipelineCancelled, run_training

        def progress(event: dict):
            if cancel_event.is_set():
                raise PipelineCancelled("El usuario canceló el pipeline")
            current_progress = float(event.get("progress", 0))
            elapsed = time.time() - started
            eta = elapsed * (100 - current_progress) / current_progress if current_progress > 0 else None
            history = list(training_state.get("history", []))
            item = {**event, "at": time.strftime("%H:%M:%S")}
            history.append(item)
            _update_pipeline_state(stage=event.get("stage"), progress=current_progress, message=event.get("message", ""), detail=event.get("detail", ""), elapsed_seconds=round(elapsed, 1), eta_seconds=round(eta, 1) if eta is not None else None, history=history[-80:])

        result = run_training(fast=fast, seed=int(os.getenv("SEED", "42")), folds=folds, tuning_trials=tuning_trials, artifacts_dir=ARTIFACTS_DIR, progress_callback=progress)
        predictor.reload()
        elapsed = time.time() - started
        _update_pipeline_state(status="completed", stage="ready", progress=100, message="Pipeline completado: módulos listos", detail=f"Modelo seleccionado: {result.get('model_name', 'ok')}", elapsed_seconds=round(elapsed, 1), eta_seconds=0, completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"), result={"model_name": result.get("model_name"), "rows": result.get("rows"), "excluded_rows": result.get("excluded_rows"), "folds": result.get("folds")})
    except PipelineCancelled as exc:
        _update_pipeline_state(status="cancelled", message="Pipeline cancelado", detail=str(exc), eta_seconds=None, error=str(exc))
    except Exception as exc:
        _update_pipeline_state(status="failed", message="Pipeline detenido por un error", detail=str(exc), eta_seconds=None, error=str(exc))


def _start_pipeline(fast: bool, user: dict, folds: int | None = None, tuning_trials: int | None = None) -> dict:
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Solo el administrador puede lanzar el pipeline.")
    with pipeline_start_lock:
        if training_state["status"] in {"starting", "running", "cancelling"}:
            return dict(training_state)
        if not dataset_status(DATA_DIR).get("available"):
            raise HTTPException(status_code=422, detail="Cargue y valide un dataset real antes de iniciar el pipeline.")
        requested_folds = max(2, min(int(folds or os.getenv("CV_FOLDS", "5")), 10))
        requested_trials = max(1, min(int(tuning_trials), 100)) if tuning_trials is not None else None
        cancel_event.clear()
        _update_pipeline_state(status="starting", stage="data_validation", progress=0, message="Iniciando pipeline", detail="Las etapas se ejecutarán en orden", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"), elapsed_seconds=0, eta_seconds=None, error=None, fast_mode=fast, history=[])
        threading.Thread(target=_train_background, args=(fast, requested_folds, requested_trials), daemon=True).start()
        return {"status": "started", "message": "Pipeline iniciado: limpieza → EDA → modelos → reportes → predicción."}


@app.post("/train")
def train(fast: bool = True, folds: int | None = None, tuning_trials: int | None = None, user: dict = Depends(current_user)):
    return _start_pipeline(fast, user, folds, tuning_trials)


@app.post("/pipeline/run")
def pipeline_run(fast: bool = True, folds: int | None = None, tuning_trials: int | None = None, user: dict = Depends(current_user)):
    return _start_pipeline(fast, user, folds, tuning_trials)


@app.post("/pipeline/cancel")
def pipeline_cancel(user: dict = Depends(current_user)):
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Solo el administrador puede cancelar el pipeline.")
    if training_state["status"] == "running":
        cancel_event.set()
        _update_pipeline_state(status="cancelling", message="Cancelación solicitada", detail="Se detendrá al terminar la operación actual")
    return dict(training_state)


@app.get("/train/status")
def train_status(user: dict = Depends(current_user)):
    return dict(training_state)


@app.get("/pipeline/status")
def pipeline_status(user: dict = Depends(current_user)):
    return dict(training_state)
