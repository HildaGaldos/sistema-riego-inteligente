"""Optional Kaggle download; manual upload remains the primary workflow."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "data" / "raw"
    target.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Instale kagglehub o cargue manualmente el CSV/XLSX desde la interfaz.") from exc
    try:
        path = kagglehub.dataset_download("chaitanyagopidesi/smart-agriculture-dataset", output_dir=str(target))
        print(f"Dataset descargado en: {path}")
    except Exception as exc:
        raise SystemExit("Kaggle requiere consentimiento/autenticación. Configure KAGGLE_API_TOKEN o cargue el archivo manualmente.\n" + str(exc)) from exc


if __name__ == "__main__":
    main()
