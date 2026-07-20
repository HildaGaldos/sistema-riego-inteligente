from pathlib import Path
import os
import sys
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "ml" / "src"))

from irrigation_ml.training import run_training

if __name__ == "__main__":
    if not (ROOT / "data" / "raw").exists() or not list((ROOT / "data" / "raw").glob("*.csv")) + list((ROOT / "data" / "raw").glob("*.xlsx")):
        raise SystemExit("No hay dataset real en data/raw. Súbalo desde la interfaz o ejecute scripts/download_data.py.")
    print(run_training(fast=True, seed=int(os.getenv("SEED", "42"))))
