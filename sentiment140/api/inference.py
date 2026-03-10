import time, json, joblib
import numpy as np
from pathlib import Path
from preprocessing import preprocess_batch   # mismo directorio api/

BASE_DIR      = Path(__file__).parent.parent
PIPELINE_PATH = BASE_DIR / "models/best_model.pkl"
CARD_PATH     = BASE_DIR / "models/best_model_card.json"

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Pipeline no encontrado en {PIPELINE_PATH}.\n"
                "Ejecuta: PYTHONPATH=api python3 api/build_pipeline.py"
            )
        _pipeline = joblib.load(PIPELINE_PATH)
    return _pipeline

def get_model_card() -> dict:
    with open(CARD_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_inference(texts: list) -> tuple:
    pipeline = get_pipeline()
    t0     = time.perf_counter()
    preds  = pipeline.predict(texts)
    probas = pipeline.predict_proba(texts)
    return preds, probas, round(time.perf_counter() - t0, 4)

def build_prediction_item(text: str, pred: int, proba: np.ndarray) -> dict:
    return {
        "text":         text,
        "prediction":   "Positivo" if int(pred) == 1 else "Negativo",
        "label":        int(pred),
        "confidence":   round(float(max(proba)), 4),
        "probabilities": {
            "Negativo": round(float(proba[0]), 4),
            "Positivo": round(float(proba[1]), 4),
        },
    }
