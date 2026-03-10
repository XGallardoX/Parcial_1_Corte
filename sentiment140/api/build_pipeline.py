"""
Ejecutar desde sentiment140/:
    PYTHONPATH=api python3 api/build_pipeline.py
"""
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from preprocessing import preprocess_batch

BASE_DIR   = Path(__file__).parent.parent          # api/ -> sentiment140/
VEC_PATH   = BASE_DIR / "data/encoded/tfidf_vectorizer.pkl"
CLF_PATH   = BASE_DIR / "data/encoded/lrtfidfmodel.pkl"
OUT_PATH   = BASE_DIR / "models/best_model.pkl"

if __name__ == "__main__":
    print(f"Cargando vectorizer : {VEC_PATH}")
    print(f"Cargando modelo     : {CLF_PATH}")

    vec = joblib.load(VEC_PATH)
    clf = joblib.load(CLF_PATH)

    pipeline = Pipeline([
        ("preprocessor", FunctionTransformer(preprocess_batch, validate=False)),
        ("vectorizer",   vec),
        ("classifier",   clf),
    ])

    # Sanity check antes de guardar
    pred  = pipeline.predict(["I love this!"])
    proba = pipeline.predict_proba(["I love this!"])
    label = "Positivo" if pred[0] == 1 else "Negativo"
    print(f"\nSanity check: \"I love this!\" → {label} ({max(proba[0]):.1%})")

    OUT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, OUT_PATH)
    print(f"\n✔  Pipeline guardado: {OUT_PATH}")
