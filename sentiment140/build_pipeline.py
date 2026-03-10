"""
build_pipeline.py — Ejecutar UNA SOLA VEZ desde la raíz (sentiment140/).
Lee: data/encoded/tfidf_vectorizer.pkl + data/encoded/lrtfidfmodel.pkl
Escribe: models/best_model.pkl
"""
import re, joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

BASE_DIR    = Path(__file__).parent
VEC_PATH    = BASE_DIR / "data" / "encoded" / "tfidf_vectorizer.pkl"
MODEL_PATH  = BASE_DIR / "data" / "encoded" / "lrtfidfmodel.pkl"
OUT_PATH    = BASE_DIR / "models" / "best_model.pkl"

def clean_raw_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess_batch(texts):
    return [clean_raw_text(t) for t in texts]

if __name__ == "__main__":
    print(f"Vectorizer : {VEC_PATH}")
    print(f"Modelo     : {MODEL_PATH}")
    vectorizer = joblib.load(VEC_PATH)
    model      = joblib.load(MODEL_PATH)
    pipeline = Pipeline([
        ("preprocessor", FunctionTransformer(preprocess_batch, validate=False)),
        ("vectorizer",   vectorizer),
        ("classifier",   model),
    ])
    OUT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, OUT_PATH)
    print(f"\n✔  Pipeline guardado en: {OUT_PATH}")
    for t in ["I love this!", "Terrible, never again."]:
        p = pipeline.predict([t])[0]
        pr = pipeline.predict_proba([t])[0]
        print(f"  \"{t}\" → {'Positivo' if p==1 else 'Negativo'} ({max(pr):.1%})")
