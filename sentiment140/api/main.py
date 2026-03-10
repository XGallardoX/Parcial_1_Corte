"""
main.py — FastAPI: API de inferencia de sentimiento
Parcial 1 NLP | Alan Osorio · Juan Camilo Gallardo · Santiago Diaz
"""
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tabulate import tabulate
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from schemas import PredictInput
from inference import get_model_card, run_inference, build_prediction_item

app = FastAPI(
    title="Sentiment Analysis API — Parcial 1 NLP",
    description="Mejor modelo: **LR + TF-IDF Bigrama** (F1=0.8232, AUC=0.9025). Dataset: Sentiment140 — 1.6 M tweets.",
    version="1.0.0",
)

# ─── helpers ──────────────────────────────────────────────────────────────────
def _f(val, fmt=".4f"):
    """Formatea un float con fmt. Evita f-strings escapados en tiempo de escritura."""
    return format(val, fmt)

def _author_short(author):
    return "JuanCamilo" if "Gallardo" in author else author.split()[0]

# ══════════════════════════════════════════════════════════════════════════════
# DATOS REALES DE MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
ABLATION = [
    # ── ALAN OSORIO ─────────────────────────────────────────────────────────
    dict(run_name="Baseline_BoW-NoElongation",        author="Alan Osorio",          notebook="03_baseline.ipynb",                                 encoding="BoW",              classifier="MultinomialNB",              best_hp="alpha tuning CV5",  test_f1=0.7835, test_auc=0.8559, selected=False),
    dict(run_name="Baseline_BoW-EmojisToText",         author="Alan Osorio",          notebook="03_baseline.ipynb",                                 encoding="BoW",              classifier="MultinomialNB",              best_hp="alpha tuning CV5",  test_f1=0.7826, test_auc=0.8552, selected=False),
    dict(run_name="Baseline_TF-IDF_Unigrama",          author="Alan Osorio",          notebook="03_baseline.ipynb",                                 encoding="TF-IDF Unigrama",  classifier="MultinomialNB",              best_hp="alpha tuning CV5",  test_f1=0.7715, test_auc=0.8406, selected=False),
    dict(run_name="MLP_TFIDF-SVD_Alan_Osorio",         author="Alan Osorio",          notebook="04_Multilayer Perceptron _Alan_Osorio.ipynb",        encoding="TF-IDF Bigrama",   classifier="MLPClassifier",              best_hp="hidden_layers, lr", test_f1=0.7709, test_auc=0.8549, selected=False),
    dict(run_name="LDA_TFIDF-SVD_Alan_Osorio",         author="Alan Osorio",          notebook="04_Linear Discriminant Analysis_Alan_Osorio.ipynb", encoding="TF-IDF Bigrama",   classifier="LinearDiscriminantAnalysis", best_hp="n_components SVD",  test_f1=0.7312, test_auc=0.8092, selected=False),
    dict(run_name="DecisionTree_TFIDF-SVD_Alan_Osorio",author="Alan Osorio",          notebook="04_Decision_Tree_Alan_Osorio.ipynb",                encoding="TF-IDF Bigrama",   classifier="DecisionTreeClassifier",     best_hp="max_depth tuning",  test_f1=0.6471, test_auc=0.7097, selected=False),
    # ── JUAN CAMILO GALLARDO ────────────────────────────────────────────────
    dict(run_name="Baseline_MultinomialNB_BoW",        author="Juan Camilo Gallardo", notebook="03_baseline.ipynb",                                 encoding="BoW",              classifier="MultinomialNB",              best_hp="alpha=0.1 CV5",     test_f1=0.7826, test_auc=0.8552, selected=False),
    dict(run_name="Baseline_TFIDF-Bigrama",            author="Juan Camilo Gallardo", notebook="03_baseline.ipynb",                                 encoding="TF-IDF Bigrama",   classifier="MultinomialNB",              best_hp="alpha tuning",      test_f1=0.8020, test_auc=0.8833, selected=False),
    dict(run_name="LogisticRegression_TFIDFBigrama",   author="Juan Camilo Gallardo", notebook="04_logistic_regression_JuanCamiloGallardo.ipynb",   encoding="TF-IDF Bigrama",   classifier="LogisticRegression",         best_hp="C=1.0, solver=saga", test_f1=0.8232, test_auc=0.9025, selected=True),
    dict(run_name="FFNN_TFIDFBigrama",                 author="Juan Camilo Gallardo", notebook="04_FFNN_JuanCamiloGallardo.ipynb",                  encoding="TF-IDF Bigrama",   classifier="FFNN (PyTorch)",              best_hp="lr, hidden, dropout", test_f1=0.8198, test_auc=0.9024, selected=False),
    dict(run_name="XGBoost_TFIDFBigrama",              author="Juan Camilo Gallardo", notebook="04_xgboost_JuanCamiloGallardo.ipynb",               encoding="TF-IDF Bigrama",   classifier="XGBClassifier",              best_hp="n_est, max_depth",  test_f1=0.7385, test_auc=0.8188, selected=False),
    # ── SANTIAGO DIAZ ───────────────────────────────────────────────────────
    dict(run_name="LinearRegression_TFIDF_Santiago",   author="Santiago Diaz",        notebook="04_models_LinearRegression_santiagodiaz.ipynb",     encoding="TF-IDF Bigrama",   classifier="LinearRegression",           best_hp="regularization",    test_f1=0.8124, test_auc=0.8918, selected=False),
    dict(run_name="DNN_TFIDF-SVD_Santiago",            author="Santiago Diaz",        notebook="04_models_DNN_santiagodiaz.ipynb",                  encoding="TF-IDF Bigrama",   classifier="DNN",                        best_hp="layers, lr, batch", test_f1=0.7773, test_auc=0.8608, selected=False),
    dict(run_name="Baseline_stopwords_BoW",            author="Santiago Diaz",        notebook="03_baseline.ipynb",                                 encoding="BoW",              classifier="MultinomialNB",              best_hp="alpha tuning CV5",  test_f1=0.7651, test_auc=0.8410, selected=False),
    dict(run_name="Baseline_lematizacion_BoW",         author="Santiago Diaz",        notebook="03_baseline.ipynb",                                 encoding="BoW",              classifier="MultinomialNB",              best_hp="alpha tuning CV5",  test_f1=0.7600, test_auc=0.8353, selected=False),
    dict(run_name="Baseline_Puntuacion_BoW",           author="Santiago Diaz",        notebook="03_baseline.ipynb",                                 encoding="BoW",              classifier="MultinomialNB",              best_hp="alpha tuning CV5",  test_f1=0.7582, test_auc=0.8313, selected=False),
    dict(run_name="RandomForest_TFIDF-SVD_Santiago",   author="Santiago Diaz",        notebook="04_models_RandomForest_santiagodiaz.ipynb",         encoding="TF-IDF Bigrama",   classifier="RandomForestClassifier",     best_hp="n_est, max_depth",  test_f1=0.7322, test_auc=0.8073, selected=False),
    # ── REFERENCIA HUGGINGFACE ──────────────────────────────────────────────
    dict(run_name="HuggingFace_DistilBERT_SST2",       author="Alan Osorio (ref.)",   notebook="05_distilbert.ipynb",                               encoding="WordPiece tokenizer", classifier="DistilBERT-SST2",          best_hp="fine-tuned SST-2",  test_f1=0.7081, test_auc=0.7934, selected=False),
]

COMPARISON = {
    "metric": "F1-macro — Sentiment140",
    "datasets": {
        "train_n": 1_360_000,
        "test_n": 240_000,
    },
    "models": [
        {
            "name": "LR + TF-IDF Bigrama (modelo seleccionado)",
            "notebook": "04_logistic_regression_JuanCamiloGallardo.ipynb",
            "parameters": "1.600.000 tweets",
            "train": {
                "n_samples": 1_360_000,
                "duration_sec": 173.78,     
                "duration_min": 20.896,
                "f1_macro": 0.841,
                "auc": 0.9182,
            },
            "test": {
                "n_samples": 240_000,
                "duration_sec": 10.7,
                "duration_min": 0.178,
                "f1_macro": 0.8232,
                "auc": 0.9025,
            },
        },
        {
            "name": "HuggingFace DistilBERT-SST2 (referencia)",
            "notebook": "05_distilbert.ipynb",
            "parameters": "66 M parámetros",
            "train": {
                "n_samples": 1_360_000,
                "duration_sec": 43_181.1,   
                "duration_min": 719.7,
                "f1_macro": 0.7081,
                "auc": 0.7934,
            },
            "test": {
                "n_samples": 240_000,
                "duration_sec": 7_620.2,
                "duration_min": 127.0,
                "f1_macro": 0.7081,
                "auc": 0.7934,
            },
        },
    ],
}

WORK_DISTRIBUTION = [
    dict(member="Alan Osorio",          n_experiments=6, best_model="Baseline_BoW-NoElongation",       best_f1=0.7835,
         experiments=["Baseline_BoW-NoElongation F1=0.7835", "Baseline_BoW-EmojisToText F1=0.7826", "Baseline_TF-IDF_Unigrama F1=0.7715", "MLP_TFIDF-Bigrama F1=0.7709", "LDA_TFIDF-Bigrama F1=0.7312", "DecisionTree_TFIDF-Bigrama F1=0.6471"]),
    dict(member="Juan Camilo Gallardo", n_experiments=5, best_model="LogisticRegression_TFIDFBigrama SELECCIONADO", best_f1=0.8232,
         experiments=["Baseline_MultinomialNB_BoW F1=0.7826", "Baseline_TFIDF-Bigrama F1=0.8020", "FFNN_TFIDFBigrama F1=0.8198", "XGBoost_TFIDFBigrama F1=0.7385", "LogisticRegression_TFIDFBigrama F1=0.8232 SELECCIONADO"]),
    dict(member="Santiago Diaz",        n_experiments=6, best_model="LinearRegression_TFIDF",           best_f1=0.8124,
         experiments=["Baseline_stopwords_BoW F1=0.7651", "Baseline_lematizacion_BoW F1=0.7600", "Baseline_Puntuacion_BoW F1=0.7582", "LinearRegression_TFIDF F1=0.8124", "DNN_TFIDF-Bigrama F1=0.7773", "RandomForest_TFIDF-Bigrama F1=0.7322"]),
]

CONCLUSION = (
    "El ablation study evaluo 17 configuraciones sobre Sentiment140 (1.6 M tweets), "
    "distribuidas entre 3 integrantes. Las variantes de preprocesamiento del baseline "
    "(stopwords, lematizacion, puntuacion) redujeron el F1 hasta 0.7582, demostrando "
    "que tecnicas agresivas de limpieza eliminan senal util en tweets. Entre modelos "
    "avanzados, Logistic Regression + TF-IDF bigramas (Juan Camilo Gallardo, C=1.0, "
    "solver=saga) obtuvo el mejor F1-macro=0.8232 y ROC-AUC=0.9025, superando al "
    "FFNN (0.8198), LinearRegression de Santiago (0.8124) y todos los modelos de arbol. "
    "El modelo seleccionado presenta el mejor balance rendimiento/complejidad: "
    "sin overfitting (gap train/val/test < 2 pp) y tiempo de ejecucion de ~10.7 s."
)


def _build_ascii_table() -> str:
    rows = []
    for e in sorted(ABLATION, key=lambda x: x["test_f1"], reverse=True):
        rows.append([
            e["run_name"][:35],
            _author_short(e["author"]),
            e["encoding"],
            e["classifier"][:22],
            _f(e["test_f1"]),
            _f(e["test_auc"]),
            "SI" if e["selected"] else "",
        ])
    headers = ["Experimento", "Autor", "Encoding", "Clasificador", "F1-test", "AUC-test", "Best"]
    return tabulate(rows, headers=headers, tablefmt="rounded_outline")


def _build_chart_bytes() -> bytes:
    sorted_exp = sorted(ABLATION, key=lambda e: e["test_f1"])
    labels = [e["run_name"][:33] + "  (" + e["author"].split()[0] + ")" for e in sorted_exp]
    f1s = [e["test_f1"] for e in sorted_exp]
    color_map = {
        "Alan Osorio":          "#4C72B0",
        "Juan Camilo Gallardo": "#DD8452",
        "Santiago Diaz":        "#55A868",
        "Alan Osorio (ref.)":   "#9B59B6",
    }
    colors = ["#E84040" if e["selected"] else color_map.get(e["author"], "#888") for e in sorted_exp]
    fig, ax = plt.subplots(figsize=(13, 11))
    bars = ax.barh(labels, f1s, color=colors, edgecolor="white", height=0.65)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8.5)
    ax.set_xlim(0.57, 0.89)
    ax.set_xlabel("F1-macro (Test — 240k tweets)", fontsize=11)
    ax.set_title(
        "Ablation Study — F1-macro por Experimento\n"
        "Parcial 1 NLP | Alan Osorio · Juan Camilo Gallardo · Santiago Diaz",
        fontsize=11, pad=10)
    ax.axvline(x=0.8232, color="#E84040", ls="--", lw=1.5, alpha=0.8)
    ax.legend(handles=[
        Patch(facecolor="#E84040", label="Seleccionado (LR, Juan Camilo)"),
        Patch(facecolor="#DD8452", label="Juan Camilo Gallardo"),
        Patch(facecolor="#4C72B0", label="Alan Osorio"),
        Patch(facecolor="#55A868", label="Santiago Diaz"),
        Patch(facecolor="#9B59B6", label="HuggingFace (referencia)"),
    ], fontsize=9, loc="lower right")
    ax.grid(axis="x", ls="--", alpha=0.35)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/model_info", summary="Descripcion del modelo seleccionado")
def model_info():
    return get_model_card()


@app.post("/predict", summary="Inferencia individual o por lotes")
def predict(body: PredictInput):
    """**Individual:** {"text": "I love this!"} · **Batch:** {"texts": ["Great!", "Terrible..."]}"""
    if body.text is not None:
        texts, single = [body.text], True
    elif body.texts:
        texts, single = body.texts, False
    else:
        raise HTTPException(status_code=422, detail="Provee 'text' o 'texts'.")
    preds, probas, elapsed = run_inference(texts)
    results = [build_prediction_item(t, p, pr) for t, p, pr in zip(texts, preds, probas)]
    if single:
        return {**results[0], "inference_time_sec": elapsed}
    return {"predictions": results, "count": len(results), "inference_time_sec": elapsed}


@app.get("/ablation_chart", summary="Descarga la grafica del ablation study (PNG)",
         response_class=Response)
def ablation_chart():
    """Retorna el PNG directamente. Abrelo en browser o descargalo con:
    curl http://HOST:8000/ablation_chart -o ablation_chart.png"""
    return Response(content=_build_chart_bytes(), media_type="image/png")


@app.get("/ablation_table", summary="Descarga la tabla del ablation study (.txt)",
         response_class=Response)
def ablation_table():
    """Retorna la tabla ASCII como archivo de texto descargable."""
    content = "ABLATION STUDY — Parcial 1 NLP\n"
    content += "Alan Osorio · Juan Camilo Gallardo · Santiago Diaz\n"
    content += "=" * 80 + "\n\n"
    content += _build_ascii_table()
    content += "\n\n" + "=" * 80 + "\n"
    content += "CONCLUSIONES:\n" + CONCLUSION + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=ablation_study.txt"},
    )


@app.get("/ablation_summary", summary="Reporte completo del ablation study")
def ablation_summary():
    return {
        "total_experiments": len(ABLATION),
        "chart_url":  "/ablation_chart",
        "table_url":  "/ablation_table",
        "hints": {
            "chart": "curl http://54.84.197.7:8000/ablation_chart -o ablation_chart.png",
            "table": "curl http://54.84.197.7:8000/ablation_table -o ablation_study.txt",
        },
        "conclusions": CONCLUSION,
    }



@app.get("/comparison", summary="LR+TF-IDF vs DistilBERT (HuggingFace)")
def comparison():
    return COMPARISON


@app.get("/work_distribution", summary="Distribucion de experimentos por miembro")
def work_distribution():
    return {
        "project": "Parcial 1 — NLP | Sentiment140",
        "members": ["Alan Osorio", "Juan Camilo Gallardo", "Santiago Diaz"],
        "total_experiments": 17,
        "hf_reference": 1,
        "distribution": WORK_DISTRIBUTION,
    }
