"""
main.py — FastAPI: API de inferencia de sentimiento
Proyecto grupal Parcial 1 — NLP
Integrantes: Alan Osorio | Juan Camilo Gallardo | Santiago Diaz
Modelo seleccionado: Logistic Regression + TF-IDF Bigrama (Juan Camilo Gallardo)
"""
import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from fastapi import FastAPI, HTTPException
from schemas import PredictInput
from inference import get_model_card, run_inference, build_prediction_item


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API — Parcial 1 NLP",
    description=(
        "Mejor modelo del ablation study: **Logistic Regression + TF-IDF Bigrama** "
        "(Juan Camilo Gallardo). "
        "Dataset: Sentiment140 — 1.6 M tweets."
    ),
    version="1.0.0",
)


# ── Ablation: TODOS los experimentos del grupo ────────────────────────────────
ABLATION = [
    # ── Alan Osorio ──────────────────────────────────────────────────────────
    {
        "experiment": "Decision Tree",
        "author":     "Alan Osorio",
        "notebook":   "04_Decision_Tree_Alan_Osorio.ipynb",
        "encoding":   "TF-IDF",
        "classifier": "DecisionTreeClassifier",
        "best_hp":    None,
        "val_f1":     None, "test_acc": None, "test_f1": None, "test_auc": None,
        "selected":   False,
    },
    {
        "experiment": "Linear Discriminant Analysis",
        "author":     "Alan Osorio",
        "notebook":   "04_Linear Discriminant Analysis_Alan_Osorio.ipynb",
        "encoding":   "TF-IDF",
        "classifier": "LinearDiscriminantAnalysis",
        "best_hp":    None,
        "val_f1":     None, "test_acc": None, "test_f1": None, "test_auc": None,
        "selected":   False,
    },
    {
        "experiment": "Multilayer Perceptron",
        "author":     "Alan Osorio",
        "notebook":   "04_Multilayer Perceptron_Alan_Osorio.ipynb",
        "encoding":   "TF-IDF",
        "classifier": "MLPClassifier",
        "best_hp":    None,
        "val_f1":     None, "test_acc": None, "test_f1": None, "test_auc": None,
        "selected":   False,
    },
    # ── Juan Camilo Gallardo ─────────────────────────────────────────────────
    {
        "experiment": "Baseline — MultinomialNB + BoW",
        "author":     "Juan Camilo Gallardo",
        "notebook":   "03_baseline.ipynb",
        "encoding":   "Bag of Words (CountVectorizer)",
        "classifier": "MultinomialNB",
        "best_hp":    "alpha=0.1",
        "val_f1":     0.8078, "test_acc": 0.7773, "test_f1": 0.7773, "test_auc": None,
        "selected":   False,
    },
    {
        "experiment": "Logistic Regression + TF-IDF Bigrama",
        "author":     "Juan Camilo Gallardo",
        "notebook":   "04_logistic_regression_JuanCamiloGallardo.ipynb",
        "encoding":   "TF-IDF (unigramas+bigramas, 100k features)",
        "classifier": "LogisticRegression",
        "best_hp":    "C=1.0, solver=saga",
        "val_f1":     0.8415, "test_acc": 0.8232, "test_f1": 0.8232, "test_auc": 0.9025,
        "selected":   True,
    },
    {
        "experiment": "FFNN",
        "author":     "Juan Camilo Gallardo",
        "notebook":   "04_FFNN_JuanCamiloGallardo.ipynb",
        "encoding":   "TF-IDF",
        "classifier": "FeedForwardNeuralNetwork",
        "best_hp":    None,
        "val_f1":     None, "test_acc": None, "test_f1": None, "test_auc": None,
        "selected":   False,
    },
    {
        "experiment": "XGBoost",
        "author":     "Juan Camilo Gallardo",
        "notebook":   "04_xgboost_JuanCamiloGallardo.ipynb",
        "encoding":   "TF-IDF",
        "classifier": "XGBClassifier",
        "best_hp":    None,
        "val_f1":     None, "test_acc": None, "test_f1": None, "test_auc": None,
        "selected":   False,
    },
    # ── Santiago Diaz ────────────────────────────────────────────────────────
    {
        "experiment": "DNN",
        "author":     "Santiago Diaz",
        "notebook":   "04_models_DNN_santiagodiaz.ipynb",
        "encoding":   "TF-IDF",
        "classifier": "DeepNeuralNetwork",
        "best_hp":    None,
        "val_f1":     None, "test_acc": None, "test_f1": None, "test_auc": None,
        "selected":   False,
    },
]

COMPARISON = {
    "metric_used": "F1-macro (sentimiento binario)",
    "dataset":     "Sentiment140 — 240 000 tweets (test set)",
    "models": [
        {
            "name":                     "LR + TF-IDF Bigrama (este API)",
            "test_f1":                  0.8232,
            "test_acc":                 0.8232,
            "test_auc":                 0.9025,
            "train_inference_time_sec": None,   # completar desde 05_distilbert.ipynb
            "test_inference_time_sec":  None,
            "parameters":               "~100 000 features × 2 clases",
            "notes":                    "Notebook 04_logistic_regression_JuanCamiloGallardo",
        },
        {
            "name":                     "HuggingFace: DistilBERT (05_distilbert.ipynb)",
            "test_f1":                  None,   # completar desde 05_distilbert.ipynb
            "test_acc":                 None,
            "test_auc":                 None,
            "train_inference_time_sec": None,
            "test_inference_time_sec":  None,
            "parameters":               "~67 M parámetros",
            "notes":                    "Notebook 05_distilbert.ipynb",
        },
    ],
    "note": "Medir tiempos en la misma instancia EC2 para comparabilidad.",
}

WORK_DISTRIBUTION = [
    {
        "member":      "Alan Osorio",
        "experiments": [
            "04_Decision_Tree_Alan_Osorio.ipynb",
            "04_Linear Discriminant Analysis_Alan_Osorio.ipynb",
            "04_Multilayer Perceptron_Alan_Osorio.ipynb",
        ],
        "best_model": "Multilayer Perceptron",
        "test_f1":    None,
    },
    {
        "member":      "Juan Camilo Gallardo",
        "experiments": [
            "03_baseline.ipynb — MultinomialNB + BoW",
            "04_logistic_regression_JuanCamiloGallardo.ipynb — LR + TF-IDF  ✅ SELECCIONADO",
            "04_FFNN_JuanCamiloGallardo.ipynb",
            "04_xgboost_JuanCamiloGallardo.ipynb",
        ],
        "best_model": "Logistic Regression + TF-IDF Bigrama",
        "test_f1":    0.8232,
    },
    {
        "member":      "Santiago Diaz",
        "experiments": [
            "04_models_DNN_santiagodiaz.ipynb",
        ],
        "best_model": "DNN",
        "test_f1":    None,
    },
]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/model_info", summary="Descripción del modelo seleccionado")
def model_info():
    return get_model_card()


@app.post("/predict", summary="Inferencia individual o por lotes (batch)")
def predict(body: PredictInput):
    """
    **Individual:**  `{"text": "I love this product!"}`
    **Batch:**       `{"texts": ["Great!", "Terrible..."]}`
    """
    if body.text is not None:
        texts, single = [body.text], True
    elif body.texts:
        texts, single = body.texts, False
    else:
        raise HTTPException(
            status_code=422,
            detail="Provee \'text\' (str) o \'texts\' (list[str]).",
        )

    preds, probas, elapsed = run_inference(texts)
    results = [build_prediction_item(t, p, pr)
               for t, p, pr in zip(texts, preds, probas)]

    if single:
        return {**results[0], "inference_time_sec": elapsed}
    return {"predictions": results, "count": len(results),
            "inference_time_sec": elapsed}


@app.get("/ablation_summary", summary="Reporte del ablation study grupal")
def ablation_summary():
    # ── Gráfica solo con experimentos que tienen test_f1 ─────────────────────
    with_metrics = [e for e in ABLATION if e["test_f1"] is not None]
    labels = [f"{e['experiment']} ({e['author'].split()[0]})"
              for e in with_metrics]
    f1s    = [e["test_f1"] for e in with_metrics]
    colors = ["#DD8452" if e["selected"] else "#4C72B0"
              for e in with_metrics]

    fig, ax = plt.subplots(figsize=(9, max(3, len(with_metrics) * 0.8)))
    bars = ax.barh(labels, f1s, color=colors, edgecolor="white", height=0.45)
    ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=10)
    ax.set_xlim(0.60, 0.95)
    ax.set_xlabel("F1-macro (Test)", fontsize=11)
    ax.set_title("Ablation Study — F1-macro en Test (experimentos con métricas)",
                 fontsize=11, pad=8)
    ax.legend(
        handles=[
            Patch(facecolor="#DD8452", label="Seleccionado"),
            Patch(facecolor="#4C72B0", label="Otros"),
        ],
        fontsize=9, loc="lower right",
    )
    ax.grid(axis="x", ls="--", alpha=0.4)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode("utf-8")

    conclusion = (
        "El ablation study grupal comparó 8 experimentos sobre Sentiment140 "
        "(1.6 M tweets de Twitter). El baseline MultinomialNB + BoW (Juan Camilo "
        "Gallardo, alpha=0.1) obtuvo F1-macro=0.7773 en test. La configuración "
        "Logistic Regression + TF-IDF bigramas (C=1.0, solver=saga) alcanzó "
        "F1-macro=0.8232 y ROC-AUC=0.9025, siendo el mejor modelo entre los que "
        "tienen métricas completas. TF-IDF captura pesos relativos por documento "
        "y los bigramas preservan contexto local; la LR aprende pesos negativos "
        "sin asumir independencia entre features. "
        "Modelo seleccionado: LR + TF-IDF Bigrama (Juan Camilo Gallardo)."
    )

    return {
        "table":             ABLATION,
        "chart_png_base64":  chart_b64,
        "chart_description": "Barras horizontales F1-macro en test. Naranja=seleccionado.",
        "conclusions":       conclusion,
    }


@app.get("/comparison", summary="Comparación con modelo HuggingFace (DistilBERT)")
def comparison():
    return COMPARISON


@app.get("/work_distribution", summary="Distribución de experimentos por miembro")
def work_distribution():
    return {
        "project":           "Parcial 1 — NLP | Sentiment140",
        "group_members":     ["Alan Osorio", "Juan Camilo Gallardo", "Santiago Diaz"],
        "total_experiments": sum(len(m["experiments"]) for m in WORK_DISTRIBUTION),
        "distribution":      WORK_DISTRIBUTION,
    }
