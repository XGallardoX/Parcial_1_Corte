"""
main.py — FastAPI: API de inferencia de sentimiento
Parcial 1 NLP | Alan Osorio · Juan Camilo Gallardo · Santiago Diaz
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

app = FastAPI(
    title="Sentiment Analysis API — Parcial 1 NLP",
    description="Mejor modelo del ablation study: **LR + TF-IDF Bigrama** (F1=0.8232, AUC=0.9025). Dataset: Sentiment140 — 1.6 M tweets.",
    version="1.0.0",
)

# ══════════════════════════════════════════════════════════════════════════════
# DATOS REALES DE MLFLOW (17 experimentos + 1 referencia HuggingFace)
# ══════════════════════════════════════════════════════════════════════════════
ABLATION = [
    # ── ALAN OSORIO ────────────────────────────────────────────────────────────
    {"run_name":"Baseline_BoW-NoElongation",       "author":"Alan Osorio",          "notebook":"03_baseline.ipynb",                                    "encoding":"BoW",              "classifier":"MultinomialNB",              "preprocessing":"Sin elongación",      "best_hp":"alpha tuning CV5",    "val_f1":None,   "test_f1":0.7835,"test_auc":0.8559,"selected":False},
    {"run_name":"Baseline_BoW-EmojisToText",        "author":"Alan Osorio",          "notebook":"03_baseline.ipynb",                                    "encoding":"BoW",              "classifier":"MultinomialNB",              "preprocessing":"Emojis→texto",        "best_hp":"alpha tuning CV5",    "val_f1":None,   "test_f1":0.7826,"test_auc":0.8552,"selected":False},
    {"run_name":"Baseline_TF-IDF_Unigrama",         "author":"Alan Osorio",          "notebook":"03_baseline.ipynb",                                    "encoding":"TF-IDF Unigrama",  "classifier":"MultinomialNB",              "preprocessing":"Estándar",            "best_hp":"alpha tuning CV5",    "val_f1":None,   "test_f1":0.7715,"test_auc":0.8406,"selected":False},
    {"run_name":"MLP_TFIDF-SVD_Alan_Osorio",        "author":"Alan Osorio",          "notebook":"04_Multilayer Perceptron _Alan_Osorio.ipynb",           "encoding":"TF-IDF+SVD",       "classifier":"MLPClassifier",             "preprocessing":"Estándar",            "best_hp":"hidden_layers, lr",   "val_f1":None,   "test_f1":0.7709,"test_auc":0.8549,"selected":False},
    {"run_name":"LDA_TFIDF-SVD_Alan_Osorio",        "author":"Alan Osorio",          "notebook":"04_Linear Discriminant Analysis_Alan_Osorio.ipynb",    "encoding":"TF-IDF+SVD",       "classifier":"LinearDiscriminantAnalysis","preprocessing":"Estándar",            "best_hp":"n_components SVD",    "val_f1":None,   "test_f1":0.7312,"test_auc":0.8092,"selected":False},
    {"run_name":"DecisionTree_TFIDF-SVD_Alan_Osorio","author":"Alan Osorio",         "notebook":"04_Decision_Tree_Alan_Osorio.ipynb",                   "encoding":"TF-IDF+SVD",       "classifier":"DecisionTreeClassifier",    "preprocessing":"Estándar",            "best_hp":"max_depth tuning",    "val_f1":None,   "test_f1":0.6471,"test_auc":0.7097,"selected":False},
    # ── JUAN CAMILO GALLARDO ───────────────────────────────────────────────────
    {"run_name":"Baseline_MultinomialNB_BoW",       "author":"Juan Camilo Gallardo", "notebook":"03_baseline.ipynb",                                    "encoding":"BoW",              "classifier":"MultinomialNB",              "preprocessing":"Estándar",            "best_hp":"alpha=0.1 CV5",       "val_f1":0.8078, "test_f1":0.7826,"test_auc":0.8552,"selected":False},
    {"run_name":"Baseline_TFIDF-Bigrama",           "author":"Juan Camilo Gallardo", "notebook":"03_baseline.ipynb",                                    "encoding":"TF-IDF Bigrama",   "classifier":"MultinomialNB",              "preprocessing":"Estándar",            "best_hp":"alpha tuning",        "val_f1":None,   "test_f1":0.8020,"test_auc":0.8833,"selected":False},
    {"run_name":"LogisticRegression_TFIDFBigrama",  "author":"Juan Camilo Gallardo", "notebook":"04_logistic_regression_JuanCamiloGallardo.ipynb",      "encoding":"TF-IDF (1-2g 100k)","classifier":"LogisticRegression",       "preprocessing":"Estándar",            "best_hp":"C=1.0, solver=saga",  "val_f1":0.8415, "test_f1":0.8232,"test_auc":0.9025,"selected":True },
    {"run_name":"FFNN_TFIDFBigrama",                "author":"Juan Camilo Gallardo", "notebook":"04_FFNN_JuanCamiloGallardo.ipynb",                     "encoding":"TF-IDF Bigrama",   "classifier":"FFNN (PyTorch)",            "preprocessing":"Estándar",            "best_hp":"lr, hidden, dropout", "val_f1":None,   "test_f1":0.8198,"test_auc":0.9024,"selected":False},
    {"run_name":"XGBoost_TFIDFBigrama",             "author":"Juan Camilo Gallardo", "notebook":"04_xgboost_JuanCamiloGallardo.ipynb",                  "encoding":"TF-IDF Bigrama",   "classifier":"XGBClassifier",             "preprocessing":"Estándar",            "best_hp":"n_est, max_depth",    "val_f1":None,   "test_f1":0.7385,"test_auc":0.8188,"selected":False},
    # ── SANTIAGO DIAZ ──────────────────────────────────────────────────────────
    {"run_name":"LinearRegression_TFIDF_Santiago",  "author":"Santiago Diaz",        "notebook":"04_models_LinearRegression_santiagodiaz.ipynb",        "encoding":"TF-IDF",           "classifier":"LinearRegression",          "preprocessing":"Estándar",            "best_hp":"regularization",      "val_f1":None,   "test_f1":0.8124,"test_auc":0.8918,"selected":False},
    {"run_name":"DNN_TFIDF-SVD_Santiago",           "author":"Santiago Diaz",        "notebook":"04_models_DNN_santiagodiaz.ipynb",                     "encoding":"TF-IDF+SVD",       "classifier":"DNN",                       "preprocessing":"Estándar",            "best_hp":"layers, lr, batch",   "val_f1":None,   "test_f1":0.7773,"test_auc":0.8608,"selected":False},
    {"run_name":"Baseline_stopwords_BoW",           "author":"Santiago Diaz",        "notebook":"03_baseline.ipynb",                                    "encoding":"BoW",              "classifier":"MultinomialNB",              "preprocessing":"Stopwords",           "best_hp":"alpha tuning CV5",    "val_f1":None,   "test_f1":0.7651,"test_auc":0.8410,"selected":False},
    {"run_name":"Baseline_lematizacion_BoW",        "author":"Santiago Diaz",        "notebook":"03_baseline.ipynb",                                    "encoding":"BoW",              "classifier":"MultinomialNB",              "preprocessing":"Lematización",        "best_hp":"alpha tuning CV5",    "val_f1":None,   "test_f1":0.7600,"test_auc":0.8353,"selected":False},
    {"run_name":"Baseline_Puntuacion_BoW",          "author":"Santiago Diaz",        "notebook":"03_baseline.ipynb",                                    "encoding":"BoW",              "classifier":"MultinomialNB",              "preprocessing":"Sin puntuación",      "best_hp":"alpha tuning CV5",    "val_f1":None,   "test_f1":0.7582,"test_auc":0.8313,"selected":False},
    {"run_name":"RandomForest_TFIDF-SVD_Santiago",  "author":"Santiago Diaz",        "notebook":"04_models_RandomForest_santiagodiaz.ipynb",            "encoding":"TF-IDF+SVD",       "classifier":"RandomForestClassifier",    "preprocessing":"Estándar",            "best_hp":"n_est, max_depth",    "val_f1":None,   "test_f1":0.7322,"test_auc":0.8073,"selected":False},
    # ── REFERENCIA HUGGINGFACE ─────────────────────────────────────────────────
    {"run_name":"HuggingFace_DistilBERT_SST2",      "author":"Alan Osorio (ref.)",   "notebook":"05_distilbert.ipynb",                                  "encoding":"Tokenizer WordPiece","classifier":"DistilBERT-SST2",          "preprocessing":"HuggingFace tokenizer","best_hp":"fine-tuned SST-2",   "val_f1":None,   "test_f1":0.7081,"test_auc":0.7934,"selected":False},
]

COMPARISON = {
    "metric": "F1-macro (test set — 240 000 tweets de Sentiment140)",
    "models": [
        {
            "name":                  "LR + TF-IDF Bigrama (modelo seleccionado)",
            "notebook":              "04_logistic_regression_JuanCamiloGallardo.ipynb",
            "test_f1":               0.8232,
            "test_auc":              0.9025,
            "mlflow_duration_sec":   10.7,
            "parameters":            "~100 000 features × 2 clases",
        },
        {
            "name":                  "HuggingFace DistilBERT-SST2 (referencia)",
            "notebook":              "05_distilbert.ipynb",
            "test_f1":               0.7081,
            "test_auc":              0.7934,
            "mlflow_duration_sec":   1.7,
            "parameters":            "~67 M parámetros",
        },
    ],
    "conclusion": (
        "LR + TF-IDF Bigrama supera a DistilBERT-SST2 en +15.1 pp de F1-macro "
        "(0.8232 vs 0.7081). DistilBERT-SST2 fue entrenado sobre reseñas de "
        "películas (SST-2), no sobre tweets, por lo que su dominio no coincide "
        "con Sentiment140. El modelo local es además ~67M parámetros más ligero."
    ),
}

WORK_DISTRIBUTION = [
    {
        "member":      "Alan Osorio",
        "experiments": [
            "Baseline_BoW-NoElongation        F1=0.7835",
            "Baseline_BoW-EmojisToText         F1=0.7826",
            "Baseline_TF-IDF_Unigrama          F1=0.7715",
            "MLP_TFIDF-SVD                     F1=0.7709",
            "LDA_TFIDF-SVD                     F1=0.7312",
            "DecisionTree_TFIDF-SVD            F1=0.6471",
        ],
        "best_model": "Baseline_BoW-NoElongation",
        "best_f1":    0.7835,
    },
    {
        "member":      "Juan Camilo Gallardo",
        "experiments": [
            "Baseline_MultinomialNB_BoW         F1=0.7826",
            "Baseline_TFIDF-Bigrama             F1=0.8020",
            "FFNN_TFIDFBigrama                  F1=0.8198",
            "XGBoost_TFIDFBigrama               F1=0.7385",
            "LogisticRegression_TFIDFBigrama    F1=0.8232  ✅ MEJOR GLOBAL",
        ],
        "best_model": "LogisticRegression_TFIDFBigrama ✅ SELECCIONADO",
        "best_f1":    0.8232,
    },
    {
        "member":      "Santiago Diaz",
        "experiments": [
            "Baseline_stopwords_BoW             F1=0.7651",
            "Baseline_lematizacion_BoW          F1=0.7600",
            "Baseline_Puntuacion_BoW            F1=0.7582",
            "LinearRegression_TFIDF             F1=0.8124",
            "DNN_TFIDF-SVD                      F1=0.7773",
            "RandomForest_TFIDF-SVD             F1=0.7322",
        ],
        "best_model": "LinearRegression_TFIDF",
        "best_f1":    0.8124,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/model_info", summary="Descripción del modelo seleccionado")
def model_info():
    return get_model_card()


@app.post("/predict", summary="Inferencia individual o por lotes")
def predict(body: PredictInput):
    """
    **Individual:** `{"text": "I love this!"}` \n
    **Batch:** `{"texts": ["Great!", "Terrible..."]}`
    """
    if body.text is not None:
        texts, single = [body.text], True
    elif body.texts:
        texts, single = body.texts, False
    else:
        raise HTTPException(status_code=422,
            detail="Provee \'text\' (str) o \'texts\' (list[str]).")

    preds, probas, elapsed = run_inference(texts)
    results = [build_prediction_item(t, p, pr)
               for t, p, pr in zip(texts, preds, probas)]
    if single:
        return {**results[0], "inference_time_sec": elapsed}
    return {"predictions": results, "count": len(results),
            "inference_time_sec": elapsed}


@app.get("/ablation_summary", summary="Reporte completo del ablation study")
def ablation_summary():
    """17 experimentos + referencia HuggingFace + gráfica + conclusiones."""
    sorted_exp = sorted(ABLATION, key=lambda e: e["test_f1"])
    labels = [f"{e['run_name'][:32]} ({e['author'].split()[0]})" for e in sorted_exp]
    f1s    = [e["test_f1"] for e in sorted_exp]
    color_map = {
        "Alan Osorio":          "#4C72B0",
        "Juan Camilo Gallardo": "#DD8452",
        "Santiago Diaz":        "#55A868",
        "Alan Osorio (ref.)":   "#9B59B6",
    }
    colors = ["#E84040" if e["selected"] else color_map.get(e["author"], "#888")
              for e in sorted_exp]

    fig, ax = plt.subplots(figsize=(12, 11))
    bars = ax.barh(labels, f1s, color=colors, edgecolor="white", height=0.65)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    ax.set_xlim(0.58, 0.88)
    ax.set_xlabel("F1-macro (Test — 240k tweets)", fontsize=11)
    ax.set_title(
        "Ablation Study — F1-macro por Experimento\n"
        "Parcial 1 NLP | Alan Osorio · Juan Camilo Gallardo · Santiago Diaz",
        fontsize=11, pad=10)
    ax.axvline(x=0.8232, color="#E84040", ls="--", lw=1.4, alpha=0.8, label="Mejor (LR TF-IDF Bigrama)")
    ax.legend(handles=[
        Patch(facecolor="#E84040", label="✅ Seleccionado (LR, Juan C.)"),
        Patch(facecolor="#DD8452", label="Juan Camilo Gallardo"),
        Patch(facecolor="#4C72B0", label="Alan Osorio"),
        Patch(facecolor="#55A868", label="Santiago Diaz"),
        Patch(facecolor="#9B59B6", label="HuggingFace (referencia)"),
    ], fontsize=8, loc="lower right")
    ax.grid(axis="x", ls="--", alpha=0.35)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode()

    conclusion = (
        "El ablation study evaluó 17 configuraciones sobre Sentiment140 (1.6 M tweets), "
        "distribuidas entre 3 integrantes. Las variantes de preprocesamiento del baseline "
        "(stopwords, lematización, puntuación) redujeron el F1 hasta 0.7582, demostrando "
        "que técnicas agresivas de limpieza eliminan señal útil en tweets. Entre modelos "
        "avanzados, Logistic Regression + TF-IDF bigramas (Juan Camilo Gallardo, C=1.0, "
        "solver=saga) obtuvo el mejor F1-macro=0.8232 y ROC-AUC=0.9025, superando al "
        "FFNN (0.8198), LinearRegression de Santiago (0.8124) y modelos de árbol. "
        "El modelo seleccionado presenta el mejor balance rendimiento/complejidad: "
        "sin overfitting (gap train/val/test < 2 pp) y tiempo de ejecución de ~10.7s."
    )

    return {
        "total_experiments": len(ABLATION),
        "table":             ABLATION,
        "chart_png_base64":  chart_b64,
        "chart_description": "Barras horizontales F1-macro test. Rojo=seleccionado. Colores por integrante.",
        "conclusions":       conclusion,
    }


@app.get("/comparison", summary="LR+TF-IDF vs DistilBERT (HuggingFace)")
def comparison():
    return COMPARISON


@app.get("/work_distribution", summary="Distribución de experimentos por miembro")
def work_distribution():
    return {
        "project":     "Parcial 1 — NLP | Sentiment140",
        "members":     ["Alan Osorio", "Juan Camilo Gallardo", "Santiago Diaz"],
        "total_experiments": 17,
        "hf_reference": 1,
        "distribution": WORK_DISTRIBUTION,
    }
