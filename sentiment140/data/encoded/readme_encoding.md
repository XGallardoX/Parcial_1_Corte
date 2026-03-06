
---
### `bow_vectorizer.pkl`
- **Qué es:** El objeto `CountVectorizer` de scikit-learn ya ajustado (`fit`) con el train.
- **Para qué sirve:** Contiene el vocabulario de 50,000 palabras aprendido. Se necesita para transformar texto nuevo en el futuro (inferencia en producción).
- **⚠️ Importante:** El `fit` se hizo **solo con train**. Nunca re-ajustar con test.

### `X_train_bow.npz`
- **Qué es:** Matriz sparse BoW del train completo (1,360,000 × 50,000).
- **Para qué sirve:** Entrenamiento final del modelo una vez que ya se eligieron los mejores hiperparámetros con `X_tr` y `X_val`.

### `X_test_bow.npz`
- **Qué es:** Matriz sparse BoW del test (240,000 × 50,000).
- **Para qué sirve:** Evaluación final única del modelo en `03_baseline.ipynb`.
- **⚠️ Importante:** No se usa para entrenar ni para ajustar hiperparámetros. Solo para el reporte final.

### `X_tr.npz`
- **Qué es:** 80% del train (1,088,000 filas) en formato sparse.
- **Para qué sirve:** Entrenar los modelos durante la fase de experimentación y ajuste de hiperparámetros.

### `X_val.npz`
- **Qué es:** 20% del train (272,000 filas) en formato sparse.
- **Para qué sirve:** Evaluar el rendimiento durante el desarrollo sin tocar el test. Se usa para comparar modelos y elegir hiperparámetros.

### `y_train.pkl` / `y_test.pkl` / `y_tr.pkl` / `y_val.pkl`
- **Qué son:** Arrays NumPy con los labels (0/1) correspondientes a cada split.
- **Para qué sirven:** Se pasan junto a las matrices X al momento de entrenar y evaluar los modelos.



## Por qué `.npz` y no `.parquet` o `.csv`

Las matrices BoW son **sparse** (>99% ceros). Guardarlas como CSV o Parquet
las convertiría en matrices densas que ocuparían cientos de GB en disco.
`.npz` es el formato nativo de NumPy/SciPy que conserva la estructura sparse:
solo guarda los valores no-cero y sus posiciones, resultando en archivos de
pocos cientos de MB.