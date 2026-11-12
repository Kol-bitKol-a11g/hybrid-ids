"""Базова та контрольні моделі порівняльного аналізу (підрозділи 2.4-2.5).

Random Forest - базова модель; SVM, DNN, LSTM, Isolation Forest і One-Class SVM -
контрольні. Усі моделі навчаються на тих самих частинах даних і порівнюються
у двокласовій постановці «норма проти атаки» (аналог табл. 2.23).
Гіперпараметри - табл. 2.12 і 2.21.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import BENIGN, RANDOM_STATE, RF_PARAMS
from .evaluation import binary_metrics

MODELS = ("rf", "svm", "dnn", "lstm", "iforest", "ocsvm")
WINDOW = 10   # довжина вікна послідовних потоків для LSTM


def make_windows(X: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Перетворює послідовність потоків на вікна (n, window, d); вікно i закінчується
    потоком i. Початок послідовності доповнюється першим потоком."""
    padded = np.concatenate([np.repeat(X[:1], window - 1, axis=0), X])
    index = np.arange(len(X))[:, None] + np.arange(window)[None, :]
    return padded[index]


def _keras_classifier(kind: str, n_features: int, n_classes: int):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = tf.keras.Sequential()
    if kind == "dnn":      # 256-128-64-32, dropout 0,3, L2 1e-3
        model.add(tf.keras.Input(shape=(n_features,)))
        for units in (256, 128, 64, 32):
            model.add(layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(1e-3)))
            model.add(layers.Dropout(0.3))
    else:                  # LSTM 128-64 + Dense 32, dropout 0,2
        model.add(tf.keras.Input(shape=(WINDOW, n_features)))
        model.add(layers.LSTM(128, return_sequences=True, dropout=0.2))
        model.add(layers.LSTM(64, dropout=0.2))
        model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(n_classes, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    return model


def _run(name: str, X_tr, y_tr, X_te, max_samples: int):
    """Навчає модель і повертає (час навчання, с; прогноз атаки; оцінка підозрілості)."""
    rng = np.random.default_rng(RANDOM_STATE)
    attack_tr = (y_tr != BENIGN).astype(int)
    started = time.perf_counter()
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(**RF_PARAMS).fit(X_tr, y_tr)
        predict = lambda X: 1.0 - model.predict_proba(X)[:, list(model.classes_).index(BENIGN)]
        threshold = 0.5
    elif name == "svm":    # ядрові методи навчаються на підвибірці: витрати ростуть надлінійно
        from sklearn.svm import SVC

        idx = rng.choice(len(X_tr), min(max_samples, len(X_tr)), replace=False)
        model = SVC(C=1.0, gamma="scale", kernel="rbf").fit(X_tr[idx], attack_tr[idx])
        predict, threshold = model.decision_function, 0.0
    elif name == "iforest":
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(n_estimators=200, contamination=0.18, random_state=RANDOM_STATE).fit(X_tr)
        predict, threshold = (lambda X: -model.decision_function(X)), 0.0
    elif name == "ocsvm":  # однокласове навчання: лише нормальні потоки
        from sklearn.svm import OneClassSVM

        normal = X_tr[attack_tr == 0]
        idx = rng.choice(len(normal), min(max_samples, len(normal)), replace=False)
        model = OneClassSVM(nu=0.1, gamma="scale", kernel="rbf").fit(normal[idx])
        predict, threshold = (lambda X: -model.decision_function(X)), 0.0
    elif name in ("dnn", "lstm"):
        classes, encoded = np.unique(y_tr, return_inverse=True)
        shape = make_windows if name == "lstm" else (lambda X: X)
        model = _keras_classifier(name, X_tr.shape[1], len(classes))
        batch, epochs = (256, 30) if name == "lstm" else (512, 50)
        model.fit(shape(X_tr), encoded, batch_size=batch, epochs=epochs, verbose=0)
        benign = list(classes).index(BENIGN)
        predict = lambda X: 1.0 - model.predict(shape(X), batch_size=4096, verbose=0)[:, benign]
        threshold = 0.5
    else:
        raise ValueError(f"Невідома модель: {name}")
    train_s = time.perf_counter() - started
    started = time.perf_counter()
    score = np.asarray(predict(X_te))
    infer_ms = (time.perf_counter() - started) * 1000.0 / max(len(X_te), 1)
    return train_s, infer_ms, score > threshold, score


def compare_models(X_train, y_train, X_test, y_test, models=("rf", "svm", "iforest", "ocsvm"),
                   max_samples: int = 50_000) -> pd.DataFrame:
    """Порівнює моделі на однакових розбиттях; повертає таблицю метрик і часу.

    X_* мають бути нормалізовані параметрами навчальної частини. Моделі dnn і lstm
    потребують TensorFlow; для lstm потоки мають бути впорядковані за часом.
    """
    X_train, X_test = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
    y_train, is_attack = np.asarray(y_train), np.asarray(y_test) != BENIGN
    rows = {}
    for name in models:
        train_s, infer_ms, predicted, score = _run(name, X_train, y_train, X_test, max_samples)
        rows[name] = {**binary_metrics(is_attack, predicted, score),
                      "train_s": round(train_s, 2), "infer_ms_per_flow": round(infer_ms, 4)}
    return pd.DataFrame.from_dict(rows, orient="index")
