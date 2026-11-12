"""Аномалійний детектор: автоенкодер, навчений лише на нормальному трафіку.

Вихід l_ae - похибка реконструкції потоку (формули (1.9), (2.4)): чим вона більша,
тим сильніше потік відхиляється від моделі нормальної поведінки. У коді похибка
нормована на кількість ознак (середній квадрат відхилення), що відповідає шкалі
функції втрат MSE та порогам tau1, tau2, tau_base. Архітектура - табл. 2.19.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

from .config import AE_PARAMS, FEATURES, RANDOM_STATE

log = logging.getLogger(__name__)


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import tensorflow  # noqa: F401
        return "keras"
    except ImportError:
        log.warning("TensorFlow не знайдено: використано спрощений замінник scikit-learn")
        return "sklearn"


class AutoencoderDetector:
    """Автоенкодер 22-128-64-16-[4]-16-64-128-22.

    backend = "keras" - основна реалізація (TensorFlow/Keras); "sklearn" - спрощений
    замінник тієї самої архітектури (MLPRegressor, без dropout) для демонстрації й тестів.
    """

    def __init__(self, n_features: int = len(FEATURES), params: dict | None = None,
                 backend: str = "auto", random_state: int = RANDOM_STATE) -> None:
        self.n_features = n_features
        self.params = {**AE_PARAMS, **(params or {})}
        self.backend = _resolve_backend(backend)
        self.random_state = random_state
        self.tau_base: float | None = None
        self.model = None

    def _build_keras(self):
        import tensorflow as tf
        from tensorflow.keras import layers, regularizers

        p = self.params
        tf.keras.utils.set_random_seed(self.random_state)
        l2 = regularizers.l2(p["l2"])
        encoder, decoder = list(p["encoder"]), list(reversed(p["encoder"]))
        model = tf.keras.Sequential(name="flow_autoencoder")
        model.add(tf.keras.Input(shape=(self.n_features,)))
        for units in encoder:                                   # енкодер: 128 -> 64 -> 16
            model.add(layers.Dense(units, activation="relu", kernel_regularizer=l2))
            model.add(layers.Dropout(p["dropout"]))
        model.add(layers.Dense(p["latent"], activation="relu", name="latent"))   # стиснення до 4
        for units in decoder:                                   # декодер: 16 -> 64 -> 128
            model.add(layers.Dense(units, activation="relu", kernel_regularizer=l2))
        model.add(layers.Dense(self.n_features, activation="linear"))   # реконструкція
        optimizer = tf.keras.optimizers.Adam(learning_rate=p["learning_rate"], beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def fit(self, X_normal, X_val_normal=None, verbose: int = 0) -> "AutoencoderDetector":
        """Навчає модель відновлювати нормальні потоки (вхід = ціль)."""
        X, p = np.asarray(X_normal, dtype="float32"), self.params
        if self.backend == "keras":
            import tensorflow as tf

            self.model = self._build_keras()
            if X_val_normal is not None:
                X_val = np.asarray(X_val_normal, dtype="float32")
                validation = dict(validation_data=(X_val, X_val))
            else:
                validation = dict(validation_split=0.1)
            stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=p["patience"], restore_best_weights=True)
            self.model.fit(X, X, epochs=p["epochs"], batch_size=p["batch_size"], shuffle=True,
                           callbacks=[stop], verbose=verbose, **validation)
        else:
            from sklearn.neural_network import MLPRegressor

            hidden = (*p["encoder"], p["latent"], *reversed(p["encoder"]))
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=p["l2"],
                batch_size=min(p["batch_size"], len(X)), learning_rate_init=p["learning_rate"],
                max_iter=p["epochs"], early_stopping=len(X) >= 50, n_iter_no_change=p["patience"],
                random_state=self.random_state,
            ).fit(X, X)
        return self

    def reconstruct(self, X) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        if self.backend == "keras":
            return self.model.predict(X, batch_size=4096, verbose=0)
        return self.model.predict(X)

    def reconstruction_error(self, X) -> np.ndarray:
        """l_ae для кожного потоку: середній квадрат відхилення входу від реконструкції."""
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            return np.empty(0)
        return np.mean((X - self.reconstruct(X)) ** 2, axis=1)

    def calibrate_threshold(self, X_val, is_attack) -> float:
        """Базовий поріг tau_base: максимум F1 на валідаційній вибірці (підрозділ 2.4.3)."""
        from sklearn.metrics import precision_recall_curve

        errors = self.reconstruction_error(X_val)
        precision, recall, thresholds = precision_recall_curve(np.asarray(is_attack).astype(int), errors)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        self.tau_base = float(thresholds[int(np.argmax(f1[:-1]))])
        return self.tau_base

    def save(self, directory) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self.backend == "keras":
            self.model.save(path / "autoencoder.keras")
        else:
            joblib.dump(self.model, path / "autoencoder.joblib")
        meta = dict(backend=self.backend, n_features=self.n_features, params=self.params,
                    tau_base=self.tau_base, random_state=self.random_state)
        (path / "autoencoder.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory) -> "AutoencoderDetector":
        path = Path(directory)
        meta = json.loads((path / "autoencoder.json").read_text(encoding="utf-8"))
        detector = cls(meta["n_features"], meta["params"], meta["backend"], meta["random_state"])
        detector.tau_base = meta["tau_base"]
        if detector.backend == "keras":
            import tensorflow as tf

            detector.model = tf.keras.models.load_model(path / "autoencoder.keras")
        else:
            detector.model = joblib.load(path / "autoencoder.joblib")
        return detector
