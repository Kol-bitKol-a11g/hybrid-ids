"""ML-класифікатор потоків: XGBoost, вихід p_xgb - оцінка ймовірності атаки.

Модель багатокласова (норма + категорії атак); p_xgb = 1 - P(BENIGN | x).
Гіперпараметри - табл. 2.15. Оцінка не трактується як калібрована ймовірність.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from .config import BENIGN, XGB_PARAMS

log = logging.getLogger(__name__)
FILENAME = "xgboost_classifier.joblib"


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import xgboost  # noqa: F401
        return "xgboost"
    except ImportError:
        log.warning("Бібліотеку xgboost не знайдено: використано спрощений замінник scikit-learn")
        return "sklearn"


class XGBoostFlowClassifier:
    """Класифікатор мережевих потоків.

    backend = "xgboost" - основна реалізація; "sklearn" - спрощений замінник
    (HistGradientBoostingClassifier) для демонстрації й тестів без XGBoost.
    """

    def __init__(self, params: dict | None = None, backend: str = "auto",
                 benign_label: str = BENIGN) -> None:
        self.params = {**XGB_PARAMS, **(params or {})}
        self.backend = _resolve_backend(backend)
        self.benign_label = benign_label
        self.classes_: np.ndarray | None = None
        self.model = None

    def _build(self, n_classes: int):
        p = dict(self.params)
        if self.backend == "xgboost":
            from xgboost import XGBClassifier

            if n_classes == 2:   # multi:softprob потребує щонайменше трьох класів
                p.update(objective="binary:logistic", eval_metric="logloss")
            return XGBClassifier(**p)
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=min(p["n_estimators"], 200), max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], l2_regularization=p["reg_lambda"],
            random_state=p["random_state"],
        )

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None) -> "XGBoostFlowClassifier":
        """Навчає модель; (X_val, y_val) використовуються лише для контролю mlogloss."""
        y = np.asarray(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.model = self._build(len(self.classes_))
        kwargs = {"sample_weight": sample_weight}
        if self.backend == "xgboost" and X_val is not None:
            y_val = np.asarray(y_val)
            known = np.isin(y_val, self.classes_)   # класи, відсутні в train, не оцінюються
            if known.any():
                val_encoded = np.searchsorted(self.classes_, y_val[known])
                kwargs.update(eval_set=[(np.asarray(X_val)[known], val_encoded)], verbose=False)
        self.model.fit(np.asarray(X), encoded, **kwargs)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.model.predict_proba(np.asarray(X)))

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Повертає (p_xgb, attack_class): оцінку ймовірності атаки та найімовірнішу
        категорію атаки для кожного потоку."""
        proba = self.predict_proba(X)
        benign = np.flatnonzero(self.classes_ == self.benign_label)
        if len(benign) == 0:
            return np.ones(len(proba)), self.classes_[proba.argmax(axis=1)]
        attack_proba = proba.copy()
        attack_proba[:, benign[0]] = -1.0
        return 1.0 - proba[:, benign[0]], self.classes_[attack_proba.argmax(axis=1)]

    def predict_class(self, X) -> np.ndarray:
        """Багатокласовий прогноз (включно з нормою)."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def save(self, directory) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / FILENAME)
        return path / FILENAME

    @staticmethod
    def load(directory) -> "XGBoostFlowClassifier":
        return joblib.load(Path(directory) / FILENAME)
