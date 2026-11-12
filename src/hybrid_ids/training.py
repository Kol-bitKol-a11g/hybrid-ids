"""Сценарії експерименту: підготовка даних, навчання моделей, оцінювання (рис. 2.7).

Тестова частина використовується лише після фіксації конфігурації; усі параметри
перетворень і пороги оцінюються на навчальній та валідаційній частинах.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .autoencoder_detector import AutoencoderDetector
from .config import BENIGN, LABEL_COLUMN, Thresholds
from .data_preparation import (MinMaxNormalizer, balance_train, class_weights, clean_flows,
                               ordered_split, time_split)
from .evaluation import ablation, multiclass_metrics
from .feature_extraction import feature_matrix, load_flows
from .pipeline import NORMALIZER_FILE, THRESHOLDS_FILE
from .xgboost_classifier import XGBoostFlowClassifier

log = logging.getLogger(__name__)
BACKENDS = {"auto": ("auto", "auto"), "full": ("xgboost", "keras"), "lite": ("sklearn", "sklearn")}


def prepare_dataset(csv_paths, out_dir, min_class_size: int = 5) -> pd.DataFrame:
    """Очищує потоки та зберігає частини train/val/test (70/15/15) і звіт очищення."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [load_flows(path) for path in csv_paths]
    cleaned, report = clean_flows(pd.concat(frames, ignore_index=True), min_class_size=min_class_size)
    if "Timestamp" in cleaned.columns:
        parts = time_split(cleaned)
    else:
        log.warning("Колонки Timestamp немає: використано поділ за порядком рядків у файлах")
        parts = ordered_split(cleaned)
    for name, part in zip(("train", "val", "test"), parts):
        part.to_csv(out / f"{name}.csv", index=False)
    report.to_csv(out / "cleaning_report.csv", index=False)
    return report


def train_models(train: pd.DataFrame, val: pd.DataFrame, models_dir, backend: str = "auto",
                 balance: bool = True, calibrate: bool = False) -> dict:
    """Навчає нормалізатор, класифікатор XGBoost і автоенкодер; зберігає артефакти.

    calibrate=True додатково зберігає пороги автоенкодера, масштабовані до tau_base
    цієї моделі (для даних, відмінних від CICIDS2017).
    """
    clf_backend, ae_backend = BACKENDS[backend]
    models = Path(models_dir)
    models.mkdir(parents=True, exist_ok=True)
    y_train, y_val = train[LABEL_COLUMN].to_numpy(), val[LABEL_COLUMN].to_numpy()

    # 1. Параметри нормалізації оцінюються лише на train (формула (2.1)).
    X_train_raw, _ = feature_matrix(train)
    normalizer = MinMaxNormalizer().fit(X_train_raw)
    X_train, X_val = normalizer.transform(X_train_raw), normalizer.transform(feature_matrix(val)[0])

    # 2. Обмежене класо-залежне балансування: SMOTE лише для train + ваги класів.
    X_fit, y_fit = X_train, y_train
    if balance:
        try:
            X_fit, y_fit = balance_train(X_train, y_train)
        except ImportError:
            log.warning("imbalanced-learn не встановлено: SMOTE пропущено, застосовано лише ваги класів")
    classifier = XGBoostFlowClassifier(backend=clf_backend)
    classifier.fit(X_fit, y_fit, X_val, y_val, sample_weight=class_weights(y_fit) if balance else None)

    # 3. Автоенкодер навчається тільки на нормальних потоках; tau_base - на валідації.
    autoencoder = AutoencoderDetector(backend=ae_backend)
    autoencoder.fit(X_train[y_train == BENIGN], X_val[y_val == BENIGN])
    val_attack = y_val != BENIGN
    if val_attack.any() and not val_attack.all():
        autoencoder.calibrate_threshold(X_val, val_attack)

    joblib.dump(normalizer, models / NORMALIZER_FILE)
    classifier.save(models)
    autoencoder.save(models)
    if calibrate and autoencoder.tau_base:
        thresholds = Thresholds().scaled_to(autoencoder.tau_base)
        (models / THRESHOLDS_FILE).write_text(json.dumps(asdict(thresholds), indent=2), encoding="utf-8")
    return {"train_flows": len(X_train), "train_flows_balanced": len(X_fit),
            "classes": [str(c) for c in classifier.classes_], "tau_base": autoencoder.tau_base,
            "backends": {"classifier": classifier.backend, "autoencoder": autoencoder.backend}}


def evaluate_models(test: pd.DataFrame, ids, p_sur=None) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Оцінює навчену систему на тестовій частині.

    Повертає таблицю п'яти абляційних конфігурацій, багатокласові метрики
    класифікатора та повноту за класами. ids - екземпляр HybridIDS.
    """
    X_raw, _ = feature_matrix(test)
    X = ids.normalizer.transform(X_raw)
    y_true = test[LABEL_COLUMN].to_numpy()
    p_sur = np.zeros(len(X), dtype=int) if p_sur is None else np.asarray(p_sur)
    p_xgb, _ = ids.classifier.predict(X)
    l_ae = ids.autoencoder.reconstruction_error(X)   # для абляції потрібна похибка всіх потоків
    table = ablation(y_true != BENIGN, p_sur, p_xgb, l_ae, ids.thresholds)
    summary, per_class = multiclass_metrics(y_true, ids.classifier.predict_class(X))
    return table, summary, per_class
