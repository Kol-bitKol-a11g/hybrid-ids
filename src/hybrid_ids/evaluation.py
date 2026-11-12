"""Метрики якості та абляційні досліди (підрозділи 1.5, 3.4).

Позитивний клас - «атака». Основна метрика - macro-F1; додатково обчислюються
F1, precision, recall, частота хибних спрацювань FPR = FP / (FP + TN) і ROC-AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .config import Thresholds
from .reconciliation import reconcile_batch
from .schemas import Verdict


def binary_metrics(y_true, y_pred, score=None) -> dict:
    """Метрики двокласової постановки «норма проти атаки» (формула (1.10))."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": float("nan"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, np.nan_to_num(np.asarray(score, dtype=float)))
    return result


def multiclass_metrics(y_true, y_pred) -> tuple[dict, pd.Series]:
    """Метрики багатокласової постановки та повнота кожного класу (аналог табл. 2.25)."""
    labels = np.unique(np.asarray(y_true))
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }
    per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return summary, pd.Series(per_class, index=labels, name="recall")


def ablation(is_attack, p_sur, p_xgb, l_ae, thr: Thresholds = Thresholds()) -> pd.DataFrame:
    """П'ять абляційних конфігурацій на однакових даних і моделях (аналог табл. 3.7).

    У бінарній перевірці всі сформовані оповіщення (рішення «атака» та «перевірка»)
    вважаються позитивними, тому непідтверджені аномалії нормального трафіку входять до FP.
    Для конфігурацій 4 і 5 ROC-AUC рахується за узагальненою оцінкою підозрілості:
    максимумом із p_sur, p_xgb та похибки l_ae, нормованої порогом tau1.
    """
    p_sur, p_xgb, l_ae = (np.asarray(a, dtype=float) for a in (p_sur, p_xgb, l_ae))
    ae_score = np.clip(np.nan_to_num(l_ae) / thr.tau1, 0.0, 1.0)
    combined = np.maximum.reduce([p_sur, p_xgb, ae_score])
    verdicts, _ = reconcile_batch(p_sur, p_xgb, l_ae, thr)
    configurations = {
        "1. Лише Suricata": (p_sur == 1, p_sur),
        "2. Лише XGBoost": (p_xgb > 0.5, p_xgb),
        "3. Лише автоенкодер": (l_ae > thr.tau_base, l_ae),
        "4. Просте об'єднання (АБО)": ((p_sur == 1) | (p_xgb > 0.5) | (l_ae > thr.tau_base), combined),
        "5. Узгодження (3.1)": (verdicts != Verdict.NORMAL.value, combined),
    }
    rows = {}
    for name, (predicted, score) in configurations.items():
        metrics = binary_metrics(is_attack, predicted, score)
        metrics["alert_share"] = float(np.mean(predicted))
        rows[name] = metrics
    return pd.DataFrame.from_dict(rows, orient="index")
