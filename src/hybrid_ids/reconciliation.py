"""Модуль узгодження: правило (3.1) розв'язання суперечностей між детекторами.

Входи: p_sur - бінарний вихід Suricata, p_xgb - оцінка ймовірності атаки XGBoost,
l_ae - похибка реконструкції автоенкодера. Перевірки виконуються від найдешевшої
до найдорожчої (табл. 3.4), складність - O(1) на потік.
"""
from __future__ import annotations

import numpy as np

from .config import Thresholds
from .schemas import Decision, Verdict

RULE_SIGNATURE = "signature"                 # p_sur = 1
RULE_ML_CONFIDENT = "ml_confident"           # p_xgb > p1
RULE_RECONCILED = "reconciled"               # p2 <= p_xgb <= p1 і l_ae > tau1
RULE_NORMAL = "double_confirmation"          # p_xgb < p2 і l_ae < tau2
RULE_REVIEW_UNCERTAIN = "uncertain_low_lae"  # p2 <= p_xgb <= p1 і l_ae <= tau1
RULE_REVIEW_ANOMALY = "anomaly_low_pxgb"     # p_xgb < p2 і l_ae >= tau2


def needs_autoencoder(p_sur: int, p_xgb: float, thr: Thresholds) -> bool:
    """Чи потрібна похибка реконструкції для прийняття рішення.

    Автоенкодер не викликається, якщо рішення вже визначено сигнатурою
    або впевненою оцінкою XGBoost.
    """
    return p_sur != 1 and p_xgb <= thr.p1


def reconcile(p_sur: int, p_xgb: float, l_ae: float | None, thr: Thresholds = Thresholds()) -> Decision:
    """Застосовує правило (3.1) до одного потоку."""
    if p_sur == 1:                                   # крок 2: пріоритет сигнатури
        return Decision(Verdict.ATTACK, RULE_SIGNATURE)
    if p_xgb > thr.p1:                               # крок 3: впевнена класифікація
        return Decision(Verdict.ATTACK, RULE_ML_CONFIDENT)
    if l_ae is None:
        raise ValueError("Для p_xgb <= p1 потрібна похибка реконструкції l_ae")
    if p_xgb < thr.p2:                               # кроки 4-6: кандидат у норму
        if l_ae < thr.tau2:
            return Decision(Verdict.NORMAL, RULE_NORMAL)
        return Decision(Verdict.REVIEW, RULE_REVIEW_ANOMALY)
    if l_ae > thr.tau1:                              # кроки 7-8: зона невизначеності
        return Decision(Verdict.ATTACK, RULE_RECONCILED)
    return Decision(Verdict.REVIEW, RULE_REVIEW_UNCERTAIN)


def reconcile_batch(p_sur, p_xgb, l_ae, thr: Thresholds = Thresholds()) -> tuple[np.ndarray, np.ndarray]:
    """Векторизований варіант правила (3.1); повертає масиви рішень і гілок правила.

    Для потоків, де автоенкодер не викликався, l_ae може бути NaN.
    """
    p_sur, p_xgb, l_ae = (np.asarray(a, dtype=float) for a in (p_sur, p_xgb, l_ae))
    low, sig, sure = p_xgb < thr.p2, p_sur == 1, p_xgb > thr.p1
    conditions = [
        sig,
        sure,
        low & (l_ae < thr.tau2),
        low,
        l_ae > thr.tau1,
    ]
    rules = np.select(
        conditions,
        [RULE_SIGNATURE, RULE_ML_CONFIDENT, RULE_NORMAL, RULE_REVIEW_ANOMALY, RULE_RECONCILED],
        default=RULE_REVIEW_UNCERTAIN,
    )
    attack, normal, review = (v.value for v in (Verdict.ATTACK, Verdict.NORMAL, Verdict.REVIEW))
    verdicts = np.select(conditions, [attack, attack, normal, review, attack], default=review)
    return verdicts, rules
