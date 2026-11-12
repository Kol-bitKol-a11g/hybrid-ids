"""Тести модуля узгодження: кожна гілка правила (3.1) і матриці рішень (табл. 3.3)."""
import numpy as np
import pytest

from hybrid_ids.config import Thresholds
from hybrid_ids.reconciliation import (RULE_ML_CONFIDENT, RULE_NORMAL, RULE_RECONCILED,
                                       RULE_REVIEW_ANOMALY, RULE_REVIEW_UNCERTAIN, RULE_SIGNATURE,
                                       needs_autoencoder, reconcile, reconcile_batch)
from hybrid_ids.schemas import Verdict

THR = Thresholds()   # p1 = 0,90; p2 = 0,30; tau1 = 0,025; tau2 = 0,010

# (p_sur, p_xgb, l_ae, очікуване рішення, гілка правила)
CASES = [
    (1, 0.05, 0.001, Verdict.ATTACK, RULE_SIGNATURE),          # пріоритет сигнатури над ML
    (1, 0.99, None, Verdict.ATTACK, RULE_SIGNATURE),
    (0, 0.9963, None, Verdict.ATTACK, RULE_ML_CONFIDENT),      # SYN-flood, підрозділ 3.4.1
    (0, 0.67, 0.0341, Verdict.ATTACK, RULE_RECONCILED),        # обфускована SQL-ін'єкція
    (0, 0.67, 0.020, Verdict.REVIEW, RULE_REVIEW_UNCERTAIN),
    (0, 0.10, 0.005, Verdict.NORMAL, RULE_NORMAL),             # подвійне підтвердження норми
    (0, 0.10, 0.015, Verdict.REVIEW, RULE_REVIEW_ANOMALY),
    (0, 0.10, 0.040, Verdict.REVIEW, RULE_REVIEW_ANOMALY),
    # значення на межах порогів
    (0, 0.90, 0.030, Verdict.ATTACK, RULE_RECONCILED),         # p_xgb = p1: ще зона невизначеності
    (0, 0.30, 0.030, Verdict.ATTACK, RULE_RECONCILED),         # p_xgb = p2: уже зона невизначеності
    (0, 0.30, 0.025, Verdict.REVIEW, RULE_REVIEW_UNCERTAIN),   # l_ae = tau1 не перевищує порога
    (0, 0.29, 0.010, Verdict.REVIEW, RULE_REVIEW_ANOMALY),     # l_ae = tau2 не підтверджує норму
]


@pytest.mark.parametrize("p_sur, p_xgb, l_ae, verdict, rule", CASES)
def test_rule_branches(p_sur, p_xgb, l_ae, verdict, rule):
    decision = reconcile(p_sur, p_xgb, l_ae, THR)
    assert (decision.verdict, decision.rule) == (verdict, rule)


def test_batch_matches_scalar_rule():
    p_sur, p_xgb, l_ae = (np.array([c[i] for c in CASES], dtype=float) for i in range(3))
    verdicts, rules = reconcile_batch(p_sur, p_xgb, l_ae, THR)
    assert list(verdicts) == [c[3].value for c in CASES]
    assert list(rules) == [c[4] for c in CASES]


def test_autoencoder_is_skipped_when_decision_is_known():
    assert not needs_autoencoder(1, 0.10, THR)      # рішення визначила сигнатура
    assert not needs_autoencoder(0, 0.95, THR)      # впевнена оцінка класифікатора
    assert needs_autoencoder(0, 0.50, THR)
    assert needs_autoencoder(0, 0.10, THR)          # норма потребує другого підтвердження
    with pytest.raises(ValueError):
        reconcile(0, 0.50, None, THR)


def test_thresholds_validation_and_scaling():
    with pytest.raises(ValueError):
        Thresholds(p1=0.3, p2=0.9)
    with pytest.raises(ValueError):
        Thresholds(tau1=0.01, tau2=0.02)
    scaled = Thresholds().scaled_to(0.0296)         # удвічі більший базовий поріг
    assert scaled.tau1 == pytest.approx(0.050) and scaled.tau2 == pytest.approx(0.020)
    assert (scaled.p1, scaled.p2) == (0.90, 0.30)
