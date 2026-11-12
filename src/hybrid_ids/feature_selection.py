"""Відбір інформативних ознак трьома методами (підрозділ 2.3, рис. 2.5).

Ознака лишається, якщо вона пройшла кореляційний фільтр і входить до top-k
як за важливістю Random Forest (критерій Джині), так і за взаємною інформацією.
Усі оцінки виконуються лише на навчальній частині.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, RF_PARAMS


def correlation_filter(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Кореляційний фільтр Пірсона: з кожної пари ознак із |r| > threshold
    відкидається ознака з меншою дисперсією. X має бути нормалізованим."""
    corr, var = X.corr().abs(), X.var()
    cols, dropped = list(X.columns), set()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in dropped:
                break
            if b not in dropped and corr.loc[a, b] > threshold:
                dropped.add(a if var[a] < var[b] else b)
    return [c for c in cols if c not in dropped]


def gini_importance(X: pd.DataFrame, y, params: dict | None = None) -> pd.Series:
    """Важливість ознак за зменшенням неоднорідності Джині, формула (2.2)."""
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(**(params or RF_PARAMS)).fit(X, y)
    return pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)


def mutual_information(X: pd.DataFrame, y, random_state: int = RANDOM_STATE) -> pd.Series:
    """Взаємна інформація між ознакою та міткою в бітах, формула (2.3)."""
    from sklearn.feature_selection import mutual_info_classif

    nats = mutual_info_classif(X, y, random_state=random_state)
    return pd.Series(nats / np.log(2), index=X.columns).sort_values(ascending=False)


def select_features(X: pd.DataFrame, y, top_k: int = 20, corr_threshold: float = 0.95,
                    rf_params: dict | None = None) -> tuple[list[str], pd.DataFrame]:
    """Об'єднує результати трьох методів.

    Повертає список відібраних ознак і таблицю з рангами та статусом кожної ознаки
    (аналог табл. 2.9).
    """
    passed = set(correlation_filter(X, corr_threshold))
    gini = gini_importance(X, y, rf_params)
    mi = mutual_information(X, y)
    table = pd.DataFrame({
        "gini_importance": gini,
        "rf_rank": gini.rank(ascending=False, method="first").astype(int),
        "mutual_info_bits": mi,
        "mi_rank": mi.rank(ascending=False, method="first").astype(int),
    })
    table["corr_filter"] = [name in passed for name in table.index]
    table["selected"] = (table["rf_rank"] <= top_k) & (table["mi_rank"] <= top_k) & table["corr_filter"]
    table = table.sort_values("rf_rank")
    return list(table.index[table["selected"]]), table
