"""Підготовка даних без витоку інформації (рис. 1.8, підрозділ 2.2).

Порядок кроків принциповий: очищення -> розбиття train/val/test -> оцінювання
параметрів нормалізації лише на train -> балансування лише train.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from .config import FEATURES, LABEL_COLUMN, RANDOM_STATE
from .feature_extraction import parse_timestamps

# Розміри вікна TCP у CICFlowMeter можуть дорівнювати -1 (вікно не оголошено),
# тому для них перевірка на від'ємні значення не застосовується.
NON_NEGATIVE = tuple(f for f in FEATURES if not f.startswith("Init_Win"))


def normalize_label(label: str) -> str:
    """Уніфікує мітку класу: у CSV-файлах CICIDS2017 тире в назвах веб-атак
    закодоване непослідовно, тому всі не-ASCII символи замінюються дефісом."""
    text = re.sub(r"[^\x20-\x7E]+", "-", str(label)).strip()
    return re.sub(r"\s+", " ", text)


def clean_flows(df: pd.DataFrame, label_col: str = LABEL_COLUMN,
                min_class_size: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Детерміноване очищення потоків (табл. 2.4).

    Повертає очищений набір і звіт: етап, рядків до, рядків після, видалено.
    Викиди не видаляються: велике значення може бути змістовною ознакою атаки.
    """
    report: list[tuple[str, int, int]] = []

    def step(name: str, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
        report.append((name, len(before), len(after)))
        return after

    feats = list(FEATURES)
    cur = step("Видалення дублікатів", df, df.drop_duplicates())

    numeric = cur[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    cur = cur.assign(**{f: numeric[f] for f in feats})
    cur = step("Обробка NaN/Inf", cur, cur.dropna(subset=feats))

    packets = cur["Total Fwd Packets"] + cur["Total Backward Packets"]
    invalid = (cur[list(NON_NEGATIVE)] < 0).any(axis=1) | (packets <= 0)
    cur = step("Видалення некоректних значень", cur, cur[~invalid])

    if label_col in cur.columns:
        cur = cur.assign(**{label_col: cur[label_col].map(normalize_label)})
        counts = cur[label_col].value_counts()
        tiny = counts[counts < min_class_size].index
        cur = step("Видалення надто малих класів", cur, cur[~cur[label_col].isin(tiny)])

    table = pd.DataFrame(report, columns=["Етап", "До", "Після"])
    table["Видалено"] = table["До"] - table["Після"]
    return cur.reset_index(drop=True), table


def time_split(df: pd.DataFrame, time_col: str = "Timestamp", train: float = 0.70,
               val: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Хронологічний поділ 70/15/15 (табл. 2.7): навчання на раніших потоках,
    перевірка на пізніших, щоб пов'язані потоки одного сценарію не розносилися
    між частинами."""
    stamps = parse_timestamps(df[time_col])
    ordered = df.assign(**{time_col: stamps}).sort_values(time_col, kind="stable")
    return ordered_split(ordered.reset_index(drop=True), train, val)


def ordered_split(df: pd.DataFrame, train: float = 0.70,
                  val: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ділить уже впорядковану таблицю на послідовні частини train/val/test."""
    i, j = int(len(df) * train), int(len(df) * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def group_split(df: pd.DataFrame, group_col: str, train: float = 0.70, val: float = 0.15,
                random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Груповий поділ для наборів без часових позначок (NSL-KDD, UNSW-NB15):
    усі записи однієї групи потрапляють в одну частину."""
    from sklearn.model_selection import GroupShuffleSplit

    first = GroupShuffleSplit(n_splits=1, train_size=train, random_state=random_state)
    train_idx, rest_idx = next(first.split(df, groups=df[group_col]))
    rest = df.iloc[rest_idx]
    second = GroupShuffleSplit(n_splits=1, train_size=val / (1.0 - train), random_state=random_state)
    val_idx, test_idx = next(second.split(rest, groups=rest[group_col]))
    return df.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]


class MinMaxNormalizer:
    """MinMax-нормалізація за формулою (2.1).

    Мінімум і максимум кожної ознаки оцінюються тільки на навчальній частині;
    validation, test і робочий трафік лише перетворюються цими параметрами.
    """

    def __init__(self, clip: bool = True) -> None:
        self.clip = clip          # значення поза діапазоном train обрізаються до [0, 1]
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(self, X_train) -> "MinMaxNormalizer":
        X = np.asarray(X_train, dtype=float)
        self.min_ = X.min(axis=0)
        span = X.max(axis=0) - self.min_
        self.range_ = np.where(span > 0, span, 1.0)   # сталі ознаки не ділимо на нуль
        return self

    def transform(self, X) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Нормалізатор не навчено: спершу викличте fit() на train")
        scaled = (np.asarray(X, dtype=float) - self.min_) / self.range_
        return np.clip(scaled, 0.0, 1.0) if self.clip else scaled


def smote_targets(y, minority_share: float = 0.01, tiny_share: float = 0.001,
                  multiplier: int = 10) -> dict:
    """Цільові розміри класів для обмеженого класо-залежного балансування (табл. 2.6).

    Класи з часткою < 1 % діляться на дві групи: малочисельні (0,1-1 %) і вкрай
    малочисельні (< 0,1 %). Кожна група дорівнюється до multiplier * (найбільший клас
    групи). Нормальний трафік і численні класи атак не змінюються: повне вирівнювання
    потребувало б надмірної кількості синтетичних прикладів.
    """
    counts = pd.Series(list(y)).value_counts()
    share = counts / counts.sum()
    groups = (counts[(share < minority_share) & (share >= tiny_share)], counts[share < tiny_share])
    targets: dict = {}
    for group in groups:
        if len(group):
            target = int(group.max() * multiplier)
            targets.update({cls: target for cls, n in group.items() if n < target})
    return targets


def balance_train(X, y, k_neighbors: int = 5, random_state: int = RANDOM_STATE, **kwargs):
    """SMOTE для малочисельних класів навчальної частини.

    До validation і test не застосовується: там зберігається природний розподіл.
    Клас з одним прикладом не синтезується (немає сусідів) - для нього лишаються ваги.
    """
    from imblearn.over_sampling import SMOTE

    counts = Counter(y)
    targets = {c: t for c, t in smote_targets(y, **kwargs).items() if counts[c] >= 2}
    if not targets:
        return X, y
    k = min(k_neighbors, min(counts[c] for c in targets) - 1)
    sampler = SMOTE(sampling_strategy=targets, k_neighbors=k, random_state=random_state)
    return sampler.fit_resample(X, y)


def class_weights(y) -> np.ndarray:
    """Ваги прикладів, обернено пропорційні частоті класу (друга складова балансування)."""
    from sklearn.utils.class_weight import compute_sample_weight

    return compute_sample_weight("balanced", y)
