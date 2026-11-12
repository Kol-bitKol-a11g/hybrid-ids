"""Конфігурація прототипу: простір ознак, пороги узгодження, гіперпараметри моделей.

Значення відповідають розділам 2 і 3 роботи (табл. 2.10, 2.12, 2.15, 2.19, 3.2).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 22 потокові ознаки CICFlowMeter, відібрані з 78 (табл. 2.10). Порядок фіксований:
# той самий порядок колонок використовується під час навчання і під час роботи сенсора.
FEATURES: tuple[str, ...] = (
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Min",
    "Bwd Packet Length Std",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Max Packet Length",
    "Avg Bwd Segment Size",
    "Average Packet Size",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Down/Up Ratio",
)

# Службові поля потоку: потрібні для кореляції з подіями Suricata та для оповіщень.
META_COLUMNS = (
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
    "Protocol", "Timestamp",
)
# Поля-витоки (табл. 2.11): ніколи не потрапляють у вектор ознак класифікатора.
LEAKAGE_COLUMNS = (
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Timestamp",
)
LABEL_COLUMN = "Label"
BENIGN = "BENIGN"
RANDOM_STATE = 42

# Random Forest - базова модель порівняння (табл. 2.12).
RF_PARAMS = dict(
    n_estimators=200, max_depth=32, min_samples_leaf=5, max_features="sqrt",
    criterion="gini", class_weight="balanced_subsample", n_jobs=8, random_state=RANDOM_STATE,
)
# XGBoost - основний класифікатор (табл. 2.15). Ваги класів (scale_pos_weight = auto)
# у багатокласовій постановці передаються як sample_weight під час навчання.
XGB_PARAMS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    reg_lambda=1.0, reg_alpha=0.5, min_child_weight=5, objective="multi:softprob",
    eval_metric="mlogloss", tree_method="hist", random_state=RANDOM_STATE,
)
# Автоенкодер - аномалійний компонент (табл. 2.19): 22-128-64-16-[4]-16-64-128-22.
AE_PARAMS = dict(
    encoder=(128, 64, 16), latent=4, dropout=0.2, l2=1e-5,
    learning_rate=1e-3, batch_size=512, epochs=50, patience=5,
)


@dataclass(frozen=True)
class Thresholds:
    """Пороги правила узгодження (3.1), табл. 3.2."""

    p1: float = 0.90        # верхній поріг оцінки XGBoost: впевнена атака
    p2: float = 0.30        # нижній поріг оцінки XGBoost: відсіювання норми
    tau1: float = 0.025     # верхній поріг похибки реконструкції: впевнена аномалія
    tau2: float = 0.010     # нижній поріг похибки реконструкції: підтвердження норми
    tau_base: float = 0.0148  # поріг автоенкодера як окремого детектора (абляція)

    def __post_init__(self) -> None:
        if not 0.0 <= self.p2 < self.p1 <= 1.0:
            raise ValueError("Очікується 0 <= p2 < p1 <= 1")
        if not 0.0 <= self.tau2 < self.tau1:
            raise ValueError("Очікується 0 <= tau2 < tau1")

    def scaled_to(self, tau_base: float) -> "Thresholds":
        """Переносить пороги автоенкодера на іншу модель або мережу: tau1 і tau2
        масштабуються пропорційно до нового базового порога tau_base (евристика
        початкового калібрування; остаточні значення добираються на валідації)."""
        k = tau_base / self.tau_base
        return Thresholds(self.p1, self.p2, self.tau1 * k, self.tau2 * k, tau_base)


@dataclass
class Settings:
    """Параметри запуску прототипу."""

    thresholds: Thresholds = field(default_factory=Thresholds)
    models_dir: str = "models"
    db_url: str = "sqlite:///data/alerts.db"
    xgb_service_url: str | None = None   # http://xgboost-service:8001 у Docker-розгортанні
    ae_service_url: str | None = None    # http://autoencoder-service:8002
    correlation_window_s: int = 300      # вікно групування повторних оповіщень (5 хв)


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    """Читає YAML-конфігурацію; змінні середовища IDS_* мають вищий пріоритет."""
    path = Path(path or os.getenv("IDS_CONFIG", "config/config.yaml"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw = raw or {}
    settings = Settings(thresholds=Thresholds(**raw.pop("thresholds", {})), **raw)
    for attr, env in (
        ("db_url", "IDS_DB_URL"), ("models_dir", "IDS_MODELS_DIR"),
        ("xgb_service_url", "IDS_XGB_URL"), ("ae_service_url", "IDS_AE_URL"),
    ):
        if os.getenv(env):
            setattr(settings, attr, os.environ[env])
    return settings
