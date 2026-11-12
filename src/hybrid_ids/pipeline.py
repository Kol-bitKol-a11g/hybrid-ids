"""Конвеєр гібридної системи (рис. 3.2): ознаки потоку -> три детектори ->
модуль узгодження -> оповіщення -> сховище.

Детектори викликаються від найдешевшого до найдорожчого: сигнал Suricata вже
відомий, XGBoost оцінює всі потоки, а автоенкодер - лише ті, для яких рішення
ще не визначено сигнатурою або впевненою оцінкою класифікатора.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .alerting import AlertCorrelator, build_alert, is_alert
from .autoencoder_detector import AutoencoderDetector
from .config import Settings, Thresholds
from .feature_extraction import feature_matrix, normalize_columns, parse_timestamps
from .metrics import RuntimeStats
from .reconciliation import reconcile_batch
from .schemas import AlertRecord
from .xgboost_classifier import XGBoostFlowClassifier

log = logging.getLogger(__name__)
NORMALIZER_FILE = "normalizer.joblib"
THRESHOLDS_FILE = "thresholds.json"


def _column(flows: pd.DataFrame, name: str, default=None) -> list:
    return flows[name].tolist() if name in flows.columns else [default] * len(flows)


def _timestamps(flows: pd.DataFrame) -> list[datetime]:
    """Час початку потоків у UTC; для потоків без позначки - поточний час."""
    now = datetime.now(timezone.utc)
    if "Timestamp" not in flows.columns:
        return [now] * len(flows)
    stamps = parse_timestamps(flows["Timestamp"])
    return [now if pd.isna(stamp) else stamp.to_pydatetime() for stamp in stamps]


class HybridIDS:
    """Гібридна система виявлення вторгнень: Suricata + XGBoost + автоенкодер."""

    def __init__(self, normalizer, classifier, autoencoder, thresholds: Thresholds = Thresholds(),
                 store=None, stats: RuntimeStats | None = None,
                 correlator: AlertCorrelator | None = None) -> None:
        self.normalizer = normalizer      # MinMax-параметри, оцінені на навчальній частині
        self.classifier = classifier      # predict(X) -> (p_xgb, attack_class)
        self.autoencoder = autoencoder    # reconstruction_error(X) -> l_ae
        self.thresholds = thresholds
        self.store = store
        self.stats = stats or RuntimeStats()
        self.correlator = correlator or AlertCorrelator()
        self.stats.detectors.update(xgboost="up", autoencoder="up")

    @classmethod
    def load(cls, settings: Settings, store=None) -> "HybridIDS":
        """Завантажує навчені артефакти. Якщо задано адреси сервісів висновування,
        детектори викликаються через HTTP, інакше - у цьому ж процесі."""
        from .services import RemoteAutoencoder, RemoteClassifier

        models = Path(settings.models_dir)
        thresholds = settings.thresholds
        calibrated = models / THRESHOLDS_FILE
        if calibrated.exists():   # пороги, відкалібровані для цих моделей під час навчання
            thresholds = Thresholds(**json.loads(calibrated.read_text(encoding="utf-8")))
        classifier = (RemoteClassifier(settings.xgb_service_url) if settings.xgb_service_url
                      else XGBoostFlowClassifier.load(models))
        autoencoder = (RemoteAutoencoder(settings.ae_service_url) if settings.ae_service_url
                       else AutoencoderDetector.load(models))
        return cls(joblib.load(models / NORMALIZER_FILE), classifier, autoencoder, thresholds,
                   store, correlator=AlertCorrelator(settings.correlation_window_s))

    def score(self, X_raw, p_sur=None) -> pd.DataFrame:
        """Обчислює сигнали детекторів і рішення за правилом (3.1) для матриці ознак."""
        X = self.normalizer.transform(X_raw)
        n = len(X)
        p_sur = np.zeros(n, dtype=int) if p_sur is None else np.asarray(p_sur, dtype=int)
        if n:
            p_xgb, ml_class = self.classifier.predict(X)
        else:
            p_xgb, ml_class = np.empty(0), np.empty(0, dtype=object)
        l_ae = np.full(n, np.nan)
        pending = (p_sur != 1) & (p_xgb <= self.thresholds.p1)
        if pending.any():
            l_ae[pending] = self.autoencoder.reconstruction_error(X[pending])
        verdict, rule = reconcile_batch(p_sur, p_xgb, l_ae, self.thresholds)
        return pd.DataFrame({"p_sur": p_sur, "p_xgb": p_xgb, "l_ae": l_ae,
                             "verdict": verdict, "rule": rule, "attack_class": ml_class})

    def process(self, flows: pd.DataFrame, suricata=None) -> list[AlertRecord]:
        """Обробляє пакет потоків CICFlowMeter і повертає сформовані оповіщення.

        suricata - індекс оповіщень SuricataIndex; без нього p_sur береться з колонки
        "p_sur" (офлайн-експерименти) або вважається нульовим.
        """
        flows = normalize_columns(flows).reset_index(drop=True)
        started = time.perf_counter()
        X_raw, valid = feature_matrix(flows)
        signatures: list = [None] * len(flows)
        if suricata is not None:
            p_sur, signatures = suricata.signals(flows)
            self.stats.detectors["suricata"] = "up"
        elif "p_sur" in flows.columns:
            p_sur = flows["p_sur"].to_numpy(dtype=int)
        else:
            p_sur = np.zeros(len(flows), dtype=int)
        scores = self.score(X_raw, p_sur)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if not valid.all():   # технічні помилки не зараховуються до норми
            log.warning("Потоків із некоректними ознаками: %d", int((~valid).sum()))
        alerts: list[AlertRecord] = []
        selected = np.flatnonzero(valid & scores["verdict"].map(is_alert).to_numpy())
        if len(selected):
            stamps = _timestamps(flows)
            meta = {key: _column(flows, col) for key, col in (
                ("flow_id", "Flow ID"), ("src_ip", "Source IP"), ("dst_ip", "Destination IP"),
                ("src_port", "Source Port"), ("dst_port", "Destination Port"))}
        thr = self.thresholds
        for i in selected:
            row = scores.iloc[i]
            alert = build_alert(
                {**{key: values[i] for key, values in meta.items()}, "timestamp": stamps[i]},
                row["verdict"], row["rule"], int(row["p_sur"]), float(row["p_xgb"]),
                None if np.isnan(row["l_ae"]) else float(row["l_ae"]),
                str(row["attack_class"]), signatures[i], thr.p1, thr.p2)
            self.correlator.add(alert)
            if self.store is not None:
                self.store.insert(alert)
            alerts.append(alert)

        self.stats.observe(flows=int(valid.sum()), elapsed_ms=elapsed_ms, alerts=len(alerts),
                           autoencoder_calls=int(scores["l_ae"].notna().sum()),
                           technical_errors=int((~valid).sum()))
        if self.store is not None:
            self.store.save_stats(self.stats.snapshot())
        return alerts
