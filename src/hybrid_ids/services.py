"""Сервіси контейнерного розгортання (рис. 3.10): сервіси висновування XGBoost
і автоенкодера, їхні HTTP-клієнти та ядро узгодження reconciliation-core.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)


def _post(url: str, payload: dict, timeout: float = 10.0) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class RemoteClassifier:
    """Клієнт сервісу xgboost-service: той самий інтерфейс, що й у локального класифікатора."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        reply = _post(f"{self.url}/predict", {"features": np.asarray(X).tolist()})
        return np.asarray(reply["p_xgb"], dtype=float), np.asarray(reply["attack_class"], dtype=object)


class RemoteAutoencoder:
    """Клієнт сервісу autoencoder-service."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def reconstruction_error(self, X) -> np.ndarray:
        reply = _post(f"{self.url}/score", {"features": np.asarray(X).tolist()})
        return np.asarray(reply["l_ae"], dtype=float)


def create_inference_app(kind: str, models_dir: str):
    """Сервіс висновування: kind = "xgboost" (POST /predict) або "autoencoder" (POST /score).
    Вхід - нормалізовані вектори ознак: {"features": [[...22 значення...], ...]}."""
    from fastapi import Body, FastAPI

    app = FastAPI(title=f"hybrid-ids {kind} service", version="1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "up", "detector": kind}

    if kind == "xgboost":
        from .xgboost_classifier import XGBoostFlowClassifier

        classifier = XGBoostFlowClassifier.load(models_dir)

        @app.post("/predict")
        def predict(features: list[list[float]] = Body(..., embed=True)) -> dict:
            p_xgb, attack_class = classifier.predict(np.asarray(features, dtype=float))
            return {"p_xgb": p_xgb.tolist(), "attack_class": [str(c) for c in attack_class]}
    else:
        from .autoencoder_detector import AutoencoderDetector

        detector = AutoencoderDetector.load(models_dir)

        @app.post("/score")
        def score(features: list[list[float]] = Body(..., embed=True)) -> dict:
            return {"l_ae": detector.reconstruction_error(np.asarray(features, dtype=float)).tolist()}

    return app


def run_worker(settings: Settings, flows_csv: str, eve_json: str | None = None,
               interval_s: float = 2.0, once: bool = False) -> None:
    """Ядро узгодження: стежить за таблицею потоків CICFlowMeter і журналом Suricata,
    обробляє нові потоки конвеєром HybridIDS і зберігає оповіщення у сховищі."""
    import pandas as pd

    from .pipeline import HybridIDS
    from .storage import AlertStore
    from .suricata_runner import SuricataIndex, parse_eve

    ids = HybridIDS.load(settings, AlertStore(settings.db_url))
    index = SuricataIndex() if eve_json else None
    rows_done, eve_offset = 0, 0
    while True:
        if index is not None and Path(eve_json).exists():
            with open(eve_json, encoding="utf-8", errors="replace") as handle:
                handle.seek(eve_offset)
                for alert in parse_eve(handle.readlines()):
                    index.add(alert)
                eve_offset = handle.tell()
        if Path(flows_csv).exists():
            try:
                flows = pd.read_csv(flows_csv, skiprows=range(1, rows_done + 1), low_memory=False)
            except pd.errors.EmptyDataError:
                flows = pd.DataFrame()
            if len(flows):
                ids.stats.queue_size = len(flows)
                alerts = ids.process(flows, index)
                ids.stats.queue_size = 0
                rows_done += len(flows)
                log.info("Оброблено потоків: %d, оповіщень: %d", len(flows), len(alerts))
        if once:
            return
        time.sleep(interval_s)
