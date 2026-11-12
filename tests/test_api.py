"""Тести REST API (виконуються, якщо встановлено fastapi та httpx)."""
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from hybrid_ids.alerting import build_alert  # noqa: E402
from hybrid_ids.api import create_app  # noqa: E402
from hybrid_ids.reconciliation import RULE_ML_CONFIDENT  # noqa: E402
from hybrid_ids.storage import AlertStore  # noqa: E402


@pytest.fixture()
def client():
    store = AlertStore("sqlite:///")
    meta = {"flow_id": "f-1", "timestamp": datetime.now(timezone.utc), "src_ip": "10.0.0.5",
            "dst_ip": "10.0.0.9", "src_port": 40000, "dst_port": 80}
    alert = build_alert(meta, "attack", RULE_ML_CONFIDENT, 0, 0.9963, None, "DDoS", None, 0.90, 0.30)
    store.insert(alert)
    store.save_stats({"suricata": "up", "xgboost": "up", "autoencoder": "up", "last_inference_ms": 0.46})
    return TestClient(create_app(store)), alert


def test_alerts_health_metrics_verify(client):
    api, alert = client
    rows = api.get("/alerts", params={"attack_class": "DDoS", "min_confidence": 0.9}).json()
    assert len(rows) == 1 and rows[0]["mitre_technique"] == "T1498"
    assert api.get("/alerts", params={"attack_class": "PortScan"}).json() == []
    health = api.get("/health").json()
    assert health["xgboost"] == "up" and health["last_inference_ms"] == 0.46
    assert 'ids_alerts_last_hour{alert_type="ml_confident"} 1' in api.get("/metrics").text
    assert api.post(f"/verify/{alert.flow_id}", json={"confirmed": True}).json()["verified"] is True
    assert api.post("/verify/unknown", json={"confirmed": True}).status_code == 404
