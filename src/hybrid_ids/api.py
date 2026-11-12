"""REST API прототипу для інтеграції із засобами SOC (FastAPI).

Кінцеві точки: GET /alerts, GET /health, GET /metrics, POST /verify/{flow_id}.
Інтерактивна документація Swagger UI доступна за адресою /docs.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .metrics import render_prometheus
from .storage import AlertStore


class AnalystVerdict(BaseModel):
    """Результат ручної перевірки оповіщення аналітиком SOC."""

    confirmed: bool                 # True - атаку підтверджено, False - хибне оповіщення
    comment: Optional[str] = None


def create_app(store: AlertStore) -> FastAPI:
    """Створює застосунок REST API поверх сховища оповіщень."""
    app = FastAPI(title="hybrid-ids API", version="1.0",
                  description="Гібридна система виявлення вторгнень: Suricata + XGBoost + автоенкодер")

    @app.get("/alerts", tags=["alerts"], summary="Список оповіщень")
    def alerts(time_from: Optional[datetime] = None, time_to: Optional[datetime] = None,
               attack_class: Optional[str] = None,
               min_confidence: float = Query(0.0, ge=0.0, le=1.0),
               limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
        return store.query(time_from, time_to, attack_class, min_confidence, limit)

    @app.get("/health", tags=["health"], summary="Статус компонентів системи")
    def health() -> dict:
        snapshot = store.load_stats()
        keys = ("suricata", "xgboost", "autoencoder", "last_inference_ms", "queue_size", "uptime_s")
        defaults = {"suricata": "down", "xgboost": "down", "autoencoder": "down",
                    "last_inference_ms": 0.0, "queue_size": 0, "uptime_s": 0}
        return {key: snapshot.get(key, defaults[key]) for key in keys}

    @app.get("/metrics", tags=["metrics"], summary="Метрики у форматі Prometheus",
             response_class=PlainTextResponse)
    def metrics() -> str:
        since = datetime.now(timezone.utc) - timedelta(hours=1)
        return render_prometheus(store.load_stats(), store.alerts_by_type(since), store.verdict_counts())

    @app.post("/verify/{flow_id}", tags=["verify"],
              summary="Підтвердження або спростування оповіщення аналітиком")
    def verify(flow_id: str, verdict: AnalystVerdict) -> dict:
        if not store.verify(flow_id, verdict.confirmed, verdict.comment):
            raise HTTPException(status_code=404, detail="Оповіщення з таким flow_id не знайдено")
        return {"flow_id": flow_id, "verified": True, "confirmed": verdict.confirmed}

    return app
