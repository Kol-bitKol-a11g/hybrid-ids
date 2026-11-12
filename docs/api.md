# REST API

Запуск: `python -m hybrid_ids serve --port 8080`. Інтерактивна документація Swagger UI: `http://localhost:8080/docs`.

## GET /alerts

Повертає оповіщення, новіші спершу.

| Параметр | Тип | Опис |
|---|---|---|
| `time_from`, `time_to` | ISO 8601 | часовий діапазон |
| `attack_class` | рядок | клас атаки, наприклад `DDoS` |
| `min_confidence` | 0–1 | мінімальна оцінка `p_xgb` (сигнатурні оповіщення без оцінки мають 1) |
| `limit` | 1–1000 | кількість записів, типово 100 |

```bash
curl "http://localhost:8080/alerts?attack_class=DDoS&min_confidence=0.9&limit=10"
```

## GET /health

```json
{"suricata": "up", "xgboost": "up", "autoencoder": "up",
 "last_inference_ms": 0.46, "queue_size": 0, "uptime_s": 3600}
```

## GET /metrics

Метрики у форматі Prometheus: `ids_flows_total`, `ids_alerts_total`, `ids_alerts_last_hour{alert_type}`, `ids_response_time_ms`, `ids_autoencoder_share`, `ids_technical_errors_total`, `ids_queue_size`, `ids_false_alert_share`, `ids_detector_up{detector}`.

## POST /verify/{flow_id}

Фіксує вердикт аналітика SOC.

```bash
curl -X POST http://localhost:8080/verify/<flow_id> \
     -H "Content-Type: application/json" \
     -d '{"confirmed": false, "comment": "легітимне резервне копіювання"}'
```

Відповідь `404`, якщо оповіщення з таким `flow_id` немає.

## Сервіси висновування

У контейнерному розгортанні детектори доступні як окремі сервіси; вхід - нормалізовані вектори з 22 ознак.

| Сервіс | Запит | Відповідь |
|---|---|---|
| `xgboost-service:8001` | `POST /predict {"features": [[...]]}` | `{"p_xgb": [...], "attack_class": [...]}` |
| `autoencoder-service:8002` | `POST /score {"features": [[...]]}` | `{"l_ae": [...]}` |
