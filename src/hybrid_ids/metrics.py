"""Моніторинг: службова статистика конвеєра та її експорт у форматі Prometheus."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RuntimeStats:
    """Накопичує показники роботи конвеєра обробки потоків."""

    started_at: float = field(default_factory=time.time)
    flows_total: int = 0
    alerts_total: int = 0
    autoencoder_calls: int = 0
    technical_errors: int = 0          # потоки з некоректною схемою або значеннями
    latency_ms_sum: float = 0.0
    last_inference_ms: float = 0.0
    queue_size: int = 0
    detectors: dict = field(default_factory=lambda: {"suricata": "down", "xgboost": "down", "autoencoder": "down"})

    def observe(self, flows: int, elapsed_ms: float, alerts: int, autoencoder_calls: int,
                technical_errors: int = 0) -> None:
        """Реєструє оброблений пакет потоків."""
        self.flows_total += flows
        self.alerts_total += alerts
        self.autoencoder_calls += autoencoder_calls
        self.technical_errors += technical_errors
        self.latency_ms_sum += elapsed_ms
        self.last_inference_ms = elapsed_ms / flows if flows else 0.0

    def snapshot(self) -> dict:
        flows = max(self.flows_total, 1)
        return {
            **self.detectors,
            "last_inference_ms": round(self.last_inference_ms, 4),
            "avg_response_ms": round(self.latency_ms_sum / flows, 4),
            "autoencoder_share": round(self.autoencoder_calls / flows, 4),
            "flows_total": self.flows_total,
            "alerts_total": self.alerts_total,
            "technical_errors": self.technical_errors,
            "queue_size": self.queue_size,
            "uptime_s": int(time.time() - self.started_at),
        }


def render_prometheus(snapshot: dict, alerts_last_hour: dict[str, int],
                      verdicts: tuple[int, int] = (0, 0)) -> str:
    """Формує текст метрик у форматі експозиції Prometheus (кінцева точка /metrics)."""
    lines: list[str] = []

    def metric(name: str, kind: str, help_text: str, samples: list[tuple[str, float]]) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        lines.extend(f"{name}{labels} {value}" for labels, value in samples)

    metric("ids_flows_total", "counter", "Processed network flows", [("", snapshot.get("flows_total", 0))])
    metric("ids_alerts_total", "counter", "Generated alerts", [("", snapshot.get("alerts_total", 0))])
    metric("ids_alerts_last_hour", "gauge", "Alerts during the last hour by type",
           [(f'{{alert_type="{kind}"}}', count) for kind, count in sorted(alerts_last_hour.items())]
           or [('{alert_type="none"}', 0)])
    metric("ids_response_time_ms", "gauge", "Average decision time per flow, ms",
           [("", snapshot.get("avg_response_ms", 0.0))])
    metric("ids_autoencoder_share", "gauge", "Share of flows passed to the autoencoder",
           [("", snapshot.get("autoencoder_share", 0.0))])
    metric("ids_technical_errors_total", "counter", "Flows rejected because of schema or value errors",
           [("", snapshot.get("technical_errors", 0))])
    metric("ids_queue_size", "gauge", "Flows waiting for processing", [("", snapshot.get("queue_size", 0))])
    confirmed, refuted = verdicts
    share = refuted / (confirmed + refuted) if confirmed + refuted else 0.0
    metric("ids_false_alert_share", "gauge", "Share of alerts refuted by SOC analysts", [("", round(share, 4))])
    metric("ids_detector_up", "gauge", "Detector availability",
           [(f'{{detector="{name}"}}', int(snapshot.get(name) == "up"))
            for name in ("suricata", "xgboost", "autoencoder")])
    return "\n".join(lines) + "\n"
