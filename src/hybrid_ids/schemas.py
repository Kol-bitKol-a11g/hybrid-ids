"""Типізовані структури даних, якими обмінюються модулі прототипу."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum


class Verdict(str, Enum):
    """Фінальне рішення модуля узгодження."""

    ATTACK = "attack"
    NORMAL = "normal"
    REVIEW = "review"   # потік передається аналітику SOC на перевірку


@dataclass(frozen=True)
class Decision:
    """Рішення та гілка правила (3.1), яка його сформувала."""

    verdict: Verdict
    rule: str


@dataclass
class AlertRecord:
    """Оповіщення; поля від flow_id до id відповідають колонкам таблиці alert_records."""

    flow_id: str
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None
    alert_type: str              # signature | ml_confident | reconciled | review
    detector: str                # suricata | suricata+xgboost | xgboost | xgboost+autoencoder
    attack_class: str | None
    mitre_technique: str | None
    p_xgb: float | None
    l_ae: float | None
    severity: int = 1
    verified: bool = False
    id: int | None = None
    decision: str = Verdict.ATTACK.value
    rule: str = ""
    signature: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
