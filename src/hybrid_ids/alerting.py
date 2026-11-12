"""Підсистема оповіщень: формування запису AlertRecord, зіставлення з техніками
MITRE ATT&CK, рівень критичності та групування повторних спрацювань.

Оповіщення містить оцінку моделі та спостережувані ознаки потоку і не підмінює
їх твердженнями про вміст зашифрованого сеансу.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from .reconciliation import RULE_ML_CONFIDENT, RULE_RECONCILED, RULE_SIGNATURE
from .schemas import AlertRecord, Verdict

# Ключове слово в назві класу -> (техніка MITRE ATT&CK, рівень критичності 1..4).
# Порядок важливий: "ddos" перевіряється раніше за "dos".
ATTACK_PROFILES = (
    ("heartbleed", "T1040,T1210", 4),   # Network Sniffing; Exploitation of Remote Services
    ("sql", "T1190", 3),                # Exploit Public-Facing Application
    ("xss", "T1059.007", 3),            # Command and Scripting Interpreter: JavaScript
    ("patator", "T1110", 3),            # Brute Force
    ("brute", "T1110", 3),
    ("portscan", "T1046", 2),           # Network Service Discovery
    ("scan", "T1046", 2),
    ("ddos", "T1498", 4),               # Network Denial of Service
    ("dos", "T1499", 4),                # Endpoint Denial of Service
    ("bot", "T1071", 3),                # Application Layer Protocol (канал керування)
)
# Ключове слово в назві сигнатури Suricata -> клас атаки (коли ML вважає потік нормою).
SIGNATURE_CLASSES = (
    ("SCAN", "PortScan"), ("DDOS", "DDoS"), ("FLOOD", "DDoS"), ("DOS", "DoS"),
    ("SQL", "SQLInjection"), ("XSS", "XSS"), ("BRUTE", "BruteForce"),
)


def attack_profile(attack_class: str | None) -> tuple[str | None, int]:
    """Техніка MITRE ATT&CK і рівень критичності для класу атаки."""
    name = (attack_class or "").lower().replace(" ", "").replace("-", "")
    for keyword, technique, severity in ATTACK_PROFILES:
        if keyword in name:
            return technique, severity
    return None, 1


def class_from_signature(signature: str | None) -> str:
    text = (signature or "").upper()
    return next((cls for keyword, cls in SIGNATURE_CLASSES if keyword in text), "SignatureMatch")


def flow_uuid(raw_id, timestamp: datetime) -> str:
    """Стабільний UUID потоку: повторні повідомлення про той самий потік CICFlowMeter
    отримують той самий ідентифікатор і не завищують кількість інцидентів."""
    if raw_id is None or str(raw_id) in ("", "nan"):
        return str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{raw_id}|{timestamp.isoformat()}"))


def build_alert(meta: dict, verdict: str, rule: str, p_sur: int, p_xgb: float, l_ae: float | None,
                ml_class: str | None, signature: str | None, p1: float, p2: float) -> AlertRecord:
    """Формує оповіщення для потоку з рішенням «атака» або «перевірка».

    meta - службові поля потоку: flow_id, timestamp, src_ip, dst_ip, src_port, dst_port.
    """
    if rule == RULE_SIGNATURE:
        alert_type = "signature"
        confident = p_xgb > p1
        detector = "suricata+xgboost" if confident else "suricata"
        attack_class = ml_class if p_xgb >= p2 else class_from_signature(signature)
    elif rule == RULE_ML_CONFIDENT:
        alert_type, detector, attack_class = "ml_confident", "xgboost", ml_class
    elif rule == RULE_RECONCILED:
        alert_type, detector, attack_class = "reconciled", "xgboost+autoencoder", ml_class
    else:   # сумнівний потік або аномалія за низької оцінки XGBoost -> аналітику SOC
        alert_type, detector, attack_class = "review", "xgboost+autoencoder", "Anomaly"
    technique, severity = attack_profile(attack_class)
    timestamp = meta.get("timestamp") or datetime.now(timezone.utc)
    return AlertRecord(
        flow_id=flow_uuid(meta.get("flow_id"), timestamp),
        timestamp=timestamp,
        src_ip=str(meta.get("src_ip") or "0.0.0.0"), dst_ip=str(meta.get("dst_ip") or "0.0.0.0"),
        src_port=meta.get("src_port"), dst_port=meta.get("dst_port"),
        alert_type=alert_type, detector=detector, attack_class=attack_class,
        mitre_technique=technique, p_xgb=None if p_xgb is None else round(float(p_xgb), 4),
        l_ae=None if l_ae is None else round(float(l_ae), 5), severity=severity,
        decision=verdict, rule=rule, signature=signature,
    )


class AlertCorrelator:
    """Кореляційне групування: повторні оповіщення однієї атаки між тими самими
    вузлами в межах вікна (типово 5 хв) рахуються як один інцидент."""

    def __init__(self, window_s: int = 300) -> None:
        self.window = timedelta(seconds=window_s)
        self._last_seen: dict[tuple, datetime] = {}
        self.alerts = 0
        self.groups = 0

    def add(self, alert: AlertRecord) -> bool:
        """Реєструє оповіщення; True - якщо воно відкриває нову групу (інцидент)."""
        key = (alert.attack_class, alert.src_ip, alert.dst_ip)
        last = self._last_seen.get(key)
        is_new = last is None or alert.timestamp - last > self.window
        self._last_seen[key] = alert.timestamp
        self.alerts += 1
        self.groups += int(is_new)
        return is_new


def is_alert(verdict: str) -> bool:
    """Оповіщення формується для рішень «атака» та «перевірка»."""
    return verdict in (Verdict.ATTACK.value, Verdict.REVIEW.value)
