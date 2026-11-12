"""Сигнатурний детектор: розбір журналу Suricata EVE JSON і формування сигналу p_sur.

Події Suricata та потоки CICFlowMeter - результати різних інструментів, тому вони
зводяться за адресами, портами, протоколом і часовим інтервалом (підрозділ 1.4).
Ідентифікатор потоку Suricata не вважається тотожним ідентифікатору CICFlowMeter.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .feature_extraction import parse_timestamps

PROTOCOLS = {"TCP": 6, "UDP": 17, "ICMP": 1}


@dataclass(frozen=True)
class SuricataAlert:
    """Подія event_type = alert із журналу EVE JSON."""

    timestamp: datetime
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: int
    signature: str
    signature_id: int
    category: str
    severity: int


def _parse_time(value: str) -> datetime:
    """Час EVE JSON має вигляд 2025-11-14T10:23:41.123456+0200."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value)


def _as_utc(value: datetime) -> datetime:
    """Час без зони (CICFlowMeter) трактується як UTC."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _key(ip_a: str, port_a: int, ip_b: str, port_b: int, proto: int) -> tuple:
    """Двонапрямний ключ з'єднання: прямий і зворотний напрям дають один ключ."""
    return (proto, *sorted([(str(ip_a), int(port_a)), (str(ip_b), int(port_b))]))


def parse_eve(lines: Iterable[str]) -> list[SuricataAlert]:
    """Вибирає з рядків EVE JSON оповіщення; пошкоджені рядки пропускаються."""
    alerts = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("event_type") != "alert":
            continue
        info = event.get("alert", {})
        proto = str(event.get("proto", "")).upper()
        alerts.append(SuricataAlert(
            timestamp=_as_utc(_parse_time(event["timestamp"])),
            src_ip=event.get("src_ip", ""), src_port=int(event.get("src_port", 0) or 0),
            dst_ip=event.get("dest_ip", ""), dst_port=int(event.get("dest_port", 0) or 0),
            proto=PROTOCOLS.get(proto, int(proto) if proto.isdigit() else 0),
            signature=info.get("signature", ""), signature_id=int(info.get("signature_id", 0) or 0),
            category=info.get("category", ""), severity=int(info.get("severity", 3) or 3),
        ))
    return alerts


def read_eve(path) -> list[SuricataAlert]:
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        return parse_eve(handle)


class SuricataIndex:
    """Індекс оповіщень Suricata для зіставлення з потоками CICFlowMeter."""

    def __init__(self, alerts: Iterable[SuricataAlert] = (), tolerance_s: float = 5.0) -> None:
        self.tolerance = timedelta(seconds=tolerance_s)
        self._by_key: dict[tuple, list[SuricataAlert]] = defaultdict(list)
        for alert in alerts:
            self.add(alert)

    def add(self, alert: SuricataAlert) -> None:
        self._by_key[_key(alert.src_ip, alert.src_port, alert.dst_ip, alert.dst_port, alert.proto)].append(alert)

    def match(self, src_ip, src_port, dst_ip, dst_port, proto, start: datetime,
              end: datetime | None = None) -> SuricataAlert | None:
        """Шукає оповіщення для з'єднання в інтервалі життя потоку (з допуском)."""
        start, end = _as_utc(start), _as_utc(end or start)
        for alert in self._by_key.get(_key(src_ip, src_port, dst_ip, dst_port, int(proto)), ()):
            if start - self.tolerance <= alert.timestamp <= end + self.tolerance:
                return alert
        return None

    def signals(self, flows: pd.DataFrame) -> tuple[np.ndarray, list[str | None]]:
        """Обчислює p_sur і назву спрацьованої сигнатури для кожного потоку."""
        p_sur = np.zeros(len(flows), dtype=int)
        signatures: list[str | None] = [None] * len(flows)
        if not self._by_key or flows.empty:
            return p_sur, signatures
        starts = parse_timestamps(flows["Timestamp"])
        # Flow Duration у CICFlowMeter подається в мікросекундах.
        ends = starts + pd.to_timedelta(pd.to_numeric(flows["Flow Duration"], errors="coerce").fillna(0), unit="us")
        columns = ("Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol")
        for i, row in enumerate(flows[list(columns)].itertuples(index=False)):
            if pd.isna(starts.iloc[i]):
                continue
            hit = self.match(*row, starts.iloc[i].to_pydatetime(), ends.iloc[i].to_pydatetime())
            if hit is not None:
                p_sur[i], signatures[i] = 1, hit.signature
        return p_sur, signatures
