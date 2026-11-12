"""Генератор синтетичних потоків для демонстрації та тестів.

Дає змогу запустити весь конвеєр без завантаження наборів даних. Синтетичні потоки
лише імітують структуру таблиці CICFlowMeter і НЕ відтворюють результати роботи:
для цього потрібні CICIDS2017, NSL-KDD або UNSW-NB15.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BENIGN, RANDOM_STATE

# Клас: (частка, тривалість мкс, прямих пакетів, зворотних пакетів, середній розмір
# прямого пакета, середній розмір зворотного пакета, вікно TCP вперед, вікно TCP назад).
PROFILES = {
    BENIGN: (0.800, 2.0e6, 12, 14, 180, 520, 29200, 28960),
    "DoS Hulk": (0.080, 8.0e7, 7, 5, 60, 1900, 251, 235),
    "PortScan": (0.055, 6.0e1, 1, 1, 2, 6, 1024, 0),
    "DDoS": (0.045, 1.5e6, 4, 3, 8, 1400, 256, 229),
    "SSH-Patator": (0.015, 6.0e6, 22, 32, 90, 120, 29200, 247),
    "Web Attack - Sql Injection": (0.005, 5.0e6, 5, 4, 320, 610, 29200, 252),
}
# Демонстраційні сигнатури (не правила ET Open) і частка потоків класу, на які вони спрацьовують.
DEMO_SIGNATURES = {
    "PortScan": ("DEMO SCAN Potential SYN Scan", 0.90),
    "DDoS": ("DEMO DOS SYN Flood", 0.70),
    "DoS Hulk": ("DEMO DOS HTTP Flood", 0.60),
    "SSH-Patator": ("DEMO BRUTE SSH Login Attempts", 0.50),
    BENIGN: ("DEMO POLICY Suspicious Outbound", 0.01),
}


def generate_flows(n: int = 20_000, seed: int = RANDOM_STATE, sigma: float = 0.45) -> pd.DataFrame:
    """Створює n потоків із 22 ознаками, службовими полями та міткою Label."""
    rng = np.random.default_rng(seed)
    names = list(PROFILES)
    shares = np.array([PROFILES[c][0] for c in names])
    labels = rng.choice(names, size=n, p=shares / shares.sum())
    base = np.array([PROFILES[c][1:] for c in labels], dtype=float)
    noise = rng.lognormal(0.0, sigma, size=(n, 5))
    duration = base[:, 0] * noise[:, 0]
    fwd = np.maximum(1, np.round(base[:, 1] * noise[:, 1]))
    bwd = np.maximum(0, np.round(base[:, 2] * noise[:, 2]))
    fwd_mean, bwd_mean = base[:, 3] * noise[:, 3], base[:, 4] * noise[:, 4]
    bwd_mean = np.where(bwd > 0, bwd_mean, 0.0)
    fwd_max, bwd_max = fwd_mean * rng.uniform(1, 3, n), bwd_mean * rng.uniform(1, 3, n)
    bwd_min = bwd_mean * rng.uniform(0, 1, n)
    total_fwd, total_bwd, packets = fwd * fwd_mean, bwd * bwd_mean, fwd + bwd
    pkt_mean, pkt_max = (total_fwd + total_bwd) / packets, np.maximum(fwd_max, bwd_max)
    pkt_std = pkt_max / 4 * rng.uniform(0.5, 1.5, n)
    seconds = duration / 1e6
    window_noise = rng.choice([1.0, 1.0, 1.0, 0.5, 2.0], size=(n, 2))
    start = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
    offsets = np.cumsum(rng.exponential(0.22, n))
    src = [f"192.168.10.{h}" for h in rng.integers(5, 60, n)]
    dst = [f"192.168.10.{h}" for h in rng.choice([3, 50, 51], n)]
    sport, dport = rng.integers(1024, 65535, n), rng.choice([22, 80, 443, 8080], n)
    frame = pd.DataFrame({
        "Flow ID": [f"{s}-{d}-{sp}-{dp}-6" for s, d, sp, dp in zip(src, dst, sport, dport)],
        "Source IP": src, "Source Port": sport, "Destination IP": dst, "Destination Port": dport,
        "Protocol": 6,
        "Timestamp": [(start + timedelta(seconds=float(o))).isoformat() for o in offsets],
        "Flow Duration": np.round(duration),
        "Total Fwd Packets": fwd, "Total Backward Packets": bwd,
        "Total Length of Fwd Packets": total_fwd, "Total Length of Bwd Packets": total_bwd,
        "Fwd Packet Length Max": fwd_max, "Fwd Packet Length Mean": fwd_mean,
        "Bwd Packet Length Max": bwd_max, "Bwd Packet Length Mean": bwd_mean,
        "Bwd Packet Length Min": bwd_min, "Bwd Packet Length Std": (bwd_max - bwd_min) / 4,
        "Packet Length Mean": pkt_mean, "Packet Length Std": pkt_std,
        "Packet Length Variance": pkt_std ** 2, "Max Packet Length": pkt_max,
        "Avg Bwd Segment Size": bwd_mean, "Average Packet Size": pkt_mean * (1 + 1 / packets),
        "Flow Bytes/s": (total_fwd + total_bwd) / seconds, "Flow Packets/s": packets / seconds,
        "Init_Win_bytes_forward": np.round(base[:, 5] * window_noise[:, 0]),
        "Init_Win_bytes_backward": np.round(base[:, 6] * window_noise[:, 1]),
        "Down/Up Ratio": np.floor(bwd / fwd),
        "Label": labels,
    })
    return frame


def write_demo_eve(flows: pd.DataFrame, path, seed: int = RANDOM_STATE) -> int:
    """Імітує журнал Suricata EVE JSON: записує оповіщення для частини потоків
    відповідно до DEMO_SIGNATURES. Повертає кількість записаних оповіщень."""
    rng = np.random.default_rng([seed, 1])   # окремий потік, незалежний від generate_flows
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row, draw in zip(flows.itertuples(index=False), rng.random(len(flows))):
            signature, rate = DEMO_SIGNATURES.get(row.Label, ("", 0.0))
            if draw >= rate:
                continue
            event = {
                "timestamp": row.Timestamp, "event_type": "alert", "proto": "TCP",
                "src_ip": row[1], "src_port": int(row[2]), "dest_ip": row[3], "dest_port": int(row[4]),
                "alert": {"signature": signature, "signature_id": 9000000 + written % 5,
                          "category": "demo", "severity": 2},
            }
            handle.write(json.dumps(event) + "\n")
            written += 1
    return written
