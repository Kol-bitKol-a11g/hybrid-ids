"""Витяг ознак: таблиця потоків CICFlowMeter -> вектор із 22 ознак.

CICFlowMeter агрегує пакети у двонапрямні потоки й обчислює 78 ознак. Назви
колонок відрізняються між версіями інструмента, тому вони зводяться до єдиної схеми.
Потоки з неповними або нечисловими ознаками не зараховуються до норми, а
позначаються як технічні помилки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES

# Відповідність назв колонок різних версій CICFlowMeter канонічним назвам
# (скорочений формат CICFlowMeter-V4 та snake_case Python-реалізації).
# Перед запуском варто звірити заголовок CSV вашої версії інструмента.
ALIASES = {
    "Src IP": "Source IP", "src_ip": "Source IP",
    "Dst IP": "Destination IP", "dst_ip": "Destination IP",
    "Src Port": "Source Port", "src_port": "Source Port",
    "Dst Port": "Destination Port", "dst_port": "Destination Port",
    "protocol": "Protocol", "timestamp": "Timestamp", "flow_id": "Flow ID", "label": "Label",
    "flow_duration": "Flow Duration",
    "Tot Fwd Pkts": "Total Fwd Packets", "tot_fwd_pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets", "tot_bwd_pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets", "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets", "totlen_bwd_pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max", "fwd_pkt_len_max": "Fwd Packet Length Max",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean", "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "Bwd Pkt Len Max": "Bwd Packet Length Max", "bwd_pkt_len_max": "Bwd Packet Length Max",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean", "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Min": "Bwd Packet Length Min", "bwd_pkt_len_min": "Bwd Packet Length Min",
    "Bwd Pkt Len Std": "Bwd Packet Length Std", "bwd_pkt_len_std": "Bwd Packet Length Std",
    "Pkt Len Mean": "Packet Length Mean", "pkt_len_mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std", "pkt_len_std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance", "pkt_len_var": "Packet Length Variance",
    "Pkt Len Max": "Max Packet Length", "pkt_len_max": "Max Packet Length",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size", "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "Pkt Size Avg": "Average Packet Size", "pkt_size_avg": "Average Packet Size",
    "Flow Byts/s": "Flow Bytes/s", "flow_byts_s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s", "flow_pkts_s": "Flow Packets/s",
    "Init Fwd Win Byts": "Init_Win_bytes_forward", "init_fwd_win_byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward", "init_bwd_win_byts": "Init_Win_bytes_backward",
    "down_up_ratio": "Down/Up Ratio",
}


class SchemaError(ValueError):
    """Схема вхідної таблиці не збігається зі схемою, зафіксованою під час навчання."""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Прибирає пробіли в назвах колонок і зводить їх до канонічної схеми."""
    stripped = {col: str(col).strip() for col in df.columns}
    return df.rename(columns={col: ALIASES.get(name, name) for col, name in stripped.items()})


def load_flows(path, **read_csv_kwargs) -> pd.DataFrame:
    """Читає CSV із потоками CICFlowMeter (або файл CICIDS2017) у канонічній схемі."""
    return normalize_columns(pd.read_csv(path, low_memory=False, **read_csv_kwargs))


def parse_timestamps(values) -> pd.Series:
    """Розбирає часові позначки потоків і зводить їх до UTC.

    ISO 8601 читається як є; інші записи трактуються у форматі CSV-файлів CICIDS2017
    «день/місяць/рік» (наприклад, 03/07/2017 08:55:58). Час без зони вважається UTC.
    """
    text = pd.Series(values).astype("string")
    iso = pd.to_datetime(text, format="ISO8601", errors="coerce", utc=True)
    other = pd.to_datetime(text.where(iso.isna()), format="mixed", dayfirst=True,
                           errors="coerce", utc=True)
    return iso.fillna(other)


def feature_matrix(df: pd.DataFrame, features=FEATURES) -> tuple[np.ndarray, np.ndarray]:
    """Формує матрицю ознак у зафіксованому порядку колонок.

    Повертає (X, valid): valid[i] = False для потоків із пропусками, нечисловими
    або нескінченними значеннями - їх слід записати до журналу технічних помилок.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SchemaError(f"У таблиці потоків бракує ознак: {missing}")
    X = df[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(X).all(axis=1)
    return np.where(np.isfinite(X), X, 0.0), valid
