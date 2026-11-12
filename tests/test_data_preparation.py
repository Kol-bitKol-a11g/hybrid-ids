"""Тести підготовки даних: очищення, поділ без витоку, нормалізація, балансування."""
import numpy as np
import pandas as pd

from hybrid_ids.config import FEATURES
from hybrid_ids.data_preparation import (MinMaxNormalizer, clean_flows, group_split,
                                         smote_targets, time_split)
from hybrid_ids.feature_extraction import feature_matrix, normalize_columns, parse_timestamps
from hybrid_ids.synthetic import generate_flows


def test_normalizer_uses_only_train_statistics():
    train = np.array([[0.0, 10.0, 5.0], [10.0, 20.0, 5.0]])
    test = np.array([[20.0, 15.0, 7.0]])              # значення поза діапазоном train
    normalizer = MinMaxNormalizer().fit(train)
    assert normalizer.transform(train).tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    assert normalizer.transform(test).tolist() == [[1.0, 0.5, 1.0]]      # обрізано до [0, 1]
    assert MinMaxNormalizer(clip=False).fit(train).transform(test)[0, 0] == 2.0


def test_cleaning_removes_invalid_rows_and_reports_steps():
    flows = generate_flows(2000, seed=1)
    flows = pd.concat([flows, flows.iloc[:10]], ignore_index=True)      # 10 повних дублікатів
    flows.loc[20, "Flow Bytes/s"] = np.inf
    flows.loc[21, "Flow Packets/s"] = np.nan
    flows.loc[22, "Flow Duration"] = -5
    flows.loc[23, "Label"] = "Infiltration"                              # клас з одним потоком
    cleaned, report = clean_flows(flows)
    assert report["Видалено"].tolist() == [10, 2, 1, 1]
    assert len(cleaned) == 1996
    assert np.isfinite(cleaned[list(FEATURES)].to_numpy(dtype=float)).all()


def test_timestamps_iso_and_day_first():
    stamps = parse_timestamps(["2026-03-02T09:00:00+00:00", "03/07/2017 08:55:58", "нема"])
    assert (stamps[0].month, stamps[0].day) == (3, 2)                    # ISO: рік-місяць-день
    assert (stamps[1].month, stamps[1].day) == (7, 3)                    # CICIDS2017: день/місяць
    assert pd.isna(stamps[2])


def test_time_split_keeps_chronology():
    flows = generate_flows(1000, seed=2).sample(frac=1.0, random_state=0)
    train, val, test = time_split(flows)
    assert (len(train), len(val), len(test)) == (700, 150, 150)
    assert train["Timestamp"].max() <= val["Timestamp"].min()
    assert val["Timestamp"].max() <= test["Timestamp"].min()


def test_group_split_does_not_share_groups():
    frame = pd.DataFrame({"group": np.repeat(np.arange(50), 4), "x": np.arange(200)})
    parts = group_split(frame, "group")
    groups = [set(part["group"]) for part in parts]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    assert sum(len(part) for part in parts) == len(frame)


def test_smote_targets_are_class_dependent():
    y = ["BENIGN"] * 8000 + ["DoS Hulk"] * 1500 + ["GoldenEye"] * 60 + ["Slowloris"] * 40 + ["Heartbleed"] * 4
    targets = smote_targets(y)
    assert "BENIGN" not in targets and "DoS Hulk" not in targets     # численні класи не змінюються
    assert targets["GoldenEye"] == targets["Slowloris"] == 600        # малочисельні: 10 x найбільший
    assert targets["Heartbleed"] == 40                                # вкрай малочисельні - окрема група


def test_feature_matrix_flags_technical_errors():
    flows = generate_flows(5, seed=3).rename(columns={"Flow Bytes/s": " Flow Byts/s"})
    flows = normalize_columns(flows)                  # назви зведено до канонічної схеми
    flows["Flow Packets/s"] = flows["Flow Packets/s"].astype(object)
    flows.loc[2, "Flow Packets/s"] = "Infinity"
    X, valid = feature_matrix(flows)
    assert X.shape == (5, len(FEATURES))
    assert valid.tolist() == [True, True, False, True, True]
