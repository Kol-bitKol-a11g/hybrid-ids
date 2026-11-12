"""Наскрізний тест конвеєра на синтетичних потоках (спрощені моделі scikit-learn)."""
import numpy as np
import pandas as pd
import pytest

from hybrid_ids.config import Settings
from hybrid_ids.pipeline import HybridIDS
from hybrid_ids.storage import AlertStore
from hybrid_ids.suricata_runner import SuricataIndex, read_eve
from hybrid_ids.synthetic import generate_flows, write_demo_eve
from hybrid_ids.training import evaluate_models, prepare_dataset, train_models


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    work = tmp_path_factory.mktemp("ids")
    flows = generate_flows(3000, seed=7)
    flows.to_csv(work / "flows.csv", index=False)
    write_demo_eve(flows, work / "eve.json", seed=7)
    prepare_dataset([work / "flows.csv"], work / "processed")
    train, val, test = (pd.read_csv(work / "processed" / f"{name}.csv") for name in ("train", "val", "test"))
    summary = train_models(train, val, work / "models", backend="lite", calibrate=True)
    ids = HybridIDS.load(Settings(models_dir=str(work / "models")), AlertStore("sqlite:///"))
    return work, test, ids, summary


def test_training_summary(workspace):
    _, _, ids, summary = workspace
    assert summary["train_flows"] == 2100
    assert "BENIGN" in summary["classes"] and summary["tau_base"] > 0
    assert ids.thresholds.tau2 < ids.thresholds.tau_base < ids.thresholds.tau1


def test_pipeline_produces_alerts_and_respects_signature_priority(workspace):
    work, test, ids, _ = workspace
    index = SuricataIndex(read_eve(work / "eve.json"))
    p_sur, _ = index.signals(test)
    assert 0 < p_sur.sum() < len(test)
    alerts = ids.process(test, index)
    by_type = pd.Series([a.alert_type for a in alerts]).value_counts()
    assert by_type["signature"] == p_sur.sum()            # жодне сигнатурне спрацювання не втрачено
    assert by_type.get("ml_confident", 0) > 0
    assert len(ids.store.query(limit=1000)) == len(alerts)
    stats = ids.stats.snapshot()
    assert stats["flows_total"] == len(test) and stats["suricata"] == "up"
    assert 0 < stats["autoencoder_share"] < 1             # автоенкодер викликається не для всіх потоків


def test_ablation_has_five_configurations(workspace):
    work, test, ids, _ = workspace
    p_sur, _ = SuricataIndex(read_eve(work / "eve.json")).signals(test)
    table, summary, per_class = evaluate_models(test, ids, p_sur)
    assert len(table) == 5 and table["recall"].between(0, 1).all()
    union, hybrid = table.iloc[3], table.iloc[4]
    assert union["recall"] >= table.iloc[:3]["recall"].max()   # АБО не втрачає жодного спрацювання
    assert hybrid["recall"] >= table.iloc[0]["recall"]         # сигнатура має пріоритет
    assert summary["macro_f1"] > 0.5 and per_class.index.tolist() == sorted(test["Label"].unique())


def test_broken_flows_are_technical_errors_not_normal(workspace):
    _, test, ids, _ = workspace
    broken = test.head(20).copy()
    broken["Flow Bytes/s"] = broken["Flow Bytes/s"].astype(object)
    broken.iloc[0, broken.columns.get_loc("Flow Bytes/s")] = "NaN"
    before = ids.stats.technical_errors
    ids.process(broken)
    assert ids.stats.technical_errors == before + 1


def test_scores_without_signature_signal(workspace):
    _, test, ids, _ = workspace
    scores = ids.score(test[list(ids_features())].to_numpy(dtype=float))
    assert set(scores["verdict"]) <= {"attack", "normal", "review"}
    confident = scores["p_xgb"] > ids.thresholds.p1
    assert scores.loc[confident, "l_ae"].isna().all()     # впевнені потоки не йдуть в автоенкодер
    assert np.isfinite(scores.loc[~confident, "l_ae"]).all()


def ids_features():
    from hybrid_ids.config import FEATURES
    return FEATURES
