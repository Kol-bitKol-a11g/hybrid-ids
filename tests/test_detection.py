"""Тести сигнатурного сигналу, оповіщень і сховища."""
import json
from datetime import datetime, timedelta, timezone

from hybrid_ids.alerting import AlertCorrelator, attack_profile, build_alert
from hybrid_ids.reconciliation import RULE_RECONCILED, RULE_SIGNATURE
from hybrid_ids.storage import AlertStore
from hybrid_ids.suricata_runner import SuricataIndex, parse_eve

T0 = datetime(2026, 3, 2, 10, 23, 41, tzinfo=timezone.utc)
EVE_ALERT = {
    "timestamp": "2026-03-02T12:23:41.120000+0200", "event_type": "alert", "proto": "TCP",
    "src_ip": "192.168.56.105", "src_port": 51234, "dest_ip": "192.168.56.20", "dest_port": 22,
    "alert": {"signature": "ET SCAN Potential SYN Scan", "signature_id": 2001219,
              "category": "attempted-recon", "severity": 2},
}


def _index() -> SuricataIndex:
    lines = [json.dumps(EVE_ALERT), "пошкоджений рядок", json.dumps({"event_type": "flow"})]
    alerts = parse_eve(lines)
    assert len(alerts) == 1 and alerts[0].timestamp == T0 + timedelta(microseconds=120000)
    return SuricataIndex(alerts)


def test_suricata_alert_matches_flow_in_both_directions():
    index = _index()
    forward = index.match("192.168.56.105", 51234, "192.168.56.20", 22, 6, T0)
    backward = index.match("192.168.56.20", 22, "192.168.56.105", 51234, 6, T0)
    assert forward is backward and forward.signature_id == 2001219
    assert index.match("192.168.56.105", 51234, "192.168.56.20", 22, 6, T0 + timedelta(minutes=5)) is None
    assert index.match("192.168.56.105", 51234, "192.168.56.20", 80, 6, T0) is None      # інший порт
    assert index.match("192.168.56.105", 51234, "192.168.56.20", 22, 17, T0) is None     # інший протокол


def test_attack_classes_map_to_mitre_techniques():
    assert attack_profile("DDoS") == ("T1498", 4)
    assert attack_profile("DoS Hulk") == ("T1499", 4)
    assert attack_profile("PortScan") == ("T1046", 2)
    assert attack_profile("SSH-Patator") == ("T1110", 3)
    assert attack_profile("Web Attack - Sql Injection") == ("T1190", 3)
    assert attack_profile("Heartbleed") == ("T1040,T1210", 4)
    assert attack_profile("Anomaly") == (None, 1)


def _alert(rule=RULE_RECONCILED, p_xgb=0.67, l_ae=0.0341, ml_class="Web Attack - Sql Injection",
           signature=None, offset_s=0, flow_id="f-1"):
    meta = {"flow_id": flow_id, "timestamp": T0 + timedelta(seconds=offset_s),
            "src_ip": "192.168.56.105", "dst_ip": "192.168.56.20", "src_port": 40000, "dst_port": 80}
    return build_alert(meta, "attack", rule, int(rule == RULE_SIGNATURE), p_xgb, l_ae,
                       ml_class, signature, p1=0.90, p2=0.30)


def test_alert_types_follow_rule_branches():
    reconciled = _alert()
    assert (reconciled.alert_type, reconciled.detector) == ("reconciled", "xgboost+autoencoder")
    assert reconciled.mitre_technique == "T1190"
    flood = _alert(RULE_SIGNATURE, 0.9963, None, "DDoS", "ET DOS SYN Flood")
    assert (flood.alert_type, flood.detector, flood.severity) == ("signature", "suricata+xgboost", 4)
    scan = _alert(RULE_SIGNATURE, 0.05, None, "DoS Hulk", "ET SCAN Potential SYN Scan")
    assert (scan.detector, scan.attack_class, scan.mitre_technique) == ("suricata", "PortScan", "T1046")
    assert _alert(flow_id="f-1").flow_id == reconciled.flow_id       # стабільний ідентифікатор потоку


def test_correlator_groups_repeated_alerts():
    correlator = AlertCorrelator(window_s=300)
    flags = [correlator.add(_alert(offset_s=s, flow_id=f"f-{s}")) for s in (0, 60, 200, 900)]
    assert flags == [True, False, False, True]       # три спрацювання за 5 хв - один інцидент
    assert (correlator.alerts, correlator.groups) == (4, 2)


def test_store_keeps_alerts_and_analyst_verdicts():
    store = AlertStore("sqlite:///")
    first, second = _alert(), _alert(RULE_SIGNATURE, 0.05, None, "DoS Hulk", "ET SCAN Potential SYN Scan",
                                    offset_s=30, flow_id="f-2")
    assert store.insert(first) == 1 and store.insert(second) == 2
    rows = store.query()
    assert [row["flow_id"] for row in rows] == [second.flow_id, first.flow_id]     # новіші спершу
    assert len(store.query(attack_class="PortScan")) == 1
    assert len(store.query(min_confidence=0.5)) == 1
    assert len(store.query(time_from=T0 + timedelta(seconds=10))) == 1
    assert store.verify(first.flow_id, confirmed=False, comment="фоновий трафік")
    assert not store.verify("00000000-0000-0000-0000-000000000000", confirmed=True)
    assert store.verdict_counts() == (0, 1)
    assert store.alerts_by_type(T0 - timedelta(hours=1)) == {"reconciled": 1, "signature": 1}
    store.save_stats({"xgboost": "up"})
    assert store.load_stats() == {"xgboost": "up"}
