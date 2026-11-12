"""Захоплення трафіку: запуск сенсорів Suricata та CICFlowMeter.

Обидва інструменти отримують ту саму копію трафіку: або той самий мережевий
інтерфейс (режим AF_PACKET / порт SPAN), або той самий PCAP-файл у відтворюваному
експерименті. Модуль лише формує команди запуску й керує процесами.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def suricata_command(out_dir, interface: str | None = None, pcap: str | None = None,
                     config: str = "/etc/suricata/suricata.yaml") -> list[str]:
    """Команда Suricata: живий інтерфейс (AF_PACKET) або відтворення PCAP.
    Журнал подій записується у <out_dir>/eve.json."""
    if bool(interface) == bool(pcap):
        raise ValueError("Вкажіть рівно одне джерело: interface або pcap")
    source = [f"--af-packet={interface}"] if interface else ["-r", str(pcap)]
    return ["suricata", "-c", config, *source, "-l", str(out_dir)]


def flowmeter_command(out_csv, interface: str | None = None, pcap: str | None = None) -> list[str]:
    """Команда CICFlowMeter: агрегація пакетів у потоки та запис ознак у CSV."""
    if bool(interface) == bool(pcap):
        raise ValueError("Вкажіть рівно одне джерело: interface або pcap")
    source = ["-i", interface] if interface else ["-f", str(pcap)]
    return ["cicflowmeter", *source, "-c", str(out_csv)]


def replay_pcap(pcap, out_dir) -> tuple[Path, Path]:
    """Відтворює PCAP через обидва сенсори; повертає шляхи до flows.csv та eve.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    flows = out / "flows.csv"
    subprocess.run(suricata_command(out, pcap=pcap), check=True)
    subprocess.run(flowmeter_command(flows, pcap=pcap), check=True)
    return flows, out / "eve.json"


def start_live(interface: str, out_dir) -> list[subprocess.Popen]:
    """Запускає обидва сенсори на живому інтерфейсі (потрібні права на захоплення)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        subprocess.Popen(suricata_command(out, interface=interface)),
        subprocess.Popen(flowmeter_command(out / "flows.csv", interface=interface)),
    ]
