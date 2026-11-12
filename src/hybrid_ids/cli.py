"""Інтерфейс командного рядка прототипу: python -m hybrid_ids <команда>."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import LABEL_COLUMN, META_COLUMNS, load_settings


def _parts(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(pd.read_csv(Path(data_dir) / f"{name}.csv", low_memory=False)
                 for name in ("train", "val", "test"))


def _settings(args):
    settings = load_settings(getattr(args, "config", None))
    if getattr(args, "models", None):
        settings = replace(settings, models_dir=args.models)
    if getattr(args, "db", None):
        settings = replace(settings, db_url=args.db)
    return settings


def cmd_prepare(args) -> None:
    from .training import prepare_dataset

    print(prepare_dataset(args.csv, args.out, args.min_class_size).to_string(index=False))


def cmd_select_features(args) -> None:
    from .data_preparation import MinMaxNormalizer
    from .feature_selection import select_features

    train = _parts(args.data)[0]
    if args.sample and len(train) > args.sample:
        train = train.sample(args.sample, random_state=42)
    skip = {*META_COLUMNS, LABEL_COLUMN, "p_sur"}
    X = train[[c for c in train.columns if c not in skip]].select_dtypes("number")
    X = pd.DataFrame(MinMaxNormalizer().fit(X).transform(X), columns=X.columns)
    selected, table = select_features(X, train[LABEL_COLUMN].to_numpy(), top_k=args.top_k)
    print(table.round(4).to_string())
    print(f"\nВідібрано ознак: {len(selected)} із {X.shape[1]}")


def cmd_train(args) -> None:
    from .training import train_models

    train, val, _ = _parts(args.data)
    summary = train_models(train, val, _settings(args).models_dir, args.backend,
                           balance=not args.no_balance, calibrate=args.calibrate)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_evaluate(args) -> None:
    from .pipeline import HybridIDS
    from .suricata_runner import SuricataIndex, read_eve
    from .training import evaluate_models

    test = _parts(args.data)[2]
    p_sur = test["p_sur"].to_numpy() if "p_sur" in test.columns else None
    if args.eve:
        p_sur, _ = SuricataIndex(read_eve(args.eve)).signals(test)
    table, summary, per_class = evaluate_models(test, HybridIDS.load(_settings(args)), p_sur)
    print("Абляційні конфігурації (норма проти атаки):")
    print(table.drop(columns=["tp", "fp", "fn", "tn"]).round(3).to_string())
    print("\nБагатокласова класифікація:", {k: round(v, 3) for k, v in summary.items()})
    print(per_class.round(3).to_string())


def cmd_compare(args) -> None:
    from .baselines import compare_models
    from .data_preparation import MinMaxNormalizer
    from .feature_extraction import feature_matrix

    train, _, test = _parts(args.data)
    normalizer = MinMaxNormalizer().fit(feature_matrix(train)[0])
    table = compare_models(normalizer.transform(feature_matrix(train)[0]), train[LABEL_COLUMN],
                           normalizer.transform(feature_matrix(test)[0]), test[LABEL_COLUMN],
                           models=args.only.split(","))
    print(table.drop(columns=["tp", "fp", "fn", "tn"]).round(3).to_string())


def cmd_run(args) -> None:
    from .feature_extraction import load_flows
    from .pipeline import HybridIDS
    from .storage import AlertStore
    from .suricata_runner import SuricataIndex, read_eve

    settings = _settings(args)
    ids = HybridIDS.load(settings, AlertStore(settings.db_url))
    index = SuricataIndex(read_eve(args.eve)) if args.eve else None
    alerts = ids.process(load_flows(args.flows), index)
    if args.out:
        Path(args.out).write_text(
            "".join(json.dumps(a.to_dict(), ensure_ascii=False) + "\n" for a in alerts), encoding="utf-8")
    print(json.dumps({**ids.stats.snapshot(), "alerts": len(alerts),
                      "alert_groups": ids.correlator.groups}, ensure_ascii=False, indent=2))


def cmd_worker(args) -> None:
    from .services import run_worker

    run_worker(_settings(args), args.flows, args.eve, args.interval)


def cmd_serve(args) -> None:
    import uvicorn

    from .api import create_app
    from .storage import AlertStore

    uvicorn.run(create_app(AlertStore(_settings(args).db_url)), host=args.host, port=args.port)


def cmd_service(args) -> None:
    import uvicorn

    from .services import create_inference_app

    uvicorn.run(create_inference_app(args.kind, _settings(args).models_dir), host=args.host, port=args.port)


def cmd_demo(args) -> None:
    """Повний цикл на синтетичних потоках: підготовка -> навчання -> абляція -> оповіщення."""
    from .pipeline import HybridIDS
    from .storage import AlertStore
    from .suricata_runner import SuricataIndex, read_eve
    from .synthetic import generate_flows, write_demo_eve
    from .training import evaluate_models, prepare_dataset, train_models

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    flows = generate_flows(args.flows)
    flows.to_csv(work / "flows.csv", index=False)
    write_demo_eve(flows, work / "eve.json")
    print("УВАГА: дані синтетичні. Запуск показує працездатність конвеєра, а не результати,\n"
          "отримані в роботі на наборах CICIDS2017, NSL-KDD та UNSW-NB15.\n")
    print("1. Очищення та поділ 70/15/15:")
    print(prepare_dataset([work / "flows.csv"], work / "processed").to_string(index=False))
    train, val, test = _parts(work / "processed")
    print("\n2. Навчання моделей:")
    summary = train_models(train, val, work / "models", args.backend, calibrate=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    settings = replace(load_settings(), models_dir=str(work / "models"))
    index = SuricataIndex(read_eve(work / "eve.json"))
    ids = HybridIDS.load(settings, AlertStore(f"sqlite:///{work / 'alerts.db'}"))
    print("\n3. Абляційні конфігурації на тестовій частині:")
    table, _, per_class = evaluate_models(test, ids, index.signals(test)[0])
    print(table[["macro_f1", "f1", "precision", "recall", "fpr", "roc_auc"]].round(3).to_string())
    print("\nПовнота класифікатора за класами:")
    print(per_class.round(3).to_string())

    alerts = ids.process(test, index)
    stats = ids.stats.snapshot()
    print(f"\n4. Обробка тестових потоків конвеєром: {stats['flows_total']} потоків, "
          f"{len(alerts)} оповіщень у {ids.correlator.groups} групах, "
          f"{stats['avg_response_ms']} мс/потік, автоенкодер викликано для "
          f"{stats['autoencoder_share']:.1%} потоків")
    by_type = pd.Series([a.alert_type for a in alerts]).value_counts()
    print(by_type.to_string())
    for alert_type in by_type.index[:3]:
        example = next(a for a in alerts if a.alert_type == alert_type)
        print(json.dumps(example.to_dict(), ensure_ascii=False))
    print(f"\nОповіщення збережено: {work / 'alerts.db'}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="hybrid-ids", description=__doc__)
    parser.add_argument("--config", help="шлях до config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        return p

    p = command("demo", cmd_demo, "повний цикл на синтетичних потоках")
    p.add_argument("--flows", type=int, default=20_000)
    p.add_argument("--workdir", default="data/demo")
    p.add_argument("--backend", choices=("auto", "full", "lite"), default="auto")

    p = command("prepare", cmd_prepare, "очищення та поділ train/val/test")
    p.add_argument("csv", nargs="+")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--min-class-size", type=int, default=5)

    p = command("select-features", cmd_select_features, "відбір ознак трьома методами")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--sample", type=int, default=200_000)

    p = command("train", cmd_train, "навчання XGBoost та автоенкодера")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--models")
    p.add_argument("--backend", choices=("auto", "full", "lite"), default="auto")
    p.add_argument("--no-balance", action="store_true", help="без SMOTE і ваг класів (абляція)")
    p.add_argument("--calibrate", action="store_true", help="масштабувати tau1, tau2 до tau_base моделі")

    p = command("evaluate", cmd_evaluate, "абляційні досліди на тестовій частині")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--models")
    p.add_argument("--eve", help="журнал Suricata EVE JSON для обчислення p_sur")

    p = command("compare", cmd_compare, "порівняння базової та контрольних моделей")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--only", default="rf,svm,iforest,ocsvm", help="rf,svm,dnn,lstm,iforest,ocsvm")

    for name, func, help_text in (("run", cmd_run, "обробка таблиці потоків CICFlowMeter"),
                                  ("worker", cmd_worker, "ядро узгодження (безперервний режим)")):
        p = command(name, func, help_text)
        p.add_argument("--flows", required=True, help="CSV із потоками CICFlowMeter")
        p.add_argument("--eve", help="журнал Suricata EVE JSON")
        p.add_argument("--models")
        p.add_argument("--db", help="рядок підключення до сховища оповіщень")
        if name == "run":
            p.add_argument("--out", help="файл JSON Lines для оповіщень")
        else:
            p.add_argument("--interval", type=float, default=2.0)

    p = command("serve", cmd_serve, "REST API для SOC")
    p.add_argument("--db")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)

    p = command("service", cmd_service, "сервіс висновування детектора")
    p.add_argument("kind", choices=("xgboost", "autoencoder"))
    p.add_argument("--models")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
