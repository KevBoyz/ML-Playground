"""Command-line lifecycle for configured tabular experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_playground.data.loader import auto_read
from ml_playground.experiments.config import discover_experiments, load_experiment
from ml_playground.experiments.runner import build_run_plan, preflight_experiment, run_grid
from ml_playground.experiments.templates import initialize_experiment
from ml_playground.models.inference import predict_batch
from ml_playground.utils import Timer, log_summary, setup_logger


def main(argv=None) -> int:
    """Run, inspect, scaffold or score an experiment without hidden defaults."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"
    logger = setup_logger("ml_playground.experiments", "experiments")
    timer = Timer(logger)
    processed = passed = failed = skipped = 0
    try:
        if command == "init":
            paths = _initialize(args)
            print(json.dumps({"created": [str(path) for path in paths]}, ensure_ascii=False, indent=2))
            processed = passed = 1
        elif command == "predict":
            _predict(args)
            processed = passed = 1
        else:
            paths = _resolve_paths(args.experiment, args.all, args.experiments_root)
            for path in paths:
                processed += 1
                try:
                    config = load_experiment(path)
                    if command == "validate":
                        runtime = preflight_experiment(config)
                        print(_json_output({"experiment": config["experiment_name"], "preflight": runtime["preflight_metadata"]}))
                    elif command == "dry-run":
                        print(_json_output(build_run_plan(config)))
                    else:
                        result = run_grid(config, write_reports=True)
                        if result.get("best") is None:
                            failed += 1
                            logger.error("Nenhum resultado válido em %s", path)
                            continue
                        print(
                            _json_output(
                                {
                                    "experiment": config["experiment_name"],
                                    "run_id": (result.get("tracker") or {}).get("run_id"),
                                    "selected_candidate": result["best"].get("candidate_id"),
                                    "development_metrics": result["best"].get("metrics", {}),
                                    "final_test_metrics": (result["best"].get("final_test") or {}).get("metrics", {}),
                                    "reports": result.get("reports", {}),
                                }
                            )
                        )
                    passed += 1
                except Exception:
                    failed += 1
                    logger.exception("Falha no experimento %s", path)
    except Exception:
        failed += 1
        logger.exception("Falha ao executar o comando %s", command)
    finally:
        log_summary(logger, processed, passed, failed, skipped)
        timer.finish()
    return 0 if processed > 0 and failed == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimentos tabulares reproduzíveis")
    parser.add_argument("--experiment", help="Nome ou caminho de uma pasta de experimento (alias de run)")
    parser.add_argument("--all", action="store_true", help="Executa todas as pastas válidas (alias de run)")
    parser.add_argument("--experiments-root", default="experiments", help="Raiz das pastas de experimento")
    subcommands = parser.add_subparsers(dest="command")

    for command, help_text in (
        ("run", "treina, seleciona, avalia e publica um experimento"),
        ("validate", "executa somente o preflight de dados"),
        ("dry-run", "mostra candidatos e custo sem treinar"),
    ):
        child = subcommands.add_parser(command, help=help_text)
        _add_experiment_selector(child)

    init = subcommands.add_parser("init", help="cria um template de experimento")
    init.add_argument("--task", choices=["classification", "regression", "clustering"], required=True)
    init.add_argument("--name", required=True, help="Nome da pasta e do experimento")
    init.add_argument("--experiments-root", default="experiments")
    init.add_argument("--directory", help="Destino explícito; padrão: <experiments-root>/<name>")
    init.add_argument("--overwrite", action="store_true", help="Permite sobrescrever YAMLs do template")

    predict = subcommands.add_parser("predict", help="pontua um lote com um modelo persistido")
    predict.add_argument("--model", required=True, help="Caminho para model.joblib")
    predict.add_argument("--input", required=True, help="CSV, Parquet ou planilha de entrada")
    predict.add_argument("--output", required=True, help="CSV ou Parquet de saída")
    predict.add_argument("--format", choices=["csv", "parquet"], dest="output_format")
    predict.add_argument("--id-column")
    predict.add_argument("--non-strict", action="store_true", help="Aceita divergências de dtype da assinatura")
    predict.add_argument("--allow-extra-columns", action="store_true")
    return parser


def _add_experiment_selector(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", help="Nome ou caminho de uma pasta de experimento")
    group.add_argument("--all", action="store_true", help="Seleciona todas as pastas válidas")
    parser.add_argument("--experiments-root", default="experiments", help="Raiz das pastas de experimento")


def _resolve_paths(experiment, run_all, root) -> list[Path]:
    root_path = Path(root)
    if run_all:
        paths = discover_experiments(root_path)
        if not paths:
            raise ValueError(f"Nenhuma pasta de experimento encontrada em {root_path}")
        return paths
    if not experiment:
        raise ValueError("Informe --experiment ou --all")
    candidate = Path(experiment)
    if not candidate.exists():
        candidate = root_path / experiment
    return [candidate]


def _initialize(args) -> list[Path]:
    directory = Path(args.directory) if args.directory else Path(args.experiments_root) / args.name
    return initialize_experiment(directory, task=args.task, name=args.name, overwrite=args.overwrite)


def _predict(args) -> None:
    batch = auto_read(args.input)
    result = predict_batch(
        args.model,
        batch,
        id_column=args.id_column,
        strict=not args.non_strict,
        allow_extra_columns=True if args.allow_extra_columns else None,
        output_path=args.output,
        output_format=args.output_format,
    )
    print(_json_output({"output": result.output_path, "validation": result.validation, "rows": len(result.frame)}))


def _json_output(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
