"""Command line entrypoint for one or all experiment folders."""

import argparse
from pathlib import Path

from ml_playground.experiments.config import discover_experiments, load_experiment
from ml_playground.experiments.runner import run_grid
from ml_playground.utils import Timer, log_summary, setup_logger


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Executa experimentos de ML configurados")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", help="Nome ou caminho de uma pasta de experimento")
    group.add_argument("--all", action="store_true", help="Executa todas as pastas válidas")
    parser.add_argument(
        "--experiments-root",
        default="experiments",
        help="Raiz das pastas de experimento (padrão: experiments)",
    )
    args = parser.parse_args(argv)

    logger = setup_logger("ml_playground.experiments", "experiments")
    timer = Timer(logger)
    processed = passed = failed = skipped = 0
    try:
        paths = _resolve_paths(args.experiment, args.all, args.experiments_root)
        for path in paths:
            processed += 1
            try:
                config = load_experiment(path)
                result = run_grid(config, write_reports=True)
                if result.get("best") is None:
                    failed += 1
                    logger.error("Nenhum resultado válido em %s", path)
                else:
                    passed += 1
            except Exception:
                failed += 1
                logger.exception("Falha no experimento %s", path)
    except Exception:
        failed += 1
        logger.exception("Falha ao descobrir os experimentos")
    finally:
        log_summary(logger, processed, passed, failed, skipped)
        timer.finish()
    return 0 if processed > 0 and failed == 0 else 1


def _resolve_paths(experiment, run_all, root) -> list[Path]:
    root_path = Path(root)
    if run_all:
        paths = discover_experiments(root_path)
        if not paths:
            raise ValueError(f"Nenhuma pasta de experimento encontrada em {root_path}")
        return paths

    candidate = Path(experiment)
    if not candidate.exists():
        candidate = root_path / experiment
    return [candidate]


if __name__ == "__main__":
    raise SystemExit(main())
