import logging
import time
from pathlib import Path


def setup_logger(name: str, script_name: str, log_dir: str | Path = "logs") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_dir / f"{script_name}.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


def log_summary(
    logger: logging.Logger, processed: int, passed: int, failed: int, skipped: int = 0
):
    logger.warning("--- Summary ---")
    logger.warning(f"Total processed: {processed}")
    logger.warning(f"Passed: {passed}")
    logger.warning(f"Failed: {failed}")
    logger.warning(f"Skipped: {skipped}")


class Timer:
    def __init__(self, logger: logging.Logger):
        self._start = time.perf_counter()
        self._logger = logger

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def finish(self):
        self._logger.warning(f"Execution time: {self.elapsed():.2f}s")
