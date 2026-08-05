import logging
from ml_playground.utils import setup_logger, log_summary


def test_setup_logger(tmp_path):
    logger = setup_logger("test_log", "test_script", log_dir=tmp_path)
    assert logger.level == logging.INFO
    log_file = tmp_path / "test_script.log"
    assert log_file.exists()
    for h in logger.handlers:
        h.close()
        logger.removeHandler(h)


def test_log_summary(tmp_path):
    logger = setup_logger("test_summary", "test_summary", log_dir=tmp_path)
    log_summary(logger, processed=10, passed=8, failed=2, skipped=0)
    log_file = tmp_path / "test_summary.log"
    content = log_file.read_text(encoding="utf-8")
    assert "Summary" in content
    assert "Total processed: 10" in content
    assert "Passed: 8" in content
    assert "Failed: 2" in content
    for h in logger.handlers:
        h.close()
        logger.removeHandler(h)
