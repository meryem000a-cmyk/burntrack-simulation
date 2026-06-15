import logging


def setup_logging(level: int = logging.WARNING) -> None:
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
