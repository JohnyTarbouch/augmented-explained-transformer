"""Training pipeline entry point."""

from aet.training.train import train_baseline
from aet.utils.logging import get_logger


def run(cfg: dict) -> None:
    """Run the baseline training pipeline."""
    logger = get_logger(__name__)
    logger.info("Running baseline training pipeline.")
    train_baseline(cfg)
