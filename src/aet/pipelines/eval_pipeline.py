from aet.training.eval import evaluate_model
from aet.utils.logging import get_logger


def run(cfg: dict) -> None:
    logger = get_logger(__name__)
    logger.info("Running evaluation pipeline.")
    evaluate_model(cfg)
