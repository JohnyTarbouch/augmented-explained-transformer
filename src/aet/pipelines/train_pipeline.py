from aet.utils.logging import get_logger


def run(cfg: dict) -> None:
    logger = get_logger(__name__)
    logger.info("Training pipeline scaffold. Implement train_baseline/train_augmented.")
    raise NotImplementedError("Training pipeline not implemented.")
