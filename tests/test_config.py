from pathlib import Path

from aet.config import load_config


def test_load_config():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "base.yaml")
    assert cfg["model"]["name"] == "distilbert-base-uncased"
