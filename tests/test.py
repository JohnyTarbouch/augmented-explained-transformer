import os

import pytest

transformers_interpret = pytest.importorskip("transformers_interpret")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

if not os.getenv("RUN_SLOW_TESTS"):
    pytest.skip("Set RUN_SLOW_TESTS=1 to enable model download tests.", allow_module_level=True)


def test_transformers_interpret_smoke():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    explainer = SequenceClassificationExplainer(model, tokenizer)
    word_attributions = explainer("I love you, I like you")
    assert word_attributions
