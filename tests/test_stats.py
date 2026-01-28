from aet.utils.stats import summarize_with_ci


def test_summarize_with_ci_empty():
    stats = summarize_with_ci([], num_bootstrap=10, confidence=0.9, seed=0)
    assert stats["n"] == 0
    assert stats["mean"] == 0.0
    assert stats["std"] == 0.0
    assert stats["ci_low"] == 0.0
    assert stats["ci_high"] == 0.0


def test_summarize_with_ci_constant():
    stats = summarize_with_ci([2.0, 2.0, 2.0], num_bootstrap=200, confidence=0.9, seed=123)
    assert stats["n"] == 3
    assert stats["mean"] == 2.0
    assert stats["std"] == 0.0
    assert stats["ci_low"] == 2.0
    assert stats["ci_high"] == 2.0


def test_summarize_with_ci_reproducible():
    values = [1.0, 2.0, 3.0, 4.0]
    stats_a = summarize_with_ci(values, num_bootstrap=300, confidence=0.8, seed=42)
    stats_b = summarize_with_ci(values, num_bootstrap=300, confidence=0.8, seed=42)
    assert stats_a["ci_low"] == stats_b["ci_low"]
    assert stats_a["ci_high"] == stats_b["ci_high"]
    assert stats_a["mean"] == stats_b["mean"]


def test_summarize_with_ci_bounds():
    values = [1.0, 2.0, 3.0, 4.0]
    stats = summarize_with_ci(values, num_bootstrap=300, confidence=0.8, seed=7)
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
    assert stats["ci_low"] < stats["ci_high"]
