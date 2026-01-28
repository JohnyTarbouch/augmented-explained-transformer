from aet.utils.filters import should_skip_flip


def test_should_skip_flip():
    assert should_skip_flip(True, True)
    assert not should_skip_flip(False, True)
    assert not should_skip_flip(True, False)
    assert not should_skip_flip(False, False)
