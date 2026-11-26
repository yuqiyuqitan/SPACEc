import pytest


@pytest.mark.gpu
def test_gpu_available():
    import spacec as sp

    assert sp.hf.check_for_gpu(torch=True)
