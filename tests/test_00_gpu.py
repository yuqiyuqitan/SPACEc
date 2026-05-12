import pytest


@pytest.mark.gpu
def test_gpu_available():
    """TODO: Add description.
    
    Returns
    -------
    Any
        TODO: Describe return value.
    """
    import spacec as sp

    assert sp.hf.check_for_gpu(tensorflow=True, torch=True)
