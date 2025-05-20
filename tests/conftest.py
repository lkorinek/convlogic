import pytest
import torch

# (implementation, device)
_IMPLS = [("python", "cpu")] + ([("cuda", "cuda")] if torch.cuda.is_available() else [])


@pytest.fixture(params=_IMPLS, ids=lambda x: x[0])
def impl_and_device(request):
    return request.param
