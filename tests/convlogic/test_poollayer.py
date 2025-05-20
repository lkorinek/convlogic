import pytest
import torch

from convlogic.convlogic_layer import PoolLogicLayer

device = "cpu"


@pytest.mark.parametrize(
    "batch,channels,h,w,kernel,stride",
    [
        (1, 1, 4, 4, 2, 2),
        (2, 3, 8, 8, (3, 3), (2, 2)),
    ],
)
def test_poollogic_forward_4d(batch, channels, h, w, kernel, stride):
    x = torch.randn(batch, channels, h, w, requires_grad=True)
    layer = PoolLogicLayer(kernel=kernel, stride=stride).to(device)

    y = layer(x)
    y_ref = torch.nn.functional.max_pool2d(x, kernel, stride)

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, atol=1e-6)

    (y.sum()).backward()
    assert x.grad is not None and x.grad.shape == x.shape


@pytest.mark.parametrize(
    "batch,channels,h,w,kernel,stride",
    [
        (1, 1, 4, 4, 2, 2),
        (2, 2, 6, 8, (2, 2), (2, 2)),
    ],
)
def test_poollogic_forward_5d(batch, channels, h, w, kernel, stride):
    x4 = torch.randn(batch, channels, h, w)
    x5 = x4.unsqueeze(-1).requires_grad_()
    layer = PoolLogicLayer(kernel=kernel, stride=stride).to(device)

    y = layer(x5)
    y_ref = torch.nn.functional.max_pool2d(x4, kernel, stride)

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, atol=1e-6)

    (y.sum()).backward()
    assert x5.grad is not None and x5.grad.shape == x5.shape


def test_poollogic_invalid_dim():
    layer = PoolLogicLayer(kernel=2, stride=2).to(device)

    x3 = torch.randn(1, 2, 3, device=device)
    with pytest.raises(ValueError):
        _ = layer(x3)

    x6 = torch.randn(1, 1, 1, 4, 2, device=device)
    with pytest.raises(ValueError):
        _ = layer(x6)
