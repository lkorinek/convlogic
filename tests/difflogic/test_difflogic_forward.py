import pytest
import torch

from difflogic import LogicLayer
from difflogic.functional import bin_op


def test_logiclayer_output_shape(impl_and_device):
    impl, device = impl_and_device
    layer = LogicLayer(in_dim=3, out_dim=5, implementation=impl).to(device)
    x = torch.rand(2, 3, device=device)
    y = layer(x)
    assert y.shape == (2, 5)


def test_logiclayer_no_nans_infs(impl_and_device):
    impl, device = impl_and_device
    layer = LogicLayer(in_dim=4, out_dim=2, implementation=impl).to(device)
    x = torch.rand(4, 4, device=device)
    y = layer(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_logiclayer_range_0_1(impl_and_device):
    impl, device = impl_and_device
    layer = LogicLayer(in_dim=3, out_dim=4, implementation=impl).to(device)
    x = torch.rand(1, 3, device=device)
    y = layer(x)
    assert (0.0 <= y).all() and (y <= 1.0).all()


def test_logiclayer_0_1(impl_and_device):
    impl, device = impl_and_device
    layer = LogicLayer(in_dim=3, out_dim=4, implementation=impl).to(device).eval()
    x = torch.randint(0, 2, (1, 3), device=device, dtype=torch.float32)
    y = layer(x)
    assert torch.all((y == 0.0) | (y == 1.0)), "Expected all values in y to be 0.0 or 1.0"


def test_logiclayer_deterministic(impl_and_device):
    impl, device = impl_and_device
    layer = LogicLayer(in_dim=3, out_dim=4, implementation=impl).to(device)
    x = torch.rand(3, 3, device=device)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def test_logiclayer_batch_support(impl_and_device):
    impl, device = impl_and_device
    layer = LogicLayer(in_dim=2, out_dim=3, implementation=impl).to(device)
    x = torch.rand(8, 2, device=device)
    y = layer(x)
    assert y.shape[0] == 8


def test_logiclayer_small_input(impl_and_device):
    impl, device = impl_and_device
    layer = LogicLayer(in_dim=1, out_dim=1, implementation=impl).to(device)
    x = torch.rand(1, 1, device=device)
    y = layer(x)
    assert y.shape == (1, 1)


def test_logiclayer_irregular_input(impl_and_device):
    impl, device = impl_and_device
    layer = LogicLayer(in_dim=5, out_dim=3, implementation=impl).to(device)
    x = torch.rand(1, 5, device=device)
    y = layer(x)
    assert y.shape == (1, 3)


def test_all_logic_gates(impl_and_device):
    impl, device = impl_and_device

    layer = LogicLayer(in_dim=8, out_dim=16 * 4, implementation=impl).to(device)

    idx0 = torch.tensor([0, 2, 4, 6], dtype=torch.int64, device=device).repeat(16)
    idx1 = torch.tensor([1, 3, 5, 7], dtype=torch.int64, device=device).repeat(16)
    layer.indices_0.copy_(idx0)
    layer.indices_1.copy_(idx1)

    weights = torch.full((16 * 4, 16), float("-inf"), device=device, dtype=torch.float32)
    for gate in range(16):
        for sub in range(4):
            weights[gate * 4 + sub, gate] = 0.0
    with torch.no_grad():
        layer.weights.copy_(weights)

    flat = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    x = torch.tensor([flat], dtype=torch.float, device=device)
    y = layer(x)

    a = torch.tensor(flat[::2], device=device)
    b = torch.tensor(flat[1::2], device=device)
    gate_outputs = [bin_op(a, b, gate) for gate in range(16)]
    expected = torch.stack(gate_outputs, dim=0).view(1, -1)

    assert torch.equal(y, expected), f"\nExpected:\n{expected}\nGot:\n{y}"
