import pytest
import torch

from convlogic.treelogic_layer import TreeLogicLayer
from difflogic.functional import bin_op


def test_treelogic_output_shape(impl_and_device):
    impl, device = impl_and_device
    layer = TreeLogicLayer(channels=3, input_bits=2, implementation=impl).to(device)
    x = torch.rand(2, 3, 1, 1, 2, device=device)
    y = layer(x)
    assert y.shape == (2, 3, 1, 1, 1)


def test_treelogic_no_nans_infs(impl_and_device):
    impl, device = impl_and_device
    layer = TreeLogicLayer(channels=2, input_bits=2, implementation=impl).to(device)
    x = torch.rand(4, 2, 1, 1, 2, device=device)
    y = layer(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_treelogic_range_0_1(impl_and_device):
    impl, device = impl_and_device
    layer = TreeLogicLayer(channels=1, input_bits=2, implementation=impl).to(device)
    x = torch.rand(1, 1, 1, 1, 2, device=device)
    y = layer(x)
    assert (0.0 <= y).all() and (y <= 1.0).all()


def test_treelogic_deterministic(impl_and_device):
    impl, device = impl_and_device
    layer = TreeLogicLayer(channels=2, input_bits=2, implementation=impl).to(device)
    x = torch.rand(3, 2, 1, 1, 2, device=device)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def test_treelogic_batch_support(impl_and_device):
    impl, device = impl_and_device
    layer = TreeLogicLayer(channels=1, input_bits=2, implementation=impl).to(device)
    x = torch.rand(8, 1, 1, 1, 2, device=device)
    y = layer(x)
    assert y.shape[0] == 8


def test_treelogic_small_input(impl_and_device):
    impl, device = impl_and_device
    layer = TreeLogicLayer(channels=1, input_bits=2, implementation=impl).to(device)
    x = torch.rand(1, 1, 1, 1, 2, device=device)
    y = layer(x)
    assert y.shape == (1, 1, 1, 1, 1)


def test_treelogic_irregular_input(impl_and_device):
    impl, device = impl_and_device
    layer = TreeLogicLayer(channels=5, input_bits=4, implementation=impl).to(device)
    x = torch.rand(1, 5, 1, 1, 4, device=device)
    y = layer(x)
    assert y.shape == (1, 5, 1, 1, 2)


def test_all_logic_gates(impl_and_device):
    impl, device = impl_and_device

    layer = TreeLogicLayer(channels=16, input_bits=8, implementation=impl).to(device)
    weights = torch.full((16, 4, 16), float("-inf"), device=device, dtype=torch.float32)
    for gate in range(16):
        weights[gate, :, gate] = 0.0
    with torch.no_grad():
        layer.weights.data.copy_(weights)

    flat = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    x = torch.tensor([flat], dtype=torch.float, device=device).reshape(1, 1, 1, 1, 8)
    x = x.expand(-1, 16, -1, -1, -1)
    y = layer(x)

    a = torch.tensor(flat[::2], device=device)
    b = torch.tensor(flat[1::2], device=device)
    expected = torch.stack([bin_op(a, b, gate) for gate in range(16)], dim=0)
    expected = expected.view(1, 16, 1, 1, 4)

    assert torch.equal(y, expected), f"\nExpected:\n{expected}\nGot:\n{y}"
