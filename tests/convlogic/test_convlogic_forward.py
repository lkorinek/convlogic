import copy

import pytest
import torch

from convlogic import ConvLogicLayer


@pytest.mark.parametrize("stride,pad,kernel", [((1, 2), (1, 2), (1, 2)), ((3, 1), (3, 1), (3, 5)), (2, 2, 3)])
def test_convlogic_output_shape(impl_and_device, stride, pad, kernel):
    impl, device = impl_and_device
    h_in, w_in, kernel = 5, 5, 3

    layer = (
        ConvLogicLayer(in_channels=3, out_channels=2, stride=stride, padding=pad, implementation=impl, kernel=3)
        .to(device)
        .eval()
    )

    x = torch.randn(2, 3, h_in, w_in, device=device)
    y = layer(x)

    stride = (stride, stride) if isinstance(stride, int) else stride
    pad = (pad, pad) if isinstance(pad, int) else pad
    kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
    h_out = (h_in + 2 * pad[0] - kernel[0]) // stride[0] + 1
    w_out = (w_in + 2 * pad[1] - kernel[1]) // stride[1] + 1
    assert y.shape == (2, 2, h_out, w_out, 4)


def test_convlogic_no_nans_infs(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(2, 2, implementation=impl).to(device).eval()

    x = torch.randn(4, 2, 6, 6, device=device)
    y = layer(x)

    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_convlogic_range_0_1(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(1, 1, implementation=impl).to(device)

    x = torch.rand(1, 1, 4, 4, device=device)
    y = layer(x)

    assert (y >= 0.0).all() and (y <= 1.0).all()


def test_convlogic_0_1(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(1, 1, implementation=impl).to(device).eval()
    x = torch.randint(0, 2, (1, 1, 4, 4), device=device, dtype=torch.float32)
    y = layer(x)

    assert torch.all((y == 0.0) | (y == 1.0)), "Expected all values in y to be 0.0 or 1.0"


def test_convlogic_deterministic(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(1, 1, implementation=impl).to(device).eval()

    x = torch.rand(3, 1, 7, 7, device=device)
    y1 = layer(x)
    y2 = layer(x)

    assert torch.allclose(y1, y2, atol=1e-6)


def test_convlogic_batch_support(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(1, 1, implementation=impl).to(device).eval()

    x = torch.randn(8, 1, 5, 5, device=device)
    y = layer(x)

    assert y.shape[0] == 8


def test_convlogic_small_input(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(1, 1, implementation=impl).to(device).eval()

    x = torch.randn(1, 1, 3, 3, device=device)
    y = layer(x)

    assert y.shape == (1, 1, 3, 3, 4)


def test_convlogic_irregular_channels(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(5, 5, implementation=impl).to(device).eval()

    x = torch.randn(1, 5, 4, 4, device=device)
    y = layer(x)

    assert y.shape == (1, 5, 4, 4, 4)


def test_convlogic_identity_weights(impl_and_device):
    impl, device = impl_and_device
    layer = (
        ConvLogicLayer(in_channels=2, out_channels=2, stride=1, padding=1, residual_init=True, implementation=impl)
        .to(device)
        .eval()
    )

    # with residual_init, all 4 positions per channel should pick gate 3
    idx = layer.weights.argmax(dim=-1)  # shape [out_channels,4]
    assert torch.all(idx == 3), f"expected all branches→3, got\n{idx}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for forward python-cuda match test")
@pytest.mark.parametrize("stride,pad,kernel", [((1, 2), (1, 2), (1, 2)), ((3, 1), (3, 1), (3, 5)), (2, 2, 3)])
def test_forward_python_cuda_match(stride, pad, kernel):
    torch.manual_seed(0)
    layer = ConvLogicLayer(
        in_channels=3, out_channels=2, stride=stride, padding=pad, kernel=kernel, residual_init=False
    ).to("cuda")
    layer_py = copy.deepcopy(layer)
    layer_cuda = layer

    layer_py.implementation = "python"
    layer_cuda.implementation = "cuda"
    layer_py.eval()
    layer_cuda.eval()

    x_cuda = torch.randn(4, 3, 8, 8, device="cuda")
    x_py = copy.deepcopy(x_cuda)

    y_cuda = layer_cuda(x_cuda)
    y_py = layer_py(x_py)

    assert torch.allclose(y_py, y_cuda, atol=1e-6), (
        f"python vs cuda forward mismatch:\n max diff = {(y_py - y_cuda).abs().max().item():.3e}"
    )
