import copy

import pytest
import torch

from convlogic import ConvLogicLayer, PoolLogicLayer, TreeLogicLayer


@pytest.mark.parametrize("stride,pad", [(1, 1), (2, 2)])
def test_convlogic_output_shape(impl_and_device, stride, pad):
    impl, device = impl_and_device
    h_in, w_in, kernel = 11 + pad, 11 + pad, 3

    layer = (
        ConvLogicLayer(
            in_channels=3,
            out_channels=2,
            stride=stride,
            padding=pad,
            implementation=impl,
            kernel=3,
            complete=True,
        )
        .to(device)
        .eval()
    )

    x = torch.randn(2, 3, h_in, w_in, device=device)
    y = layer(x)

    h_out = (h_in + 2 * pad - kernel) // stride + 1
    w_out = (w_in + 2 * pad - kernel) // stride + 1
    assert y.shape == (2, 2, h_out / 2, w_out / 2)


def test_convlogic_no_nans_infs(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(2, 2, implementation=impl, complete=True).to(device).eval()

    x = torch.randn(4, 2, 6, 6, device=device)
    y = layer(x)

    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_convlogic_range_0_1(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(1, 1, implementation=impl, complete=True).to(device).eval()

    x = torch.rand(1, 1, 4, 4, device=device)
    y = layer(x)

    assert (0.0 <= y).all() and (y <= 1.0).all()


def test_convlogic_deterministic(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(1, 1, padding=1, implementation=impl, complete=True).to(device).eval()

    x = torch.rand(3, 1, 6, 6, device=device)
    y1 = layer(x)
    y2 = layer(x)

    assert torch.allclose(y1, y2, atol=1e-6)


def test_convlogic_batch_support(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(1, 1, padding=1, implementation=impl, complete=True).to(device).eval()

    x = torch.randn(8, 1, 6, 6, device=device)
    y = layer(x)

    assert y.shape[0] == 8


def test_convlogic_small_input(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(1, 1, kernel=2, padding=0, implementation=impl, complete=True).to(device).eval()

    x = torch.randn(1, 1, 3, 3, device=device)
    y = layer(x)

    assert y.shape == (1, 1, 1, 1)


def test_convlogic_irregular_channels(impl_and_device):
    impl, device = impl_and_device
    layer = ConvLogicLayer(1, 5, implementation=impl, complete=True).to(device).eval()

    x = torch.randn(1, 1, 6, 6, device=device)
    y = layer(x)

    assert y.shape[1] == 5, f"Expected output channels to be 5, but got {y.shape[1]}"
    assert y.shape[-1] == 3, f"Expected output width to be 3, but got {y.shape[-1]}"

    layer = ConvLogicLayer(5, 1, implementation=impl, complete=True).to(device).eval()

    x = torch.randn(5, 5, 4, 4, device=device)
    y = layer(x)

    assert y.shape[1] == 1, f"Expected output channels to be 1, but got {y.shape[1]}"
    assert y.shape[-1] == 2, f"Expected output width to be 2, but got {y.shape[-1]}"


def test_convlogic_identity_weights(impl_and_device):
    impl, device = impl_and_device
    layer = (
        ConvLogicLayer(
            in_channels=2,
            out_channels=2,
            stride=1,
            padding=1,
            residual_init=True,
            implementation=impl,
            complete=True,
        )
        .to(device)
        .eval()
    )

    idx_conv = layer.weights[:, :4, :].argmax(dim=-1)
    idx_tree1 = layer.weights[:, 4:5, :].argmax(dim=-1)
    idx_tree2 = layer.weights[:, 6, :].argmax(dim=-1)

    assert torch.all(idx_conv == 3), f"ConvLogic: expected all 3, got\n{idx_conv}"
    assert torch.all(idx_tree1 == 3), f"TreeLogic1: expected all 3, got\n{idx_tree1}"
    assert torch.all(idx_tree2 == 3), f"TreeLogic2: expected all 3, got\n{idx_tree2}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for forward python-cuda match test")
@pytest.mark.parametrize("stride,pad", [(1, 1), (1, 0)])
def test_forward_python_cuda_match(stride, pad):
    torch.manual_seed(0)
    layer = ConvLogicLayer(
        in_channels=3,
        out_channels=2,
        stride=stride,
        padding=pad,
        kernel=3,
        residual_init=False,
        complete=True,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for split-vs-complete test")
@pytest.mark.parametrize("stride,pad", [(1, 1), (2, 2)])
def test_split_layers_vs_complete_convlogic(stride, pad):
    torch.manual_seed(0)

    full_layer = ConvLogicLayer(
        in_channels=3,
        out_channels=2,
        stride=stride,
        padding=pad,
        kernel=3,
        residual_init=False,
        implementation="cuda",
        complete=True,
        return_indices=True,
        return_intermediate=True,
    ).to("cuda")

    conv_layer = ConvLogicLayer(
        in_channels=3,
        out_channels=2,
        stride=stride,
        padding=pad,
        kernel=3,
        residual_init=False,
        implementation="cuda",
        complete=False,
    ).to("cuda")

    tree1_layer = TreeLogicLayer(
        channels=2,
        input_bits=4,
        implementation="cuda",
        residual_init=False,
    ).to("cuda")

    tree2_layer = TreeLogicLayer(
        channels=2,
        input_bits=2,
        implementation="cuda",
        residual_init=False,
    ).to("cuda")

    pool_layer = PoolLogicLayer(return_indices=True)

    conv_layer.selection = full_layer.selection
    conv_layer.weights.data.copy_(full_layer.weights[:, 0:4, :])
    tree1_layer.weights.data.copy_(full_layer.weights[:, 4:6, :])
    tree2_layer.weights.data.copy_(full_layer.weights[:, 6:7, :])

    x = torch.randn(4, 3, 10, 10, device="cuda")

    with torch.no_grad():
        y_full, x1_full, x2_full, indices_full = full_layer(x)
        x1_split = conv_layer(x)
        x2_split = tree1_layer(x1_split)
        x3_split = tree2_layer(x2_split)
        y_split, indices_split = pool_layer(x3_split)

    indices_split = indices_split.flatten(start_dim=2)

    max_diff_x1 = (x1_split - x1_full).abs().max().item()
    max_diff_x2 = (x2_split - x2_full).abs().max().item()
    max_diff_y = (y_split - y_full).abs().max().item()

    assert torch.equal(indices_split, indices_full), "Mismatch in indices"
    assert torch.allclose(x1_split, x1_full, atol=1e-6), f"x1 mismatch (max diff: {max_diff_x1:.2e})"
    assert torch.allclose(x2_split, x2_full, atol=1e-6), f"x2 mismatch (max diff: {max_diff_x2:.2e})"
    assert torch.allclose(y_split, y_full, atol=1e-6), f"y mismatch (max diff: {max_diff_y:.2e})"
