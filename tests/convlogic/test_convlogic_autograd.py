import pytest
import torch
from torch.autograd import gradcheck

from convlogic.convlogic_layer import ConvLogicFunction, ConvLogicLayer

if not torch.cuda.is_available():
    pytest.skip("CUDA required for ConvLogic tests", allow_module_level=True)


@pytest.mark.parametrize("stride,pad,kernel", [((1, 2), (1, 2), (1, 2)), ((3, 1), (3, 1), (3, 5)), (2, 2, 3)])
def test_convlogic_smoke(stride, pad, kernel):
    layer = ConvLogicLayer(
        in_channels=1, out_channels=1, stride=stride, padding=pad, groups=1, kernel=kernel, residual_init=False
    ).to("cuda")

    x = torch.rand(1, 1, 5, 5, device="cuda", requires_grad=True)
    x = torch.sigmoid(x)
    x.retain_grad()

    y = layer(x)
    (y.sum()).backward()

    # check that gradients exist and shapes match
    assert x.grad is not None and x.grad.shape == x.shape
    assert layer.weights.grad is not None and layer.weights.grad.shape == layer.weights.shape


@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("c_in, c_out", [(1, 1), (3, 1), (1, 3)])
@pytest.mark.parametrize("stride, pad, kernel", [(1, 1, 3), (2, 2, 5)])
def test_convlogic_weight_gradcheck(batch, c_in, c_out, stride, pad, kernel):
    layer = (
        ConvLogicLayer(
            in_channels=c_in,
            out_channels=c_out,
            stride=stride,
            padding=pad,
            groups=1,
            kernel=kernel,
            residual_init=False,
        )
        .double()
        .to("cuda")
    )

    x = torch.rand(batch, c_in, 5, 5, dtype=torch.double, device="cuda", requires_grad=False)
    x = torch.sigmoid(x)

    w = layer.weights.clone().detach().double().to("cuda").requires_grad_(True)
    sel = layer.selection

    def fn(input_x, logits_w):
        w_act = torch.nn.functional.softmax(logits_w, dim=-1)
        return ConvLogicFunction.apply(input_x, w_act, sel, stride, stride, pad, pad, kernel, kernel)

    assert gradcheck(fn, (x, w), eps=1e-6, atol=1e-4, nondet_tol=1e-5), (
        f"ConvLogic gradcheck failed for stride={stride}, pad={pad}, kernel={kernel}."
    )


@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("c_in, c_out", [(1, 1), (3, 1), (1, 3)])
@pytest.mark.parametrize("stride, pad, kernel", [(1, 1, 3), (2, 2, 5)])
def test_convlogic_input_gradcheck(batch, c_in, c_out, stride, pad, kernel):
    layer = (
        ConvLogicLayer(
            in_channels=c_in,
            out_channels=c_out,
            stride=stride,
            padding=pad,
            groups=1,
            kernel=kernel,
            residual_init=False,
        )
        .double()
        .to("cuda")
    )

    # This time, x must require gradients
    x = torch.rand(batch, c_in, 5, 5, dtype=torch.double, device="cuda", requires_grad=True)

    # Fixed, detached weights (no grad)
    w_logits = layer.weights.clone().detach().double().to("cuda")
    w_act = torch.nn.functional.softmax(w_logits, dim=-1)

    sel = layer.selection

    def fn(input_x):
        return ConvLogicFunction.apply(input_x, w_act, sel, stride, stride, pad, pad, kernel, kernel)

    assert gradcheck(fn, (x,), eps=1e-6, atol=1e-4, nondet_tol=1e-5), (
        f"ConvLogic input gradcheck failed for stride={stride}, pad={pad}, kernel={kernel}."
    )
