import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from difflogic.difflogic import LogicLayer, LogicLayerCudaFunction

if not torch.cuda.is_available():
    pytest.skip("CUDA required for Difflogic tests", allow_module_level=True)


@pytest.mark.parametrize("in_dim,out_dim", [(8, 8), (12, 16)])
def test_difflogic_smoke(in_dim, out_dim):
    """
    Smoke test: forward + backward for LogicLayer under various in/out dims.
    """
    layer = LogicLayer(in_dim=in_dim, out_dim=out_dim, implementation="cuda", residual_init=False).to("cuda")

    x = torch.randn(32, in_dim, device="cuda", requires_grad=True)
    x = torch.sigmoid(x)
    x.retain_grad()

    y = layer(x)
    (y.sum()).backward()

    # check that gradients exist and shapes match
    assert x.grad is not None and x.grad.shape == x.shape
    assert layer.weights.grad is not None and layer.weights.grad.shape == layer.weights.shape


@pytest.mark.parametrize("in_dim,out_dim", [(8, 8), (12, 16)])
def test_difflogic_gradcheck(in_dim, out_dim):
    """
    Gradcheck for LogicLayerCudaFunction under various in/out dims,
    including the softmax activation on the raw logits.
    """
    layer = LogicLayer(in_dim=in_dim, out_dim=out_dim, implementation="cuda", residual_init=False).double().to("cuda")

    x = torch.randn(16, in_dim, dtype=torch.double, device="cuda", requires_grad=False)
    x = torch.sigmoid(x)

    w = layer.weights.clone().detach().double().to("cuda").requires_grad_(True)
    a, b = layer.indices_0, layer.indices_1
    gs = layer.given_x_indices_of_y_start
    gy = layer.given_x_indices_of_y

    def fn(input_x, logits_w):
        w_act = F.softmax(logits_w, dim=-1)
        out = LogicLayerCudaFunction.apply(input_x.T.contiguous(), a, b, w_act, gs, gy)
        return out.T

    assert gradcheck(fn, (x, w), eps=1e-6, atol=1e-4), (
        f"Difflogic gradcheck failed for in_dim={in_dim}, out_dim={out_dim}"
    )
