import pytest
import torch
from torch.autograd import gradcheck

from convlogic.treelogic_layer import TreeLogicFunction

if not torch.cuda.is_available():
    pytest.skip("CUDA required for TreeLogic tests", allow_module_level=True)


@pytest.mark.parametrize("input_bits", [2, 4])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("height, width", [(1, 1), (3, 3)])
def test_treelogic_smoke(batch, channels, height, width, input_bits):
    x = torch.rand(batch, channels, height, width, input_bits, device="cuda", requires_grad=True)
    x = torch.sigmoid(x)
    x.retain_grad()

    w_logits = torch.rand(channels, input_bits // 2, 16, device="cuda", dtype=x.dtype, requires_grad=True)
    w_prob = torch.nn.functional.softmax(w_logits, dim=-1)

    y = TreeLogicFunction.apply(x, w_prob)
    (y.sum()).backward()

    assert x.grad is not None and x.grad.shape == x.shape
    assert w_logits.grad is not None and w_logits.grad.shape == w_logits.shape


@pytest.mark.parametrize("input_bits", [2, 4])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("height, width", [(1, 1), (3, 3)])
def test_treelogic_weight_gradcheck(batch, channels, height, width, input_bits):
    x = torch.rand(batch, channels, height, width, input_bits, dtype=torch.double, device="cuda", requires_grad=False)
    x = torch.sigmoid(x)

    w_logits = torch.rand(channels, input_bits // 2, 16, dtype=torch.double, device="cuda").requires_grad_(True)

    def fn(input_x, logits_w):
        w_prob = torch.nn.functional.softmax(logits_w, dim=-1)
        return TreeLogicFunction.apply(input_x, w_prob)

    assert gradcheck(fn, (x, w_logits), eps=1e-6, atol=1e-4, nondet_tol=1e-5), (
        f"TreeLogic weight gradcheck failed for input_bits={input_bits}, channels={channels}"
    )


@pytest.mark.parametrize("input_bits", [2, 4])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("height, width", [(1, 1), (3, 3)])
def test_treelogic_input_gradcheck(batch, channels, height, width, input_bits):
    x = torch.rand(batch, channels, height, width, input_bits, dtype=torch.double, device="cuda", requires_grad=True)

    w_logits = torch.rand(channels, input_bits // 2, 16, dtype=torch.double, device="cuda")
    w_prob = torch.nn.functional.softmax(w_logits, dim=-1)

    def fn(input_x):
        return TreeLogicFunction.apply(input_x, w_prob)

    assert gradcheck(fn, (x,), eps=1e-6, atol=1e-4, nondet_tol=1e-5), (
        f"TreeLogic input gradcheck failed for input_bits={input_bits}, channels={channels}"
    )
