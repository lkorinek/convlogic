import importlib

import torch
from torch import nn
from torch.autograd import Function

from difflogic.functional import GradFactor, bin_op_s
from difflogic.packbitstensor import PackBitsTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

convlogic_cuda = None
if device.type == "cuda":
    convlogic_cuda = importlib.import_module("convlogic_cuda")


class TreeLogicFunction(Function):
    @staticmethod
    def forward(ctx, x, w_prob):
        x = x.contiguous()
        y = convlogic_cuda.treelogic_forward(x, w_prob)
        ctx.save_for_backward(x, w_prob)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        grad_y = grad_y.contiguous()
        x, w_prob = ctx.saved_tensors

        grad_x = convlogic_cuda.treelogic_backward_x(grad_y, x, w_prob)
        grad_w_prob = convlogic_cuda.treelogic_backward_weight(grad_y, x)
        return grad_x, grad_w_prob


class TreeLogicLayer(nn.Module):
    """
    A tree-structured differentiable logic layer.

    This layer takes multiple logic inputs and combines them using binary logic gates.
    It performs one level of pairwise reduction, reducing the number of inputs by half.
    By stacking multiple layers, you can build a full tree that combines many logic outputs step by step.

    Inputs       Logic Gate       Output
    0 ──┐                       ┌── 0
        │─▶ ● ───────────────▶ │
    1 ──┘                       └──

    2 ──┐                       ┌── 1
        │─▶ ● ───────────────▶ │
    3 ──┘                       └──

    Each logic gate (●) combines two inputs (input_bits=4) into one output using a learned binary function.
    """

    def __init__(
        self,
        channels: int,
        input_bits: int,
        residual_init: bool = False,
        implementation: str = "cuda",
        grad_factor: float = 1.0,
    ):
        """
        :param channels:         Number of input/output channels.
        :param input_bits:       Number of logic inputs per output (e.g., 4 inputs = 2 logic gates).
        :param residual_init:    If True, initializes weights to favor A gate (90%) (helps stabilize training).
        :param implementation:   Which backend to use: 'cuda' for GPU (fast), 'python' for CPU/debugging.
        :param grad_factor:      A multiplier applied to the input gradients.
        """

        super().__init__()
        assert input_bits % 2 == 0, "input_bits must be even"
        self.channels = channels
        self.input_bits = input_bits
        self.implementation = implementation
        self.grad_factor = grad_factor

        out_bits = input_bits // 2
        if residual_init:
            w = torch.zeros(channels, out_bits, 16)
            w[..., 3] = 5.0
            self.weights = nn.Parameter(w)
        else:
            self.weights = nn.Parameter(torch.randn(channels, out_bits, 16))

    def forward(self, x):
        assert x.shape[-1] == self.input_bits, (
            f"Expected input last dimension to be {self.input_bits}, but got {x.shape[-1]}"
        )

        x = x.contiguous()

        if self.grad_factor != 1.0 and not isinstance(x, PackBitsTensor):
            x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == "cuda":
            return self.forward_cuda(x)
        elif self.implementation == "python":
            return self.forward_python(x)
        else:
            raise ValueError(f"Unknown implementation {self.implementation!r}")

    def forward_cuda(self, x):
        if self.training:
            w_prob = torch.nn.functional.softmax(self.weights, dim=-1)
        else:
            idx = self.weights.argmax(-1)
            w_prob = torch.nn.functional.one_hot(idx, num_classes=16).to(dtype=x.dtype)

        w_prob = w_prob.contiguous()
        return TreeLogicFunction.apply(x, w_prob)

    def forward_python(self, x):
        _, c, _, _, d = x.shape
        x_out = d // 2

        a = x[..., 0::2]
        b = x[..., 1::2]

        if self.training:
            w_prob = torch.nn.functional.softmax(self.weights, dim=-1)
        else:
            idx = self.weights.argmax(-1)
            w_prob = torch.nn.functional.one_hot(idx, num_classes=16).to(x.dtype)

        w_prob = w_prob.view(1, c, 1, 1, x_out, 16)
        y = bin_op_s(a, b, w_prob)

        return y

    def extra_repr(self):
        return f"channels={self.channels}, input_bits={self.input_bits}"
