import importlib

import numpy as np
import torch

from .functional import GradFactor, bin_op_s, get_unique_connections
from .packbitstensor import PackBitsTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

difflogic_cuda = None
if device.type == "cuda":
    difflogic_cuda = importlib.import_module("difflogic_cuda")


class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Implements a differentiable logic layer
    where each output is computed using a differentiable logic gate applied to two inputs.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grad_factor: float = 1.0,
        implementation: str = None,
        connections: str = "random",
        residual_init: bool = False,
    ):
        """
        :param in_dim:              Input dimensionality of the layer.
        :param out_dim:             Output dimensionality of the layer.
        :param grad_factor:         For deep models (>6 layers), increasing this (e.g., 2.0)
                                    helps prevent vanishing gradients.
        :param implementation:      One of {'cuda', 'python'}. Determines which backend is used
                                    for evaluation and training.
        :param connections:         Method for initializing logic gate connectivity {'random', 'unique'}.
        :param residual_init:       If True, initializes weights to favor A gate (90%) for residual behavior.
        """

        super().__init__()
        if not residual_init:
            self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16))
        else:
            weights = torch.zeros(out_dim, 16)
            indices = torch.full((out_dim, 1), 3)
            weights.scatter_(1, indices, 5)
            self.weights = torch.nn.Parameter(weights, requires_grad=True)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grad_factor = grad_factor

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks
        2. To provide a CPU implementation of differentiable logic gate networks
        """
        if implementation is None:
            implementation = "cuda" if torch.cuda.is_available() else "python"

        assert implementation in ["cuda", "python"], f"Invalid implementation: {implementation}"
        self.implementation = implementation

        self.connections = connections
        assert self.connections in ["random", "unique"], self.connections
        a, b = self.get_connections(self.connections)
        self.register_buffer("indices_0", a.to(torch.int64))
        self.register_buffer("indices_1", b.to(torch.int64))

        if self.implementation == "cuda":
            # Defining additional indices for improving the efficiency of the backward of the CUDA implementation.
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices_0.cpu().numpy()
            indices_1_np = self.indices_1.cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)

            self.register_buffer(
                "given_x_indices_of_y_start",
                torch.tensor(np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), dtype=torch.int64),
            )
            self.register_buffer(
                "given_x_indices_of_y",
                torch.tensor([item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64),
            )

        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x: torch.Tensor | PackBitsTensor) -> torch.Tensor | PackBitsTensor:
        if isinstance(x, PackBitsTensor):
            assert not self.training, "PackBitsTensor is not supported for the differentiable training mode."
            assert x.is_cuda, (
                f"PackBitsTensor is only supported for CUDA, but got device {x.device}. "
                "If you want fast inference on CPU, please use CompiledDiffLogicModel."
            )
        else:
            if self.grad_factor != 1.0:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == "cuda":
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == "python":
            return self.forward_python(x)
        else:
            raise ValueError(f"Unsupported implementation: {self.implementation}")

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        a, b = x[..., self.indices_0], x[..., self.indices_1]

        if self.training:
            w_prob = torch.nn.functional.softmax(self.weights, dim=-1)
        else:
            w_prob = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)

        y = bin_op_s(a, b, w_prob)
        return y

    def forward_cuda(self, x):
        if self.training:
            assert x.is_cuda, f"x must be a CUDA tensor, got device: {x.device}"
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices_0, self.indices_1

        if self.training:
            w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)

    def forward_cuda_eval(self, x: PackBitsTensor):
        """
        WARNING: this is an in-place operation.

        :param x:
        :return:
        """
        assert not self.training
        assert isinstance(x, PackBitsTensor)
        assert x.t.shape[0] == self.in_dim, (x.t.shape, self.in_dim)

        a, b = self.indices_0, self.indices_1
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)

        return x

    def extra_repr(self):
        return "{}, {}, {}".format(self.in_dim, self.out_dim, "train" if self.training else "eval")

    def get_connections(self, connections):
        assert self.out_dim * 2 >= self.in_dim, (
            "The number of neurons ({}) must not be smaller than half of the "
            "number of inputs ({}) because otherwise not all inputs could be "
            "used or considered.".format(self.out_dim, self.in_dim)
        )
        if connections == "random":
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            return a, b
        elif connections == "unique":
            return get_unique_connections(self.in_dim, self.out_dim)
        else:
            raise ValueError(connections)


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """

    def __init__(self, k: int, tau: float = 1.0):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        """
        super().__init__()
        self.k = k
        self.tau = tau

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return "k={}, tau={}".format(self.k, self.tau)


class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None, None
