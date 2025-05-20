import importlib

import numpy as np
import torch
from torch import nn
from torch.autograd import Function

from difflogic.functional import GradFactor, bin_op_s
from difflogic.packbitstensor import PackBitsTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

convlogic_cuda = None
if device.type == "cuda":
    convlogic_cuda = importlib.import_module("convlogic_cuda")


class ConvLogicFunction(Function):
    @staticmethod
    def forward(ctx, x, w_prob, selection, stride_h, stride_w, pad_h, pad_w, kernel_h, kernel_w):
        x = x.contiguous()
        y = convlogic_cuda.convlogic_forward(x, w_prob, selection, stride_h, stride_w, pad_h, pad_w, kernel_h, kernel_w)
        ctx.save_for_backward(x, w_prob, selection)
        ctx.stride = (stride_h, stride_w)
        ctx.padding = (pad_h, pad_w)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        grad_y = grad_y.contiguous()
        x, w_prob, selection = ctx.saved_tensors

        grad_x = convlogic_cuda.convlogic_backward_x(grad_y, x, w_prob, selection, *ctx.stride, *ctx.padding)
        grad_w_prob = convlogic_cuda.convlogic_backward_weight(grad_y, x, selection, *ctx.stride, *ctx.padding)
        return grad_x, grad_w_prob, None, None, None, None, None, None, None


class FullConvLogicFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x,
        w_prob,
        selection,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        kernel_h,
        kernel_w,
        return_intermediate=False,
        return_indices=False,
    ):
        x1 = x.contiguous()
        y, x2, x3, indices = convlogic_cuda.full_convlogic_forward(
            x1, w_prob, selection, stride_h, stride_w, pad_h, pad_w, kernel_h, kernel_w
        )
        ctx.save_for_backward(x1, x2, x3, w_prob, selection, indices)
        ctx.stride = (stride_h, stride_w)
        ctx.padding = (pad_h, pad_w)

        if return_intermediate:
            if return_indices:
                return y, x2, x3, indices
            else:
                return y, x2, x3
        else:
            if return_indices:
                return y, indices
            else:
                return y

    @staticmethod
    def backward(ctx, grad_y):
        grad_y = grad_y.contiguous()
        x1, x2, x3, w_prob, selection, indices = ctx.saved_tensors

        grad_x1, grad_x2, grad_x3 = convlogic_cuda.full_convlogic_backward_x(
            grad_y, x1, x2, x3, w_prob, indices, selection, *ctx.stride, *ctx.padding
        )
        grad_w_prob = convlogic_cuda.full_convlogic_backward_weight(
            grad_x2, grad_x3, grad_y, x1, x2, x3, indices, selection, *ctx.stride, *ctx.padding
        )
        return grad_x1, grad_w_prob, None, None, None, None, None, None, None, None, None


class ConvLogicLayer(nn.Module):
    """
    A logic-based convolutional layer that replaces dot-product kernels with logic gates.

    This module selects 8 input positions per output channel and applies 4 (or 7 (4 -> 2 -> 1), if `complete=True`)
    differentiable logic gates to them. It supports both a standalone convolution-style logic stage,
    or a complete logic pipeline with tree-style reductions and pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        kernel: int | tuple[int, int] = 3,
        groups: int = 1,
        selection: torch.Tensor | None = None,
        residual_init: bool = False,
        implementation: str = "cuda",
        grad_factor: float = 1.0,
        complete: bool = False,
        return_intermediate: bool = False,
        return_indices: bool = False,
    ):
        """
        :param in_channels:         Number of input channels.
        :param out_channels:        Number of output channels (each corresponds to one logic gate kernel).
        :param stride:              Stride of the logic kernel, given as a single int or a (height, width) tuple.
        :param padding:             Padding added to the input before logic operations.
        :param kernel:              Size of the logic kernel, specified as a single int or (height, width) tuple.
        :param groups:              Number of groups to split the input channels into for grouped logic operations.
        :param selection:           Optional[Tensor]. A [out_channels, 8] tensor encoding the packed (channel, row, col)
                                    positions for logic gate inputs. If None, a random selection is generated.
        :param residual_init:       If True, initializes weights to favor A gate (90%) (useful for training stability).
        :param implementation:      Which backend to use: either 'cuda' (fast) or 'python'
                                    (CPU, but training is not supported).
        :param grad_factor:         A multiplier applied to the input gradients.
        :param complete:            If True, runs the full 3-stage logic pipeline: ConvLogic → TreeLogic1 → TreeLogic2.
        :param return_intermediate: If True, the CUDA backend returns intermediate outputs from each stage
                                    (only supported in complete mode).
        :param return_indices:      If True, the CUDA backend returns the pooling indices used
                                    (for debugging, only in complete mode).
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        self.groups = groups
        self.grad_factor = grad_factor
        self.complete = complete
        self.return_intermediate = return_intermediate
        self.return_indices = return_indices
        self.input_bits = 8

        self.implementation = implementation
        if implementation == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA implementation requested, but CUDA is not available.")

        total_gates_per_kernel = 7 if self.complete else 4
        if residual_init:
            w = torch.zeros(out_channels, total_gates_per_kernel, 16)
            w[..., 3] = 5.0
            self.weights = nn.Parameter(w)
        else:
            self.weights = nn.Parameter(torch.randn(out_channels, total_gates_per_kernel, 16))

        if selection is None:
            selection = self.generate_selection()
        self.register_buffer("selection", selection)

        assert selection.shape == (out_channels, 8)
        assert self.weights.shape[-1] == 16, f"Expected 16 logic functions, got {self.weights.shape[-1]}"

    def forward(self, x):
        x = x.contiguous()

        if self.grad_factor != 1.0 and not isinstance(x, PackBitsTensor):
            x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == "cuda":
            return self.forward_cuda(x)
        elif self.implementation == "python":
            return self.forward_python_full(x) if self.complete else self.forward_python(x)
        else:
            raise ValueError(f"Unknown implementation {self.implementation!r}")

    def unpack_selection(self, selection: torch.Tensor):
        ch = (selection >> 16) & 0xFFFF
        ry = (selection >> 8) & 0xFF
        rx = selection & 0xFF
        return ch, ry, rx

    def forward_python(self, x):
        """
        Pure-Python ConvLogic forward using selection + bin_op_s.
        returns: y: [N, C_out, H_out, W_out, 4]
        """
        n, _, h_in, w_in = x.shape
        c_out, _ = self.selection.shape
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        k_h, k_w = self.kernel

        h_out = (h_in + 2 * pad_h - k_h) // stride_h + 1
        w_out = (w_in + 2 * pad_w - k_w) // stride_w + 1

        if self.training:
            w_prob = torch.nn.functional.softmax(self.weights, dim=-1)
        else:
            idx = self.weights.argmax(-1)
            w_prob = torch.nn.functional.one_hot(idx, num_classes=16).to(self.weights)
        w_prob = w_prob.view(1, c_out, 1, 1, 4, 16)

        a = x.new_zeros((n, c_out, h_out, w_out, 4))
        b = x.new_zeros((n, c_out, h_out, w_out, 4))

        for p in range(4):
            i1, i2 = 2 * p, 2 * p + 1
            ch1, ry1, rx1 = self.unpack_selection(self.selection[:, i1])
            ch2, ry2, rx2 = self.unpack_selection(self.selection[:, i2])

            for oy in range(h_out):
                for ox in range(w_out):
                    iy1 = oy * stride_h + ry1 - pad_h
                    ix1 = ox * stride_w + rx1 - pad_w
                    iy2 = oy * stride_h + ry2 - pad_h
                    ix2 = ox * stride_w + rx2 - pad_w

                    valid1 = (iy1 >= 0) & (iy1 < h_in) & (ix1 >= 0) & (ix1 < w_in)
                    valid2 = (iy2 >= 0) & (iy2 < h_in) & (ix2 >= 0) & (ix2 < w_in)

                    for n_idx in range(n):
                        for c in range(c_out):
                            x1 = x[n_idx, ch1[c], iy1[c], ix1[c]] if valid1[c] else 0.0
                            x2 = x[n_idx, ch2[c], iy2[c], ix2[c]] if valid2[c] else 0.0
                            a[n_idx, c, oy, ox, p] = x1
                            b[n_idx, c, oy, ox, p] = x2

        y = bin_op_s(a, b, w_prob)

        return y

    def forward_python_full(self, x):
        """
        Pure-Python full forward for 3-stage pipeline:
        ConvLogic → TreeLogic1 → TreeLogic2
        All logic stages use the same self.weights.
        """
        w_conv = self.weights[:, 0:4, :]
        w_tree1 = self.weights[:, 4:6, :]
        w_tree2 = self.weights[:, 6:7, :]

        x = x.contiguous()
        n, _, h_in, w_in = x.shape
        c_out, _ = self.selection.shape
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        k_h, k_w = self.kernel

        h_out = (h_in + 2 * pad_h - k_h) // stride_h + 1
        w_out = (w_in + 2 * pad_w - k_w) // stride_w + 1

        # --- Stage 1: ConvLogic ---
        if self.training:
            w_prob = torch.nn.functional.softmax(w_conv, dim=-1)
        else:
            idx = w_conv.argmax(-1)
            w_prob = torch.nn.functional.one_hot(idx, num_classes=16).to(x.dtype)
        w_prob_stage1 = w_prob.view(1, c_out, 1, 1, 4, 16)

        a1 = x.new_zeros((n, c_out, h_out, w_out, 4))
        b1 = x.new_zeros((n, c_out, h_out, w_out, 4))

        for p in range(4):
            i1, i2 = 2 * p, 2 * p + 1
            ch1, ry1, rx1 = self.unpack_selection(self.selection[:, i1])
            ch2, ry2, rx2 = self.unpack_selection(self.selection[:, i2])

            for oy in range(h_out):
                for ox in range(w_out):
                    iy1 = oy * stride_h + ry1 - pad_h
                    ix1 = ox * stride_w + rx1 - pad_w
                    iy2 = oy * stride_h + ry2 - pad_h
                    ix2 = ox * stride_w + rx2 - pad_w

                    valid1 = (iy1 >= 0) & (iy1 < h_in) & (ix1 >= 0) & (ix1 < w_in)
                    valid2 = (iy2 >= 0) & (iy2 < h_in) & (ix2 >= 0) & (ix2 < w_in)

                    for n_idx in range(n):
                        for c in range(c_out):
                            x1 = x[n_idx, ch1[c], iy1[c], ix1[c]] if valid1[c] else 0.0
                            x2 = x[n_idx, ch2[c], iy2[c], ix2[c]] if valid2[c] else 0.0
                            a1[n_idx, c, oy, ox, p] = x1
                            b1[n_idx, c, oy, ox, p] = x2

        x1 = bin_op_s(a1, b1, w_prob_stage1)

        # --- Stage 2: TreeLogic1 ---
        n, c, _, _, d = x1.shape
        x_out2 = d // 2
        a2 = x1[..., 0::2]
        b2 = x1[..., 1::2]

        if self.training:
            w_prob2 = torch.nn.functional.softmax(w_tree1, dim=-1)
        else:
            idx = w_tree1.argmax(-1)
            w_prob2 = torch.nn.functional.one_hot(idx, num_classes=16).to(x.dtype)
        w_prob2 = w_prob2.view(1, c, 1, 1, x_out2, 16)

        x2 = bin_op_s(a2, b2, w_prob2)

        # --- Stage 3: TreeLogic2 ---
        n, c, _, _, d = x2.shape
        x_out3 = d // 2
        a3 = x2[..., 0::2]
        b3 = x2[..., 1::2]

        if self.training:
            w_prob3 = torch.nn.functional.softmax(w_tree2, dim=-1)
        else:
            idx = w_tree2.argmax(-1)
            w_prob3 = torch.nn.functional.one_hot(idx, num_classes=16).to(x.dtype)
        w_prob3 = w_prob3.view(1, c, 1, 1, x_out3, 16)

        y = bin_op_s(a3, b3, w_prob3)
        y = PoolLogicLayer()(y)
        return y

    def forward_cuda(self, x):
        x = x.contiguous()

        if self.training:
            w_prob = torch.nn.functional.softmax(self.weights, dim=-1)
        else:
            idx = self.weights.argmax(-1)
            w_prob = torch.nn.functional.one_hot(idx, num_classes=16).to(dtype=x.dtype)

        w_prob = w_prob.contiguous()
        sel = self.selection.contiguous()

        if self.complete:
            return FullConvLogicFunction.apply(
                x,
                w_prob,
                sel,
                self.stride[0],
                self.stride[1],
                self.padding[0],
                self.padding[1],
                self.kernel[0],
                self.kernel[1],
                self.return_intermediate,
                self.return_indices,
            )
        else:
            return ConvLogicFunction.apply(
                x,
                w_prob,
                sel,
                self.stride[0],
                self.stride[1],
                self.padding[0],
                self.padding[1],
                self.kernel[0],
                self.kernel[1],
            )

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
        x.t = convlogic_cuda.eval(x.t, a, b, w)

        return x

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, groups={self.groups}"

    def generate_selection(self) -> torch.Tensor:
        dev = self.weights.device

        num_inputs = self.input_bits
        kh, kw = self.kernel
        out_channels = self.out_channels
        in_channels = self.in_channels
        groups = self.groups

        # Grouping input channels
        base = in_channels // groups
        remainder = in_channels % groups
        sizes = torch.full((groups,), base, device=dev)
        sizes[:remainder] += 1

        # Compute start index for each group
        starts = torch.cat([torch.zeros(1, device=dev), sizes.cumsum(0)])[:-1].to(torch.int32)

        # All channel indices
        all_idx = torch.arange(in_channels, device=dev, dtype=torch.int32)
        group_map = [all_idx[s : s + sizes[i]] for i, s in enumerate(starts)]

        # Map output channels to group
        out_per = out_channels // groups
        group_ids = torch.clamp(torch.arange(out_channels, device=dev) // out_per, 0, groups - 1)

        # Randomly pick two channels per output channel within its group
        chosen = torch.empty((out_channels, 2), dtype=torch.int32, device=dev)
        for gid, chans in enumerate(group_map):
            mask = group_ids == gid
            idxs = mask.nonzero(as_tuple=True)[0]
            cnt = idxs.numel()
            if cnt > 0:
                k = chans.numel()
                chosen[idxs, 0] = chans[torch.randint(0, k, (cnt,), device=dev)]
                chosen[idxs, 1] = chans[torch.randint(0, k, (cnt,), device=dev)]

        # For each of the 8 input positions, pick either of the two chosen channels
        picks = torch.randint(0, 2, (out_channels, num_inputs), device=dev, dtype=torch.int32)
        ch_sel = torch.where(
            picks == 0,
            chosen[:, 0].unsqueeze(1).expand(-1, num_inputs),
            chosen[:, 1].unsqueeze(1).expand(-1, num_inputs),
        )

        # Random (row, col) coordinates for each input position
        rows = torch.randint(0, kh, (out_channels, num_inputs), device=dev, dtype=torch.int32)
        cols = torch.randint(0, kw, (out_channels, num_inputs), device=dev, dtype=torch.int32)

        # Resolve inputs from the same position in each logic gate input pair (k, k+1)
        for k in range(0, num_inputs, 2):
            ch0 = ch_sel[:, k]
            ch1 = ch_sel[:, k + 1]
            r0 = rows[:, k]
            r1 = rows[:, k + 1]
            c0 = cols[:, k]
            c1 = cols[:, k + 1]

            # Same position but different channel is okay
            conflict = (ch0 == ch1) & (r0 == r1) & (c0 == c1)

            if conflict.any():
                idxs = conflict.nonzero(as_tuple=True)[0]
                for idx in idxs.tolist():
                    # Loop until the regenerated (r1, c1) differs from (r0, c0)
                    while True:
                        new_r = torch.randint(0, kh, (1,), device=dev, dtype=torch.int32)
                        new_c = torch.randint(0, kw, (1,), device=dev, dtype=torch.int32)
                        if not (new_r.item() == r0[idx].item() and new_c.item() == c0[idx].item()):
                            rows[idx, k + 1] = new_r
                            cols[idx, k + 1] = new_c
                            break

        # Pack (channel, row, col) into a 32-bit int: ch << 16 | row << 8 | col
        packed = (ch_sel << 16) | (rows << 8) | cols
        return packed


class PoolLogicLayer(nn.Module):
    """
    In the context of logic gate networks, max pooling simulates an OR gates:
    if any of the inputs in the window are 1, the output will be 1.
    """

    def __init__(self, kernel=2, stride=2, return_indices=False):
        super().__init__()
        self.kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.return_indices = return_indices

    def forward(self, x):
        if x.ndim == 4:
            pass
        elif x.ndim == 5:
            if x.size(4) != 1:
                raise ValueError(f"Expected last dim=1, got {x.size(4)}")
            x = x.view(x.size(0), x.size(1), x.size(2), x.size(3))
        else:
            raise ValueError("Input must be 4D or 5D")

        if self.return_indices:
            return nn.functional.max_pool2d(x, self.kernel, self.stride, return_indices=True)
        else:
            return nn.functional.max_pool2d(x, self.kernel, self.stride)
