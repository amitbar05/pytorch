"""Opaque conv2d_backward / conv1d_backward_core custom-op registrations.

M22b: split from conv.py.
M17.8.d.2: ``conv2d_backward`` opaque non-autograd op prevents AOTAutograd
  from decomposing conv backward into ``empty_like`` sub-ops.
S3.5b (2026-06-22): ``conv1d_backward_core`` opaque non-autograd op for the
  conv1d backward path.  Takes 3D tensors directly (no Python-side unsqueeze)
  so that (a) ``grad_output`` is never concretised into an FX constant and
  (b) no ``input_4d`` intermediate is created that the partitioner can
  buffer-reuse for the gradient output, which was aliasing ``gi → primals_3``.

TRAIN.7 (2026-05-27): The eager impl now routes fp32 Vulkan groups==1 through
``_slang_tile_conv2d_bwd`` (single Slang compute dispatch via
``bwd_diff(conv_inner_madd)``), eliminating the prior CPU roundtrip through
``aten.convolution_backward`` → C++ ``vulkan_convolution_backward_overrideable``.
"""

from __future__ import annotations


def _ensure_conv1d_backward_core_op_registered() -> "object":
    """Register ``torch_vulkan::conv1d_backward_core`` — a non-autograd opaque op.

    S3.5b root-cause:  in ``_conv1d_backward`` the old code built a 4-D view
    ``input_4d = input.unsqueeze(-1)`` and passed it as the first arg of
    ``conv2d_backward``.  That view is an *intermediate* in the backward FX
    graph — dead after ``conv2d_backward`` — so the Inductor memory planner
    reused its buffer for ``conv2d_backward[0]`` (grad-input).  Because
    ``input_4d`` is a view of ``primals_3`` (the primal x), grad-input ended
    up sharing ``primals_3``'s buffer and the backward graph returned ``x``
    itself as grad_input instead of the computed gradient.

    This op takes *3-D* tensors directly (no Python-side unsqueeze/squeeze)
    so ``input`` is passed as a *backward graph input* (not an intermediate).
    Graph inputs are never freed by the memory planner, so the buffer-reuse
    aliasing cannot occur.

    The fake kernel derives all outputs from ``input`` (a FunctionalTensor
    proxy) rather than from the possibly-concrete ``grad_output`` so that
    FunctionalTensorMode produces true-fresh FT outputs rather than falling
    back to conservative alias analysis (which assumed output aliases input).

    Idempotent — safe to call multiple times.
    """
    import torch
    from torch import Tensor

    op_name = "torch_vulkan::conv1d_backward_core"
    existing = getattr(torch.ops.torch_vulkan, "conv1d_backward_core", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    def _conv1d_backward_core_impl(
        input: Tensor,
        grad_output: Tensor,
        weight: Tensor,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
        has_bias: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Eager: lift 3-D → 4-D, run conv2d_backward, squeeze back to 3-D."""
        use_slang_bwd = (
            int(groups) == 1
            and input.device.type == "vulkan"
            and input.dtype == torch.float32
            and grad_output.dtype == torch.float32
            and weight.dtype == torch.float32
        )

        s = int(stride[0]) if stride else 1
        p = int(padding[0]) if padding else 0
        d = int(dilation[0]) if dilation else 1

        if use_slang_bwd:
            from ...templates.caller.conv import _slang_tile_conv2d_bwd

            # 1-D → 2-D via (N, C, L) → (N, C, L, 1)  [H=L, W=1]
            inp_4d = input.unsqueeze(-1)
            go_4d = grad_output.unsqueeze(-1)
            w_4d = weight.unsqueeze(-1)

            g_inp_4d = torch.zeros_like(inp_4d)
            g_w_4d = torch.zeros_like(w_4d)
            _slang_tile_conv2d_bwd(
                inp_4d, w_4d, go_4d, g_inp_4d, g_w_4d,
                stride=(s, 1), padding=(p, 0), dilation=(d, 1),
                bias=None, grad_bias=None,
            )
            g_inp = g_inp_4d.squeeze(-1)
            g_w = g_w_4d.squeeze(-1)
            g_b = go_4d.squeeze(-1).sum([0, 2]) if has_bias else None
            if has_bias:
                _ = g_b[0].item()
            if not has_bias:
                g_b = grad_output.new_empty((0,))
            return g_inp, g_w, g_b

        # CPU / fallback: aten.convolution_backward on 1-D strides
        result = torch.ops.aten.convolution_backward.default(
            grad_output, input, weight, None,
            list(stride), list(padding), list(dilation),
            False, [0], int(groups),
            [True, True, bool(has_bias)],
        )
        g_inp = result[0] if result[0] is not None else input.new_empty(input.shape).zero_()
        g_w = result[1] if result[1] is not None else weight.new_empty(weight.shape).zero_()
        if has_bias:
            g_b = result[2] if len(result) > 2 and result[2] is not None else grad_output.new_empty((weight.shape[0],)).zero_()
        else:
            g_b = grad_output.new_empty((0,))
        return g_inp, g_w, g_b

    _conv1d_backward_core_impl.__annotations__ = {
        "input": Tensor, "grad_output": Tensor, "weight": Tensor,
        "stride": list[int], "padding": list[int], "dilation": list[int],
        "groups": int, "has_bias": bool,
        "return": tuple[Tensor, Tensor, Tensor],
    }
    bwd1d_op = torch.library.custom_op(op_name, mutates_args=())(_conv1d_backward_core_impl)

    def _conv1d_backward_core_fake(
        input, grad_output, weight, stride, padding, dilation, groups, has_bias
    ):
        # S3.5b: derive ALL outputs from ``input`` (a FunctionalTensor proxy),
        # NOT from ``grad_output`` which may be a concrete Vulkan tensor during
        # AoT backward tracing.  Deriving from a concrete arg causes
        # FunctionalTensorMode to fall back to conservative alias analysis
        # (assume output aliases first FT input = ``input``) which incorrectly
        # concludes grad_input = primals_3 (the primal x).
        #
        # ``new_empty`` is non-aliasing: the fresh FT it returns has a distinct
        # storage_ref so AoT correctly treats grad_input as an independent
        # computation, saves primals_3 as a forward residual, and produces the
        # right backward graph.
        g_inp = input.new_empty(input.shape)
        g_w = input.new_empty(weight.shape)
        if has_bias:
            g_b = input.new_empty((weight.shape[0],))
        else:
            g_b = input.new_empty((0,))
        return g_inp, g_w, g_b

    bwd1d_op.register_fake(_conv1d_backward_core_fake)
    return torch.ops.torch_vulkan.conv1d_backward_core.default


def _ensure_conv2d_backward_op_registered() -> "object":
    """Register ``torch_vulkan::conv2d_backward`` as a non-autograd custom_op.

    M17.8.d.2 (2026-05-17): During AOTAutograd's joint-graph trace, the
    custom-op autograd ``_conv2d_backward`` runs with FakeTensors and
    previously fell through to ``torch.ops.aten.convolution_backward.default``.
    That call dispatched to our PrivateUse1 fake (``shape_ops.py::
    _convolution_backward_overrideable_fake``) whose body uses
    ``torch.empty_like(input)`` / ``torch.empty_like(weight)``. AOTAutograd's
    proxy tracer **recorded those sub-ops** into the FX graph instead of
    preserving a single op node — the joint-partitioner then saw
    ``empty_like(weight)`` as shape-only and dropped the primals from the
    backward partition. Inductor lowered the result as ``alloc + zero-init``,
    silently producing all-zero conv weight gradients in compile mode.

    This op is **non-autograd** and **opaque to the tracer**: a single
    ``torch_vulkan::conv2d_backward.default`` node lands in the FX graph,
    the joint-partitioner correctly preserves ``input`` / ``weight`` as
    backward inputs, and ``make_fallback`` (registered in
    ``lowerings/__init__.py``) makes Inductor emit a real
    ``extern_kernels.conv2d_backward(...)`` call that runs the C++ adapter
    at runtime.

    Idempotent — safe to call multiple times.
    """
    import torch
    from torch import Tensor

    op_name = "torch_vulkan::conv2d_backward"
    existing = getattr(torch.ops.torch_vulkan, "conv2d_backward", None)
    if existing is not None and hasattr(existing, "default"):
        return existing.default

    def _conv2d_backward_impl(
        input: Tensor,
        grad_output: Tensor,
        weight: Tensor,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
        has_bias: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Eager impl: route fp32 Vulkan groups==1 through _slang_tile_conv2d_bwd
        (TRAIN.7, 2026-05-27) — single Slang compute dispatch, no CPU roundtrip.
        Falls back to aten.convolution_backward for groups>1 or non-fp32.
        """
        use_slang_bwd = (
            int(groups) == 1
            and input.device.type == "vulkan"
            and input.dtype == torch.float32
            and grad_output.dtype == torch.float32
            and weight.dtype == torch.float32
        )

        if use_slang_bwd:
            from ...templates.caller.conv import _slang_tile_conv2d_bwd

            sH, sW = int(stride[0]), int(stride[-1] if len(stride) > 1 else stride[0])
            pH, pW = int(padding[0]), int(padding[-1] if len(padding) > 1 else padding[0])
            dH, dW = int(dilation[0]), int(dilation[-1] if len(dilation) > 1 else dilation[0])

            # Allocate fresh output tensors — do NOT use pool_acquire which
            # may alias with input buffers in compiled contexts.
            grad_input = torch.zeros_like(input)
            grad_weight = torch.zeros_like(weight)
            grad_bias = torch.zeros(weight.shape[0], device=input.device, dtype=input.dtype) if has_bias else None

            _slang_tile_conv2d_bwd(
                input, weight, grad_output,
                grad_input, grad_weight,
                stride=(sH, sW),
                padding=(pH, pW),
                dilation=(dH, dW),
                grad_bias=grad_bias,
            )

            # Force GPU pipeline drain before returning tensors to the
            # compiled wrapper.  In the FallbackKernel path, the wrapper may
            # consume output buffers immediately; a pending GPU write
            # produces stale data.  ``.item()`` on grad_bias forces a
            # blocking read-back that commits all prior dispatches.
            if has_bias:
                _ = grad_bias[0].item()

            if not has_bias:
                grad_bias = grad_output.new_empty((0,))
            return grad_input, grad_weight, grad_bias

        result = torch.ops.aten.convolution_backward.default(
            grad_output,
            input,
            weight,
            None,
            list(stride),
            list(padding),
            list(dilation),
            False,
            [0] * len(stride),
            int(groups),
            [True, True, bool(has_bias)],
        )
        g_inp = (
            result[0]
            if result[0] is not None
            else input.new_empty(input.shape).zero_()
        )
        g_w = (
            result[1]
            if result[1] is not None
            else weight.new_empty(weight.shape).zero_()
        )
        if has_bias:
            g_b = (
                result[2]
                if len(result) > 2 and result[2] is not None
                else grad_output.new_empty((weight.shape[0],)).zero_()
            )
        else:
            # Return a zero-size bias so the tuple arity is stable. Callers
            # ignore this when has_bias=False.
            g_b = grad_output.new_empty((0,))
        return g_inp, g_w, g_b

    _conv2d_backward_impl.__annotations__ = {
        "input": Tensor,
        "grad_output": Tensor,
        "weight": Tensor,
        "stride": list[int],
        "padding": list[int],
        "dilation": list[int],
        "groups": int,
        "has_bias": bool,
        "return": tuple[Tensor, Tensor, Tensor],
    }
    bwd_op = torch.library.custom_op(op_name, mutates_args=())(_conv2d_backward_impl)

    def _conv2d_backward_fake(
        input, grad_output, weight, stride, padding, dilation, groups, has_bias
    ):
        # Shape inference for the opaque op.  Derive all outputs from
        # ``grad_output`` (NOT from ``input`` / ``weight``) so that AoT
        # autograd cannot alias g_inp→input or g_w→weight via the
        # unsqueeze/new_empty/squeeze chain that _conv1d_backward builds
        # (M17.8.d.2 variant: squeeze(new_empty(unsqueeze(x,-1)),-1) → x).
        g_inp = grad_output.new_empty(input.shape)
        g_w = grad_output.new_empty(weight.shape)
        if has_bias:
            g_b = grad_output.new_empty((weight.shape[0],))
        else:
            g_b = grad_output.new_empty((0,))
        return g_inp, g_w, g_b

    bwd_op.register_fake(_conv2d_backward_fake)
    return torch.ops.torch_vulkan.conv2d_backward.default
