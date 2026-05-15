"""T.10 / CP.3 — shared RNN utilities: custom-op builders, argument unpacking,
registration helpers, fallbacks, and the eager-intercept orchestrator."""

from __future__ import annotations

import os

import torch

# Module-level handle to the dispatcher library so the registration outlives
# the registering function call (a Library instance with no live ref drops
# its registrations on GC). Also tracks whether the custom ops + intercepts
# have already been installed (idempotent across ``register()`` calls and
# ``torch._dynamo.reset()``).
_RNN_LIB = None
_CUSTOM_OPS_REGISTERED = False
_RNN_CUSTOM_OPS: dict[str, object] = {}

# Pre-built template callers — created lazily on first use.
_RNN_TEMPLATE_CALLERS: dict[str, object] = {}


def _build_cpu_roundtrip_op(vf_name: str):
    """Define ``torch_vulkan::<vf_name>_cpu_roundtrip`` (forward) and
    ``torch_vulkan::<vf_name>_cpu_roundtrip_backward`` custom ops.

    Forward schema (uniform across all RNN families):
        op(tensors: list[Tensor], n_params: int, is_data: bool,
           has_biases: bool, num_layers: int, dropout: float, train: bool,
           bidirectional: bool, batch_first: bool) -> list[Tensor]

    Backward schema:
        op_bwd(grad_outputs: Tensor[], saved_inputs: Tensor[],
               n_params: int, is_data: bool, has_biases: bool,
               num_layers: int, dropout: float, train: bool,
               bidirectional: bool, batch_first: bool) -> Tensor[]

    ``tensors`` ordering, by overload (matches ``_unpack_vf_args``):
      * lstm.input:  [input, h0, c0, *params]
      * lstm.data:   [data, batch_sizes, h0, c0, *params]
      * gru.input / rnn_*.input:  [input, hx, *params]
      * gru.data  / rnn_*.data:   [data, batch_sizes, hx, *params]

    Both forward and backward are opaque to AOTAutograd thanks to their
    ``register_fake`` impls, so the joint graph contains exactly one
    extern-kernel node for forward and one for backward.
    """
    import torch

    op_name = f"torch_vulkan::{vf_name}_cpu_roundtrip"
    op_bwd_name = f"torch_vulkan::{vf_name}_cpu_roundtrip_backward"

    # Explicit schema (avoids forward-ref resolution issues caused by
    # ``from __future__ import annotations``).
    schema = (
        "(Tensor[] tensors, int n_params, bool is_data, bool has_biases, "
        "int num_layers, float dropout, bool train, bool bidirectional, "
        "bool batch_first) -> Tensor[]"
    )

    @torch.library.custom_op(op_name, mutates_args=(), schema=schema)
    def _impl(
        tensors,
        n_params,
        is_data,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    ):
        device = tensors[0].device
        cpu_tensors = [t.detach().to("cpu") for t in tensors]
        cpu_args = _unpack_vf_args(
            vf_name,
            cpu_tensors,
            n_params,
            is_data,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )
        out = getattr(torch._VF, vf_name)(*cpu_args)
        return [o.to(device) if isinstance(o, torch.Tensor) else o for o in out]

    @_impl.register_fake
    def _meta(
        tensors,
        n_params,
        is_data,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    ):
        # Shape inference for the RNN forward outputs.
        # Output 0 (sequence): same shape as input but last dim swapped to
        # hidden_size * (2 if bidirectional else 1). We get hidden_size
        # from the hx shape.
        # Output 1 (h_n): [num_layers * directions, batch, hidden]
        # Output 2 (c_n, LSTM only): same shape as h_n.
        directions = 2 if bidirectional else 1
        if is_data:
            data = tensors[0]
            # batch_sizes = tensors[1]
            if vf_name == "lstm":
                h0 = tensors[2]
                hidden_size = h0.shape[-1]
                # data is packed: [total_T, input_size]
                seq = data.new_empty(data.shape[0], hidden_size * directions)
                h_n = h0.new_empty(num_layers * directions, h0.shape[1], hidden_size)
                c_n = h0.new_empty(num_layers * directions, h0.shape[1], hidden_size)
                return [seq, h_n, c_n]
            else:
                h0 = tensors[2]
                hidden_size = h0.shape[-1]
                seq = data.new_empty(data.shape[0], hidden_size * directions)
                h_n = h0.new_empty(num_layers * directions, h0.shape[1], hidden_size)
                return [seq, h_n]
        else:
            inp = tensors[0]
            if vf_name == "lstm":
                h0 = tensors[1]
                # c0 = tensors[2]
                hidden_size = h0.shape[-1]
                # inp: [seq, batch, in] or [batch, seq, in] (batch_first)
                if batch_first:
                    seq = inp.new_empty(
                        inp.shape[0], inp.shape[1], hidden_size * directions
                    )
                else:
                    seq = inp.new_empty(
                        inp.shape[0], inp.shape[1], hidden_size * directions
                    )
                h_n = h0.new_empty(num_layers * directions, h0.shape[1], hidden_size)
                c_n = h0.new_empty(num_layers * directions, h0.shape[1], hidden_size)
                return [seq, h_n, c_n]
            else:
                hx = tensors[1]
                hidden_size = hx.shape[-1]
                if batch_first:
                    seq = inp.new_empty(
                        inp.shape[0], inp.shape[1], hidden_size * directions
                    )
                else:
                    seq = inp.new_empty(
                        inp.shape[0], inp.shape[1], hidden_size * directions
                    )
                h_n = hx.new_empty(num_layers * directions, hx.shape[1], hidden_size)
                return [seq, h_n]

    # Backward custom op: opaque to AOTAutograd (its register_fake returns
    # gradient-shaped meta tensors, NOT the CPU roundtrip). Implementation
    # does the actual CPU roundtrip + torch.autograd.grad.
    bwd_schema = (
        "(Tensor[] grad_outputs, Tensor[] saved_inputs, int n_params, "
        "bool is_data, bool has_biases, int num_layers, float dropout, "
        "bool train, bool bidirectional, bool batch_first) -> Tensor[]"
    )

    @torch.library.custom_op(op_bwd_name, mutates_args=(), schema=bwd_schema)
    def _bwd_impl(
        grad_outputs,
        saved_inputs,
        n_params,
        is_data,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    ):
        import os

        if os.environ.get("TVDBG_RNN_BWD"):
            print(
                f"[_bwd_impl] {vf_name} n_grad_outputs={len(grad_outputs)} n_saved={len(saved_inputs)}"
            )
            for i, go in enumerate(grad_outputs):
                print(
                    f"  go[{i}] shape={go.shape if isinstance(go, torch.Tensor) else type(go)} norm={go.norm().item() if isinstance(go, torch.Tensor) else None}"
                )
        device = saved_inputs[0].device

        # OP.25: Vulkan-native BPTT fast path
        if (
            _use_vulkan_rnn_bwd()
            and not is_data
            and num_layers == 1
            and not bidirectional
            and dropout == 0.0
        ):
            try:
                result = _run_vulkan_rnn_bwd(
                    vf_name,
                    grad_outputs,
                    saved_inputs,
                    has_biases,
                    batch_first,
                )
                if result is not None:
                    if os.environ.get("TVDBG_RNN_BWD"):
                        print("[_bwd_impl] Vulkan-native BPTT succeeded")
                    return result
            except Exception as e:
                if os.environ.get("TVDBG_RNN_BWD"):
                    print(f"[_bwd_impl] Vulkan BPTT failed: {e}, falling back to CPU")

        # Use float dtype for tensors that hold real values; integer tensors
        # like ``batch_sizes`` should not require grad. Keep the original
        # dtype/shape but only mark floating-point tensors as needing grad.
        # NOTE: ``requires_grad_(True)`` must be called inside the
        # ``enable_grad()`` block (the autograd backward pass disables grad
        # tracking by default; even ``leaf.requires_grad_(True)`` is a no-op
        # if grad mode is off when the op is then invoked).
        with torch.enable_grad():
            import os

            if os.environ.get("TVDBG_RNN_BWD"):
                print(f"  grad_enabled inside enable_grad: {torch.is_grad_enabled()}")
                print(f"  inference_mode: {torch.is_inference_mode_enabled()}")
            cpu_tensors = []
            for t in saved_inputs:
                ct = t.detach().to("cpu", copy=True)
                if ct.is_floating_point():
                    ct.requires_grad_(True)
                cpu_tensors.append(ct)
            cpu_args = _unpack_vf_args(
                vf_name,
                cpu_tensors,
                n_params,
                is_data,
                has_biases,
                num_layers,
                dropout,
                train,
                bidirectional,
                batch_first,
            )
            cpu_out = getattr(torch._VF, vf_name)(*cpu_args)
        import os

        if os.environ.get("TVDBG_RNN_BWD"):
            for i, co in enumerate(cpu_out):
                print(
                    f"  cpu_out[{i}] shape={co.shape} req_grad={co.requires_grad} grad_fn={co.grad_fn}"
                )
            for i, ct in enumerate(cpu_tensors):
                print(
                    f"  cpu_tensors[{i}] shape={ct.shape} req_grad={ct.requires_grad} fp={ct.is_floating_point()}"
                )
        # Filter to outputs that actually have grad_fn (some outputs like
        # h_n/c_n may be tuple-detached depending on the path).
        diff_outputs = [
            o for o in cpu_out if isinstance(o, torch.Tensor) and o.requires_grad
        ]
        if not diff_outputs:
            # Cannot differentiate. Return zero grads.
            return [torch.zeros_like(t) for t in saved_inputs]
        # Build grad_outputs aligned with diff_outputs.
        grad_for_diff = []
        for go, co in zip(grad_outputs, cpu_out):
            if not (isinstance(co, torch.Tensor) and co.requires_grad):
                continue
            if go is None or (
                isinstance(go, torch.Tensor) and go.numel() == 0 and co.numel() != 0
            ):
                grad_for_diff.append(torch.zeros_like(co))
            else:
                grad_for_diff.append(go.detach().to("cpu"))
        # Pad if grad_outputs was shorter (shouldn't happen but defensive).
        while len(grad_for_diff) < len(diff_outputs):
            grad_for_diff.append(torch.zeros_like(diff_outputs[len(grad_for_diff)]))
        diff_inputs = [t for t in cpu_tensors if t.requires_grad]
        if not diff_inputs:
            return [torch.zeros_like(t) for t in saved_inputs]
        grads = torch.autograd.grad(
            outputs=diff_outputs,
            inputs=diff_inputs,
            grad_outputs=grad_for_diff,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        # Stitch back: inputs that didn't require grad get zero gradients.
        grads_iter = iter(grads)
        result = []
        for t in cpu_tensors:
            if t.requires_grad:
                g = next(grads_iter)
                result.append(
                    g.to(device) if g is not None else torch.zeros_like(t).to(device)
                )
            else:
                result.append(torch.zeros_like(t).to(device))
        return result

    @_bwd_impl.register_fake
    def _bwd_meta(
        grad_outputs,
        saved_inputs,
        n_params,
        is_data,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    ):
        # Each input gets a gradient with the same shape as that input.
        return [torch.empty_like(t) for t in saved_inputs]

    def _setup_context(ctx, inputs, output):
        tensors = inputs[0]
        ctx.save_for_backward(*tensors)
        ctx._n_params = inputs[1]
        ctx._is_data = inputs[2]
        ctx._has_biases = inputs[3]
        ctx._num_layers = inputs[4]
        ctx._dropout = inputs[5]
        ctx._train = inputs[6]
        ctx._bidirectional = inputs[7]
        ctx._batch_first = inputs[8]

    def _backward(ctx, *grad_outputs):
        # Schema returns ``Tensor[]`` so PyTorch hands us a single positional
        # list of grads. Normalize to a flat list[Tensor | None].
        if len(grad_outputs) == 1 and isinstance(grad_outputs[0], (list, tuple)):
            grad_outputs = list(grad_outputs[0])
        else:
            grad_outputs = list(grad_outputs)
        saved = list(ctx.saved_tensors)
        # Replace None grads with zeros (the bwd custom op needs concrete
        # tensors for shape inference).
        grad_outputs = [
            g if g is not None else torch.zeros(0, device=saved[0].device)
            for g in grad_outputs
        ]
        out_tensor_grads = _bwd_impl(
            grad_outputs,
            saved,
            ctx._n_params,
            ctx._is_data,
            ctx._has_biases,
            ctx._num_layers,
            ctx._dropout,
            ctx._train,
            ctx._bidirectional,
            ctx._batch_first,
        )
        # Schema: tensors (list), n_params, is_data, has_biases, num_layers,
        # dropout, train, bidirectional, batch_first → 9 inputs. Gradient
        # for the tensor list, None for each scalar.
        return list(out_tensor_grads), None, None, None, None, None, None, None, None

    _impl.register_autograd(_backward, setup_context=_setup_context)
    return _impl


def _unpack_vf_args(
    vf_name,
    tensors,
    n_params,
    is_data,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """Reassemble the ``torch._VF.<op>`` call signature from the flat
    tensor list and config scalars carried through the custom op.
    """
    is_lstm = vf_name == "lstm"
    if is_data:
        data = tensors[0]
        batch_sizes = tensors[1]
        if is_lstm:
            hx = (tensors[2], tensors[3])
            params = list(tensors[4 : 4 + n_params])
        else:
            hx = tensors[2]
            params = list(tensors[3 : 3 + n_params])
        return (
            data,
            batch_sizes,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
        )
    inp = tensors[0]
    if is_lstm:
        hx = (tensors[1], tensors[2])
        params = list(tensors[3 : 3 + n_params])
    else:
        hx = tensors[1]
        params = list(tensors[2 : 2 + n_params])
    return (
        inp,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    )


def _register_rnn_custom_ops() -> None:
    global _CUSTOM_OPS_REGISTERED
    if _CUSTOM_OPS_REGISTERED:
        return
    for vf in ("lstm", "gru", "rnn_tanh", "rnn_relu"):
        _RNN_CUSTOM_OPS[vf] = _build_cpu_roundtrip_op(vf)
    _CUSTOM_OPS_REGISTERED = True


def _use_cpu_fallback() -> bool:
    """Check whether CPU-roundtrip fallback should be used. CP.3 / OP.6."""
    return os.environ.get("TORCH_VULKAN_RNN_CPU_FALLBACK") == "1"


# ── OP.25: Vulkan-native RNN backward helpers ──────────────────────
# Imported from bwd_vulkan.py to keep common.py ≤ 800 L (anti-goal #7).
from .bwd_vulkan import (
    _use_vulkan_rnn_bwd,
    _run_vulkan_rnn_bwd,
)


def _get_rnn_template_caller(cell_type):
    from ...vulkan_template_caller import _SlangTileRNN

    caller = _RNN_TEMPLATE_CALLERS.get(cell_type)
    if caller is None:
        caller = _SlangTileRNN(cell_type)
        _RNN_TEMPLATE_CALLERS[cell_type] = caller
    return caller


def _register_rnn_eager_intercepts() -> None:
    """Register Python ``AutogradPrivateUse1`` impls for the high-level RNN ops.

    These intercept ``aten.lstm.input`` etc. before the
    ``CompositeImplicitAutograd`` C++ decomposition fires (which would
    otherwise pull in ``aten._thnn_fused_lstm_cell``, an op with no
    Vulkan eager kernel) and forward to the ``torch_vulkan::<vf>_cpu_roundtrip``
    custom ops registered by :func:`_register_rnn_custom_ops`.

    Registered at the ``AutogradPrivateUse1`` dispatch key (not plain
    ``PrivateUse1``) because the custom-op autograd binding is what
    establishes the backward path. Registering at PrivateUse1 alone would
    skip the autograd kernel for the op and produce silently-zero grads.

    Idempotent: the module-level ``_RNN_LIB`` handle ensures we register
    each impl exactly once.
    """
    global _RNN_LIB
    if _RNN_LIB is not None:
        return
    import torch
    import torch.library

    _register_rnn_custom_ops()

    _RNN_LIB = torch.library.Library("aten", "IMPL")

    from .gru import register_gru_intercepts
    from .lstm import register_lstm_intercepts

    register_lstm_intercepts(_RNN_LIB)
    register_gru_intercepts(_RNN_LIB)


# Aten op overloads to register Inductor fallbacks for. Each entry is the
# qualified path under ``torch.ops.aten``.
_RNN_OP_PATHS = (
    ("lstm", "input"),
    ("lstm", "data"),
    ("gru", "input"),
    ("gru", "data"),
    ("rnn_tanh", "input"),
    ("rnn_tanh", "data"),
    ("rnn_relu", "input"),
    ("rnn_relu", "data"),
)

# T.10-bwd: backward primitives that AOTAutograd may emit if any
# decomposition path bypasses our custom-op wrapper (e.g. a model that
# manually invokes ``torch._VF.lstm`` from a custom Module without going
# through ``aten.lstm.input``). Registered as Inductor fallbacks so the AOT
# graph-lowering step emits an ``ExternKernel`` that falls back to eager.
# In practice these are not reached when our forward intercept is active.
_RNN_BACKWARD_OP_PATHS = (
    ("_cudnn_rnn_backward", "default"),
    ("_thnn_fused_lstm_cell_backward", "default"),
    ("_thnn_fused_lstm_cell_backward_impl", "default"),
    ("_thnn_fused_gru_cell_backward", "default"),
    ("_thnn_differentiable_lstm_cell_backward", "default"),
    ("_thnn_differentiable_gru_cell_backward", "default"),
    ("miopen_rnn_backward", "default"),
    ("mkldnn_rnn_layer_backward", "default"),
)


def _register_rnn_fallbacks() -> None:
    """Register ``aten.{lstm,gru,rnn_tanh,rnn_relu}.{input,data}`` and
    backward-op fallbacks.

    Idempotent: skips ops already lowered or already in the fallback set
    (e.g. on a second ``register()`` call after ``_dynamo.reset()``).
    """
    import torch
    from torch._decomp import decomposition_table as _aot_decomps
    from torch._inductor.decomposition import decompositions as _ind_decomps
    from torch._inductor.lowering import fallbacks, lowerings, make_fallback

    _register_rnn_eager_intercepts()

    aten = torch.ops.aten

    for ns, ol in _RNN_OP_PATHS + _RNN_BACKWARD_OP_PATHS:
        op_packet = getattr(aten, ns, None)
        if op_packet is None:
            continue
        op = getattr(op_packet, ol, None)
        if op is None:
            continue
        # Defensive pop: Inductor / AOT may install a decomp entry for these
        # in some torch versions; ensure the high-level op survives to our
        # fallback. No-op if not present.
        _ind_decomps.pop(op, None)
        _aot_decomps.pop(op, None)
        if op in lowerings or op in fallbacks:
            continue
        # ``override_decomp=True`` silences the make_fallback assert that
        # fires when a decomp entry existed for the op.
        make_fallback(op, override_decomp=True)
