"""T.10-bwd — Differentiable wrapper for the Vulkan-native RNN forward.

Provides :class:`_VulkanRnnFunction` — a ``torch.autograd.Function`` that:
  * Forward: dispatches the T.10 Vulkan-native template (fast path).
  * Backward: computes gradients via the Vulkan BPTT kernel (OP.25) or
    the CPU-roundtrip fallback (same as the custom-op backward path).

This wrapper is used by ``_lstm_input_impl`` and friends so that LSTM /
GRU / RNN outputs carry a proper ``grad_fn`` and training loss.backward()
works without routing the *forward* through the CPU-roundtrip custom op.
"""

from __future__ import annotations

from typing import Optional

import torch


class _VulkanRnnFunction(torch.autograd.Function):
    """Differentiable Vulkan-native RNN forward/backward.

    Args (saved on ctx for backward):
        vf_name: ``"lstm"``, ``"gru"``, ``"rnn_tanh"``, or ``"rnn_relu"``.
        is_lstm: bool
        saved_inputs: flat list of input tensors (same ordering as the
            ``torch_vulkan::<vf>_cpu_roundtrip`` custom-op ``tensors``
            argument so we can re-use the existing backward logic).
        has_biases, num_layers, dropout, train, bidirectional, batch_first:
            scalar hyperparameters forwarded to the backward.
        n_params: length of the params sub-list in saved_inputs.
    """

    @staticmethod
    def forward(
        ctx,
        vf_name,
        is_lstm,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        n_params,
        input_t,
        *rest,  # h0, [c0,] *params
    ):
        """Run T.10 Vulkan-native RNN forward and save inputs for backward.

        Returns a flat tuple of output tensors:
          LSTM: (output, h_n, c_n)
          other: (output, h_n)
        """
        from ..rnn_template import run_rnn_via_template

        # Reconstruct hx and params from *rest.
        if is_lstm:
            # rest = h0, c0, *params
            h0, c0 = rest[0], rest[1]
            params = list(rest[2:])
            hx = (h0, c0)
            saved = [input_t, h0, c0, *params]
        else:
            # rest = hx, *params
            hx = rest[0]
            params = list(rest[1:])
            saved = [input_t, hx, *params]

        # T.10 native Vulkan dispatch.
        out = run_rnn_via_template(
            vf_name,
            input_t,
            hx,
            params,
            has_biases,
            num_layers,
            bidirectional,
            batch_first,
            dropout=dropout,
            train=train,
        )
        # out = (output, h_n, c_n|None)

        ctx.save_for_backward(*saved)
        ctx._vf_name = vf_name
        ctx._is_lstm = is_lstm
        ctx._n_params = n_params
        ctx._has_biases = has_biases
        ctx._num_layers = num_layers
        ctx._dropout = dropout
        ctx._train = train
        ctx._bidirectional = bidirectional
        ctx._batch_first = batch_first

        # Return as tuple; caller unpacks.
        if is_lstm:
            return out[0], out[1], out[2]  # output, h_n, c_n
        return out[0], out[1]  # output, h_n

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Compute RNN backward via Vulkan BPTT (OP.25) or CPU roundtrip."""
        from .common import _unpack_vf_args
        from .bwd_vulkan import _use_vulkan_rnn_bwd, _run_vulkan_rnn_bwd

        saved = list(ctx.saved_tensors)
        vf_name = ctx._vf_name
        n_params = ctx._n_params
        is_lstm = ctx._is_lstm
        is_data = False  # only non-packed path uses this function
        has_biases = ctx._has_biases
        num_layers = ctx._num_layers
        dropout = ctx._dropout
        train = ctx._train
        bidirectional = ctx._bidirectional
        batch_first = ctx._batch_first

        device = saved[0].device

        # Normalise grad_outputs: None → zero.
        grad_list = [
            g if g is not None else torch.zeros(0, device=device)
            for g in grad_outputs
        ]

        # OP.25: try Vulkan-native BPTT first (single-layer, non-bidirectional).
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
                    grad_list,
                    saved,
                    has_biases,
                    batch_first,
                )
                if result is not None:
                    return _bwd_return(result, saved, n_params, is_lstm)
            except Exception:
                pass  # fall through to CPU roundtrip

        # CPU-roundtrip backward (TP.2 fallback).
        with torch.enable_grad():
            cpu_tensors = []
            for t in saved:
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

        diff_outputs = [
            o for o in cpu_out if isinstance(o, torch.Tensor) and o.requires_grad
        ]
        if not diff_outputs:
            return _zero_grads(saved, vf_name, is_lstm)

        grad_for_diff = []
        for go, co in zip(grad_list, cpu_out):
            if not (isinstance(co, torch.Tensor) and co.requires_grad):
                continue
            if go is None or (isinstance(go, torch.Tensor) and go.numel() == 0 and co.numel() != 0):
                grad_for_diff.append(torch.zeros_like(co))
            else:
                grad_for_diff.append(go.detach().to("cpu"))
        while len(grad_for_diff) < len(diff_outputs):
            grad_for_diff.append(torch.zeros_like(diff_outputs[len(grad_for_diff)]))

        diff_inputs = [t for t in cpu_tensors if t.requires_grad]
        if not diff_inputs:
            return _zero_grads(saved, vf_name, is_lstm)

        grads = torch.autograd.grad(
            outputs=diff_outputs,
            inputs=diff_inputs,
            grad_outputs=grad_for_diff,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
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

        return _bwd_return(result, saved, n_params, is_lstm)


def _bwd_return(tensor_grads, saved, n_params, is_lstm):
    """Convert a flat tensor-grad list into the backward return tuple.

    The forward signature is:
        (vf_name, is_lstm, has_biases, num_layers, dropout, train,
         bidirectional, batch_first, n_params, input_t, h0, [c0,] *params)

    The first 9 args are non-tensor (None gradient), followed by tensor args.
    """
    # 9 non-tensor args: vf_name, is_lstm, has_biases, num_layers, dropout,
    #                    train, bidirectional, batch_first, n_params
    none_prefix = (None,) * 9
    tensor_grads_tuple = tuple(tensor_grads)
    return none_prefix + tensor_grads_tuple


def _zero_grads(saved, vf_name, is_lstm):
    """Return zero gradients matching _bwd_return signature."""
    zeros = [torch.zeros_like(t) for t in saved]
    return _bwd_return(zeros, saved, len(zeros), is_lstm)


def apply_vulkan_rnn(
    vf_name: str,
    is_lstm: bool,
    input_t: torch.Tensor,
    hx,
    params: list,
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
):
    """Convenience wrapper around :class:`_VulkanRnnFunction`.

    Returns a tuple of output tensors: ``(output, h_n, c_n)`` for LSTM,
    ``(output, h_n)`` for GRU/RNN.
    """
    n_params = len(params)
    if is_lstm:
        h0, c0 = hx
        return _VulkanRnnFunction.apply(
            vf_name, is_lstm, has_biases, num_layers, dropout, train,
            bidirectional, batch_first, n_params,
            input_t, h0, c0, *params,
        )
    else:
        return _VulkanRnnFunction.apply(
            vf_name, is_lstm, has_biases, num_layers, dropout, train,
            bidirectional, batch_first, n_params,
            input_t, hx, *params,
        )
