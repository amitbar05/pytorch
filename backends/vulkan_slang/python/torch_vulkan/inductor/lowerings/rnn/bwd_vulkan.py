"""OP.25 — Vulkan-native RNN backward via BPTT.

Implements :func:`_run_single_layer_bptt` and :func:`_run_vulkan_rnn_bwd`
which replace the CPU roundtrip in the RNN custom-op backward with Vulkan
compute shader dispatches.

The BPTT algorithm:
1. Recompute forward pass to obtain intermediate hidden/cell states
2. Initialize gradient accumulators (weights, biases)
3. Iterate time steps in reverse, dispatching the rnn_cell_bwd Slang kernel
   at each step to compute and accumulate all parameter gradients.
"""

from __future__ import annotations

import os

import torch

# ── Module-level caches ──────────────────────────────────────────────────

_RNN_BWD_CALLERS: dict[str, object] = {}


def _get_rnn_bwd_caller(cell_type: str):
    """Return a cached _SlangTileRNNBackward for *cell_type*."""
    from ...templates.caller.rnn_backward import _SlangTileRNNBackward

    caller = _RNN_BWD_CALLERS.get(cell_type)
    if caller is None:
        caller = _SlangTileRNNBackward(cell_type)
        _RNN_BWD_CALLERS[cell_type] = caller
    return caller


def _use_vulkan_rnn_bwd() -> bool:
    """Check whether the Vulkan-native RNN backward should be used.

    OP.25: Enabled by default. Set ``TV_VULKAN_RNN_BWD=0`` to fall back to
    CPU roundtrip for debugging / comparison.
    """
    return os.environ.get("TV_VULKAN_RNN_BWD", "1") == "1"


# ── OP.25: Vulkan-native RNN backward via BPTT ──────────────────────────

_RNN_BWD_CALLERS: dict[str, object] = {}


def _get_rnn_bwd_caller(cell_type: str):
    """Return a cached _SlangTileRNNBackward for *cell_type*."""
    from ...templates.caller.rnn_backward import _SlangTileRNNBackward

    caller = _RNN_BWD_CALLERS.get(cell_type)
    if caller is None:
        caller = _SlangTileRNNBackward(cell_type)
        _RNN_BWD_CALLERS[cell_type] = caller
    return caller


def _rnn_forward_sequence(
    cell_type: str,
    x_seq: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor | None,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor | None,
    b_hh: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """Run forward RNN cell for all time steps, returning (h_seq, c_seq, h_last, c_last)."""
    is_lstm = cell_type == "lstm"
    fwd_caller = _get_rnn_template_caller(cell_type)
    seq_len, batch_size, _ = x_seq.shape
    hidden_size = h0.shape[-1]
    device = x_seq.device
    dtype = x_seq.dtype

    h_seq = torch.empty(seq_len, batch_size, hidden_size, device=device, dtype=dtype)
    c_seq: torch.Tensor | None = None
    if is_lstm:
        c_seq = torch.empty(
            seq_len, batch_size, hidden_size, device=device, dtype=dtype
        )

    h_prev = h0
    c_prev = c0
    for t in range(seq_len):
        h_new = h_seq[t]
        if is_lstm:
            c_new = c_seq[t]
            fwd_caller(
                x_t=x_seq[t],
                h_prev=h_prev,
                c_prev=c_prev,
                w_ih=w_ih,
                w_hh=w_hh,
                b_ih=b_ih,
                b_hh=b_hh,
                h_t=h_new,
                c_t=c_new,
            )
            c_prev = c_new
        else:
            fwd_caller(
                x_t=x_seq[t],
                h_prev=h_prev,
                c_prev=None,
                w_ih=w_ih,
                w_hh=w_hh,
                b_ih=b_ih,
                b_hh=b_hh,
                h_t=h_new,
                c_t=None,
            )
        h_prev = h_new

    return h_seq, c_seq, h_prev, c_prev


def _run_single_layer_bptt(
    cell_type: str,
    x_seq: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor | None,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor | None,
    b_hh: torch.Tensor | None,
    grad_output_seq: torch.Tensor | None,
    grad_h_n: torch.Tensor,
    grad_c_n: torch.Tensor | None,
) -> tuple:
    """Run BPTT for a single RNN layer/direction using Vulkan kernels.

    Args:
        cell_type: ``"lstm"``, ``"gru"``, ``"rnn_tanh"``, or ``"rnn_relu"``.
        x_seq: Input sequence [seq_len, batch, input_size].
        h0: Initial hidden state [batch, hidden_size].
        c0: Initial cell state [batch, hidden_size] (LSTM only).
        w_ih, w_hh, b_ih, b_hh: Weights and biases.
        grad_output_seq: Gradient w.r.t. output sequence [seq_len, batch, hidden_size].
        grad_h_n: Gradient w.r.t. final hidden state [batch, hidden_size].
        grad_c_n: Gradient w.r.t. final cell state [batch, hidden_size] (LSTM only).

    Returns:
        (grad_x_seq, grad_h0, grad_c0, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh)
    """
    is_lstm = cell_type == "lstm"
    has_bias = b_ih is not None and b_hh is not None
    seq_len, batch_size, input_size = x_seq.shape
    hidden_size = h0.shape[-1]
    device = x_seq.device
    dtype = x_seq.dtype

    bwd_caller = _get_rnn_bwd_caller(cell_type)

    # Phase 1: Recompute forward pass to get h_seq, c_seq
    h_seq, c_seq, _h_last, _c_last = _rnn_forward_sequence(
        cell_type,
        x_seq,
        h0,
        c0,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
    )

    # Phase 2: Initialize gradient accumulators
    grad_x_seq = torch.empty(
        seq_len, batch_size, input_size, device=device, dtype=dtype
    )
    grad_w_ih = torch.zeros_like(w_ih)
    grad_w_hh = torch.zeros_like(w_hh)
    grad_b_ih = torch.zeros_like(b_ih) if has_bias else None
    grad_b_hh = torch.zeros_like(b_hh) if has_bias else None

    # Phase 3: BPTT loop (reverse time)
    # We use a buffer for grad_h_prev/grad_c_prev that gets swapped each iter.
    grad_h_buf = torch.empty(batch_size, hidden_size, device=device, dtype=dtype)
    grad_c_buf: torch.Tensor | None = None
    if is_lstm:
        grad_c_buf = torch.empty(batch_size, hidden_size, device=device, dtype=dtype)

    # Initialize incoming gradient from h_n
    grad_h_buf.copy_(grad_h_n)
    if is_lstm and grad_c_n is not None:
        grad_c_buf.copy_(grad_c_n)

    for t in range(seq_len - 1, -1, -1):
        x_t = x_seq[t]
        if t > 0:
            h_prev_t = h_seq[t - 1]
            c_prev_t = c_seq[t - 1] if is_lstm and c_seq is not None else None
        else:
            h_prev_t = h0
            c_prev_t = c0

        # Add output gradient for this time step
        grad_h_t = grad_h_buf
        if grad_output_seq is not None:
            grad_h_t = grad_h_buf + grad_output_seq[t]

        # We need separate output buffers: the bwd kernel accumulates INTO
        # grad_h_prev, but we need that value as grad_h for the next iteration.
        # Use a temp buffer for grad_h_prev_out, then swap.
        grad_h_prev_out = torch.empty(
            batch_size, hidden_size, device=device, dtype=dtype
        )
        grad_c_prev_out: torch.Tensor | None = None
        if is_lstm:
            grad_c_prev_out = torch.empty(
                batch_size, hidden_size, device=device, dtype=dtype
            )

        bwd_caller(
            x_t=x_t,
            h_prev=h_prev_t,
            c_prev=c_prev_t,
            w_ih=w_ih,
            w_hh=w_hh,
            b_ih=b_ih,
            b_hh=b_hh,
            grad_h=grad_h_t,
            grad_c=grad_c_buf,
            grad_x=grad_x_seq[t],
            grad_h_prev=grad_h_prev_out,
            grad_c_prev=grad_c_prev_out,
            grad_w_ih=grad_w_ih,
            grad_w_hh=grad_w_hh,
            grad_b_ih=grad_b_ih,
            grad_b_hh=grad_b_hh,
        )

        # Swap: grad_h_prev_out becomes grad_h for the next (earlier) time step
        grad_h_buf.copy_(grad_h_prev_out)
        if is_lstm and grad_c_prev_out is not None:
            grad_c_buf.copy_(grad_c_prev_out)

    # After the loop, grad_h_buf/grad_c_buf hold gradients w.r.t. h0/c0
    grad_h0 = grad_h_buf
    grad_c0 = grad_c_buf

    return grad_x_seq, grad_h0, grad_c0, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh


def _run_vulkan_rnn_bwd(
    vf_name: str,
    grad_outputs,
    saved_inputs,
    has_biases: bool,
    batch_first: bool,
):
    """OP.25: Run Vulkan-native BPTT for a single-layer unidirectional RNN."""
    is_lstm = vf_name == "lstm"

    if batch_first:
        input_t = saved_inputs[0].transpose(0, 1).contiguous()
        if is_lstm:
            h0, c0 = saved_inputs[1], saved_inputs[2]
            param_start = 3
        else:
            h0 = saved_inputs[1]
            c0 = None
            param_start = 2
        grad_output_seq = grad_outputs[0].transpose(0, 1).contiguous()
    else:
        input_t = saved_inputs[0]
        if is_lstm:
            h0, c0 = saved_inputs[1], saved_inputs[2]
            param_start = 3
        else:
            h0 = saved_inputs[1]
            c0 = None
            param_start = 2
        grad_output_seq = grad_outputs[0]

    params_per_layer_dir = 4 if has_biases else 2
    if has_biases:
        w_ih = saved_inputs[param_start]
        w_hh = saved_inputs[param_start + 1]
        b_ih = saved_inputs[param_start + 2]
        b_hh = saved_inputs[param_start + 3]
    else:
        w_ih = saved_inputs[param_start]
        w_hh = saved_inputs[param_start + 1]
        b_ih = None
        b_hh = None

    if is_lstm:
        grad_h_n = grad_outputs[1] if len(grad_outputs) >= 2 else torch.zeros_like(h0)
        grad_c_n = grad_outputs[2] if len(grad_outputs) >= 3 else torch.zeros_like(c0)
    else:
        grad_h_n = grad_outputs[1] if len(grad_outputs) >= 2 else torch.zeros_like(h0)
        grad_c_n = None

    hidden_size = h0.shape[-1]
    if grad_output_seq is None or grad_output_seq.numel() == 0:
        grad_output_seq = torch.zeros(
            input_t.shape[0],
            input_t.shape[1],
            hidden_size,
            device=input_t.device,
            dtype=input_t.dtype,
        )

    grad_x_seq, grad_h0, grad_c0, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh = (
        _run_single_layer_bptt(
            vf_name,
            input_t,
            h0,
            c0,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            grad_output_seq,
            grad_h_n,
            grad_c_n,
        )
    )

    if batch_first:
        grad_x_seq = grad_x_seq.transpose(0, 1).contiguous()

    result = [grad_x_seq]
    if is_lstm:
        result.append(grad_h0)
        result.append(grad_c0)
    else:
        result.append(grad_h0)
    result.append(grad_w_ih)
    result.append(grad_w_hh)
    if has_biases:
        result.append(grad_b_ih)
        result.append(grad_b_hh)
    while len(result) < len(saved_inputs):
        result.append(torch.zeros_like(saved_inputs[len(result)]))

    return result

