"""CP.3 / OP.6 / T.10-fast — RNN forward pass via Vulkan Slang cell template.

Provides :func:`run_rnn_via_template` which replaces the CPU-roundtrip
in ``rnn.py`` with cell-by-cell dispatches through ``rnn_cell.py.jinja``
(CP.3) or fused multi-time-step dispatches through ``rnn_cell_fused.py.jinja``
(T.10-fast).

Dispatch count:
  - CP.3 per-time-step:  seq_len * num_layers * num_directions
  - T.10-fast fused:     num_layers * num_directions  (LSTM/GRU/RNN, hidden ≤ 1024)

Usage from ``rnn.py`` eager intercepts::

    from .rnn_template import run_rnn_via_template

    out = run_rnn_via_template("lstm", input_t, hx, params, ...)
"""

from __future__ import annotations

from typing import Optional

import torch

from ..vulkan_template_caller import (
    _can_use_fused_rnn_template,
    _SlangTileRNN,
    _SlangTileRNNFused,
)

# Cached template callers — created lazily on first use.
_RNN_TEMPLATE_CALLERS: dict[str, _SlangTileRNN] = {}
_RNN_FUSED_CALLERS: dict[str, _SlangTileRNNFused] = {}


def _get_caller(cell_type: str) -> _SlangTileRNN:
    """Return (or create) a cached _SlangTileRNN caller for `cell_type`."""
    caller = _RNN_TEMPLATE_CALLERS.get(cell_type)
    if caller is None:
        caller = _SlangTileRNN(cell_type)
        _RNN_TEMPLATE_CALLERS[cell_type] = caller
    return caller


def _get_fused_caller(cell_type: str) -> _SlangTileRNNFused:
    """Return (or create) a cached _SlangTileRNNFused caller for `cell_type`."""
    caller = _RNN_FUSED_CALLERS.get(cell_type)
    if caller is None:
        caller = _SlangTileRNNFused(cell_type)
        _RNN_FUSED_CALLERS[cell_type] = caller
    return caller


def _run_layer_direction_fused(
    fused_caller: _SlangTileRNNFused,
    x_seq: torch.Tensor,
    h0_ld: torch.Tensor,
    c0_ld: torch.Tensor | None,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor | None,
    b_hh: torch.Tensor | None,
    out_seq: torch.Tensor,
    h_last_ld: torch.Tensor,
    c_last_ld: torch.Tensor | None,
    direction: int = 0,
) -> None:
    """Dispatch one fused (layer, direction) — one kernel call.

    Args:
        fused_caller: Pre-configured _SlangTileRNNFused instance.
        x_seq: Input sequence [seq_len, batch, input_size].
        h0_ld: Initial hidden state for this (layer, direction) [batch, hidden_size].
        c0_ld: Initial cell state [batch, hidden_size] (LSTM only, else None).
        w_ih, w_hh, b_ih, b_hh: Weights and biases.
        out_seq: Output buffer [batch, seq_len, hidden_size] (pre-allocated).
        h_last_ld: Final hidden state buffer [batch, hidden_size] (pre-allocated).
        c_last_ld: Final cell state buffer [batch, hidden_size] (LSTM only, else None).
        direction: 0=forward, 1=reverse (M10 IRnnDirection).
    """
    fused_caller(
        x_seq=x_seq,
        h0=h0_ld,
        c0=c0_ld,
        w_ih=w_ih,
        w_hh=w_hh,
        b_ih=b_ih,
        b_hh=b_hh,
        out_seq=out_seq,
        h_last=h_last_ld,
        c_last=c_last_ld,
        direction=direction,
    )


def run_rnn_via_template(
    vf_name: str,
    input_t: torch.Tensor,
    hx,
    params: list[torch.Tensor],
    has_biases: bool,
    num_layers: int,
    bidirectional: bool,
    batch_first: bool,
    *,
    dropout: float = 0.0,
    train: bool = True,
):
    """Run a full RNN forward pass using the Vulkan Slang cell template.

    T.10-fast: For LSTM/GRU/RNN with hidden_size ≤ 1024, uses the fused
    multi-time-step template (one dispatch per layer×direction).
    Otherwise falls back to CP.3 per-time-step dispatch.

    Args:
        vf_name: ``"lstm"``, ``"gru"``, ``"rnn_tanh"``, or ``"rnn_relu"``.
        input_t: Input tensor [batch, seq, input_size] or [seq, batch, input_size].
        hx: Initial hidden state (h0 [, c0] for LSTM).
        params: Flat list of weight/bias tensors in PyTorch canonical order.
        has_biases: Whether bias tensors are included.
        num_layers: Number of stacked RNN layers.
        bidirectional: Whether to include reverse direction.
        batch_first: Batch dimension first layout.
        dropout: Dropout probability (ignored in eval; no-op for now).
        train: Training mode (ignored for now; always computes).

    Returns:
        (output, h_n, c_n) tuple where `c_n` is None for non-LSTM types.
    """
    is_lstm = vf_name == "lstm"
    is_gru = vf_name == "gru"

    # ── Resolve cell type and unpack initial state ──────────────────
    if is_lstm:
        cell_type = "lstm"
        h0, c0 = hx
    elif is_gru:
        cell_type = "gru"
        h0 = hx
        c0 = None
    elif vf_name == "rnn_tanh":
        cell_type = "rnn_tanh"
        h0 = hx
        c0 = None
    elif vf_name == "rnn_relu":
        cell_type = "rnn_relu"
        h0 = hx
        c0 = None
    else:
        raise ValueError(f"Unknown vf_name: {vf_name}")

    # ── Shape extraction ────────────────────────────────────────────
    if batch_first:
        batch_size, seq_len, input_size = input_t.shape
    else:
        seq_len, batch_size, input_size = input_t.shape

    hidden_size = h0.shape[-1]
    D = 2 if bidirectional else 1

    # ── Tracing guard (Dynamo / AOTAutograd fake-tensor mode) ───────
    # When Dynamo traces _VF.lstm with a mix of FakeTensors (input, params)
    # and real Vulkan tensors (h0/c0 from nn.LSTM state), the PF.51 guard in
    # dispatch.py fires because not ALL Vulkan tensors are FakeTensors.
    # Short-circuit here: return empty tensors with the correct output shapes.
    # This is sufficient for Dynamo to infer output shapes without dispatching.
    from ..fx_passes.eager._common import _has_real_vulkan_storage

    if not _has_real_vulkan_storage(input_t):
        if batch_first:
            output = torch.empty(
                batch_size,
                seq_len,
                hidden_size * D,
                device=input_t.device,
                dtype=input_t.dtype,
            )
        else:
            output = torch.empty(
                seq_len,
                batch_size,
                hidden_size * D,
                device=input_t.device,
                dtype=input_t.dtype,
            )
        h_n = torch.empty(
            num_layers * D,
            batch_size,
            hidden_size,
            device=input_t.device,
            dtype=input_t.dtype,
        )
        c_n: Optional[torch.Tensor] = None
        if is_lstm:
            c_n = torch.empty_like(h_n)
        return output, h_n, c_n

    # T.10-fast: decide dispatch strategy.
    use_fused = _can_use_fused_rnn_template(cell_type, hidden_size)

    if use_fused:
        fused_caller = _get_fused_caller(cell_type)
    else:
        caller = _get_caller(cell_type)

    # ── Pre-allocate output tensors ─────────────────────────────────
    if batch_first:
        output = torch.empty(
            batch_size,
            seq_len,
            hidden_size * D,
            device=input_t.device,
            dtype=input_t.dtype,
        )
    else:
        output = torch.empty(
            seq_len,
            batch_size,
            hidden_size * D,
            device=input_t.device,
            dtype=input_t.dtype,
        )

    h_n = torch.empty(
        num_layers * D,
        batch_size,
        hidden_size,
        device=input_t.device,
        dtype=input_t.dtype,
    )
    c_n: Optional[torch.Tensor] = None
    if is_lstm:
        c_n = torch.empty_like(h_n)

    # ── Parameter indexing ──────────────────────────────────────────
    params_per_layer_dir = 4 if has_biases else 2

    def _get_layer_params(layer_idx: int, direction_idx: int):
        """Extract (w_ih, w_hh, b_ih, b_hh) for a given (layer, direction)."""
        base = (layer_idx * D + direction_idx) * params_per_layer_dir
        if has_biases:
            return (
                params[base],
                params[base + 1],
                params[base + 2],
                params[base + 3],
            )
        return params[base], params[base + 1], None, None

    # ── Normalise input to [seq_len, batch, input_size] ─────────────
    if batch_first:
        input_seq = input_t.transpose(0, 1).contiguous()  # [seq, batch, in]
    else:
        input_seq = input_t  # already [seq, batch, in]

    # ── Layer × direction loop ──────────────────────────────────────
    for layer in range(num_layers):
        # Determine the input sequence for this layer.
        # Layer 0 uses the original input; subsequent layers use the
        # accumulated output from the previous layer.
        if layer == 0:
            layer_input = input_seq
            layer_input_size = input_size
        else:
            # Previous layer output has both directions interleaved
            # in the last dim: [seq_len, batch, hidden_size * D].
            layer_input = output.transpose(0, 1).contiguous()  # [seq, batch, h*D]
            layer_input_size = hidden_size * D

        for direction in range(D):
            w_ih, w_hh, b_ih, b_hh = _get_layer_params(layer, direction)

            # ── T.10-fast fused dispatch ────────────────────────────
            if use_fused:
                # c0_ld is only meaningful for LSTM; GRU/RNN pass None.
                c0_ld: Optional[torch.Tensor]
                if is_lstm:
                    c0_ld = c0[layer * D + direction]
                else:
                    c0_ld = None

                # For the reverse direction, flip the input sequence.
                if direction == 0:
                    x_for_direction = layer_input  # [seq, batch, in]
                else:
                    x_for_direction = torch.flip(layer_input, [0]).contiguous()

                # Pre-allocate fused output buffers.
                out_ld = torch.empty(
                    batch_size,
                    seq_len,
                    hidden_size,
                    device=input_t.device,
                    dtype=input_t.dtype,
                )
                h_last_ld = torch.empty(
                    batch_size,
                    hidden_size,
                    device=input_t.device,
                    dtype=input_t.dtype,
                )
                c_last_ld: Optional[torch.Tensor] = None
                if is_lstm:
                    c_last_ld = torch.empty_like(h_last_ld)

                _run_layer_direction_fused(
                    fused_caller=fused_caller,
                    x_seq=x_for_direction,
                    h0_ld=h0[layer * D + direction],
                    c0_ld=c0_ld,
                    w_ih=w_ih,
                    w_hh=w_hh,
                    b_ih=b_ih,
                    b_hh=b_hh,
                    out_seq=out_ld,
                    h_last_ld=h_last_ld,
                    c_last_ld=c_last_ld,
                    direction=direction,
                )

                # Write into the main output tensor (interleave directions).
                if batch_first:
                    if direction == 0:
                        output[:, :, :hidden_size] = out_ld
                    else:
                        output[:, :, hidden_size:] = out_ld
                else:
                    if direction == 0:
                        output[:, :, :hidden_size] = out_ld.permute(1, 0, 2)
                    else:
                        output[:, :, hidden_size:] = out_ld.permute(1, 0, 2)

                h_n[layer * D + direction] = h_last_ld
                if is_lstm and c_n is not None and c_last_ld is not None:
                    c_n[layer * D + direction] = c_last_ld

            # ── CP.3 per-time-step dispatch (fallback) ──────────────
            else:
                # Initial hidden/cell state for this layer+direction
                h_state = h0[layer * D + direction].clone()
                c_state = c0[layer * D + direction].clone() if is_lstm else None

                # Time-step iteration order
                if direction == 0:
                    time_steps = range(seq_len)
                else:
                    time_steps = range(seq_len - 1, -1, -1)

                for t in time_steps:
                    # Extract input for this time step
                    # layer_input is [seq_len, batch, input_size]
                    x_t = layer_input[t, :, :]  # [batch, input_size]

                    # Allocate output buffers for this cell step
                    h_new = torch.empty_like(h_state)
                    c_new = torch.empty_like(h_state) if is_lstm else None

                    # ── Dispatch the Slang template for this cell ───
                    caller(
                        x_t=x_t,
                        h_prev=h_state,
                        c_prev=c_state,
                        w_ih=w_ih,
                        w_hh=w_hh,
                        b_ih=b_ih,
                        b_hh=b_hh,
                        h_t=h_new,
                        c_t=c_new,
                    )

                    # Update state for next time step
                    h_state = h_new
                    if is_lstm:
                        c_state = c_new

                    # Write to output tensor (interleave directions)
                    if batch_first:
                        if direction == 0:
                            output[:, t, :hidden_size] = h_new
                        else:
                            output[:, t, hidden_size:] = h_new
                    else:
                        if direction == 0:
                            output[t, :, :hidden_size] = h_new
                        else:
                            output[t, :, hidden_size:] = h_new

                # Store final hidden/cell state
                h_n[layer * D + direction] = h_state
                if is_lstm and c_n is not None and c_state is not None:
                    c_n[layer * D + direction] = c_state

    return output, h_n, c_n
