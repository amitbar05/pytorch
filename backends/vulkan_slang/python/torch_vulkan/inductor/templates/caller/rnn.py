"""RNN cell template callers.

Provides rendering, dispatch, and installation for the RNN cell Slang templates
(LSTM, GRU, RNN-tanh, RNN-relu, and fused multi-time-step variants).

B2: cell_type selects a Slang `IRnnCell` concrete struct (LstmCellImpl /
GruCellImpl / RnnTanhCellImpl / RnnReluCellImpl) via the slangc entry-point
name (``computeMain<LstmCellImpl>`` etc.) — matching the `IOptimizer`
pattern in ``caller/optimizer.py``. The rendered Slang source no longer
varies by cell_type (all four Impl structs are always emitted); only
`hidden_size` / `input_size` / `dtype` are Jinja-substituted (compile-time
numeric tunables, which anti-goal #5 allows).
"""

from __future__ import annotations

import os
import struct
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from ...buffer_pool import pool_acquire
from ...vulkan_template import _load_slang_template
from ...vulkan_template_caller import _dtype_to_slang

_rnn_cell_cache: dict[tuple, str] = {}
_rnn_installed = False

# Cell types the template supports.
_RNN_CELL_TYPES: tuple[str, ...] = ("lstm", "gru", "rnn_tanh", "rnn_relu")

# B2: cell_type -> Slang IRnnCell concrete struct name (the generic type
# argument substituted into the slangc entry-point name).
_CELL_TYPE_TO_IMPL: dict[str, str] = {
    "lstm": "LstmCellImpl",
    "gru": "GruCellImpl",
    "rnn_tanh": "RnnTanhCellImpl",
    "rnn_relu": "RnnReluCellImpl",
}

# Fixed array bound for ih_gate/hh_gate in the template — must be >= the
# largest GATE_SIZE among the IRnnCell implementations (LSTM, 4).
_GATE_SIZE_MAX = 4


def _rnn_cell_entry(cell_type: str) -> str:
    """Return the slangc entry-point string for `cell_type`, e.g.
    ``"computeMain<LstmCellImpl>"``."""
    impl = _CELL_TYPE_TO_IMPL.get(cell_type)
    if impl is None:
        raise ValueError(
            f"Unknown RNN cell_type '{cell_type}'. Must be one of: {_RNN_CELL_TYPES}"
        )
    return f"computeMain<{impl}>"


def _render_rnn_cell(
    cell_type: str,
    hidden_size: int,
    input_size: int,
    dtype: str = "float",
) -> str:
    """Render the rnn_cell Jinja2 template.

    B2: the rendered source is identical for every `cell_type` — only
    `hidden_size` / `input_size` / `dtype` (numeric tunables) vary the
    output. `cell_type` is still accepted (and validated) here so the
    cache key / caller API stay stable; it selects the slangc entry
    point (see `_rnn_cell_entry`), not the Slang source.

    Args:
        cell_type: One of ``"lstm"``, ``"gru"``, ``"rnn_tanh"``, ``"rnn_relu"``.
        hidden_size: Number of hidden units.
        input_size: Input feature dimension.
        dtype: Slang type string — ``"float"`` (f32) or ``"half"`` (f16).

    Returns:
        Rendered Slang source string ready for SPIR-V compilation.
    """
    from jinja2 import Environment

    if cell_type not in _RNN_CELL_TYPES:
        raise ValueError(
            f"Unknown RNN cell_type '{cell_type}'. Must be one of: {_RNN_CELL_TYPES}"
        )

    # B2: source no longer depends on cell_type — cache on the
    # (hidden_size, input_size, dtype) tuple only, but keep cell_type
    # in the lookup key so the cache dict shape stays familiar to callers.
    key = (hidden_size, input_size, dtype)
    if key in _rnn_cell_cache:
        return _rnn_cell_cache[key]

    src = _load_slang_template("rnn_cell")
    if not src:
        raise RuntimeError("rnn_cell.slang template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        hidden_size=hidden_size,
        input_size=input_size,
        gate_size_max=_GATE_SIZE_MAX,
        dtype=dtype,
    )
    _rnn_cell_cache[key] = rendered
    return rendered


def _dispatch_rnn_cell(
    cell_type: str,
    hidden_size: int,
    input_size: int,
    batch_size: int,
    has_bias: bool,
    dtype: str,
    x_t: torch.Tensor,
    h_prev: torch.Tensor,
    c_prev: torch.Tensor | None,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor | None,
    b_hh: torch.Tensor | None,
    h_t: torch.Tensor,
    c_t: torch.Tensor | None,
    src: str | None = None,
    cache_key: str | None = None,
) -> None:
    """Dispatch a single RNN cell computation via the Slang template.

    Computes h_t (and c_t for LSTM) from x_t, h_prev, c_prev, weights, and biases.
    One dispatch processes ALL batch elements in parallel (grid_x = batch_size).

    Args:
        cell_type: ``"lstm"``, ``"gru"``, ``"rnn_tanh"``, or ``"rnn_relu"``.
        hidden_size: Number of hidden units.
        input_size: Input feature dimension.
        batch_size: Number of batch elements.
        has_bias: Whether bias tensors are provided.
        dtype: Slang type string.
        x_t: Input tensor [batch, input_size].
        h_prev: Previous hidden state [batch, hidden_size].
        c_prev: Previous cell state [batch, hidden_size] (LSTM only, else None).
        w_ih: Input-to-hidden weight [gate_size * hidden_size, input_size].
        w_hh: Hidden-to-hidden weight [gate_size * hidden_size, hidden_size].
        b_ih: Input-to-hidden bias [gate_size * hidden_size] (or None).
        b_hh: Hidden-to-hidden bias [gate_size * hidden_size] (or None).
        h_t: Output hidden state [batch, hidden_size].
        c_t: Output cell state [batch, hidden_size] (LSTM only, else None).
        src: Pre-rendered Slang source (rendered if None).
        cache_key: SPIR-V cache key (computed if None).
    """
    from ...runtime import compile_and_dispatch

    is_lstm = cell_type == "lstm"

    if src is None or cache_key is None:
        src = _render_rnn_cell(
            cell_type=cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            dtype=dtype,
        )
        cache_key = f"slang_rnn_{cell_type}_h{hidden_size}_i{input_size}_{dtype}"

    # B2: PC is always the full 8-field (LSTM-shaped) layout — no more
    # Jinja-conditional struct fields.  stride_c is unused by the shader
    # for non-cell-state cells but always present so the layout is
    # uniform across cell types (matches B1's "always full ParamConfig").
    stride_w_ih = input_size
    stride_w_hh = hidden_size
    stride_x = input_size
    stride_h = hidden_size
    stride_c = hidden_size

    pc = struct.pack(
        "8I",
        hidden_size,
        input_size,
        stride_w_ih,
        stride_w_hh,
        stride_x,
        stride_h,
        stride_c,
        1 if has_bias else 0,
    )

    # Ensure all inputs are contiguous.
    if not x_t.is_contiguous():
        x_t = x_t.contiguous()
    if not h_prev.is_contiguous():
        h_prev = h_prev.contiguous()
    if is_lstm and c_prev is not None and not c_prev.is_contiguous():
        c_prev = c_prev.contiguous()
    if not w_ih.is_contiguous():
        w_ih = w_ih.contiguous()
    if not w_hh.is_contiguous():
        w_hh = w_hh.contiguous()
    if not h_t.is_contiguous():
        h_t = h_t.contiguous()
    if is_lstm and c_t is not None and not c_t.is_contiguous():
        c_t = c_t.contiguous()

    # B2: Build buffer list matching the now cell-type-independent
    # KernelArgs field order: [x_t, h_prev, c_prev, w_ih, w_hh, b_ih,
    # b_hh, h_t, c_t].  c_prev/c_t are always present — real buffers for
    # LSTM, zero-sized dummies otherwise (the shader never reads/writes
    # them when !Cell.HAS_CELL_STATE).
    buffers: list[torch.Tensor] = [x_t, h_prev]
    if is_lstm:
        buffers.append(
            c_prev if c_prev is not None else torch.empty(0, device=x_t.device)
        )
    else:
        buffers.append(torch.empty(0, device=x_t.device))
    buffers.extend([w_ih, w_hh])
    # Bias tensors — use zero-sized placeholder if not provided.
    if has_bias and b_ih is not None:
        buffers.append(b_ih.contiguous() if not b_ih.is_contiguous() else b_ih)
    else:
        buffers.append(torch.empty(0, device=x_t.device))
    if has_bias and b_hh is not None:
        buffers.append(b_hh.contiguous() if not b_hh.is_contiguous() else b_hh)
    else:
        buffers.append(torch.empty(0, device=x_t.device))
    buffers.append(h_t)
    if is_lstm:
        buffers.append(c_t if c_t is not None else torch.empty(0, device=x_t.device))
    else:
        buffers.append(torch.empty(0, device=x_t.device))

    # B2: KernelArgs always declares both h_t and c_t as RW bindings (the
    # struct layout is now cell-type-independent), so both trailing
    # buffer slots are outputs regardless of cell type. For non-LSTM,
    # c_t is a zero-sized dummy — marking it dirty is a no-op.
    num_outputs = 2

    grid_x = batch_size
    grid_y = 1
    grid_z = 1

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=num_outputs,
        entry=_rnn_cell_entry(cell_type),
        cache_key=cache_key,
    )


class _SlangTileRNN:
    """Picklable callable for RNN cell template dispatch.

    Each instance is configured for a specific cell_type.  The callable
    interface accepts the per-time-step tensors and returns the updated
    hidden/cell state.

    Caches the rendered Slang source per (dtype, hidden_size, input_size)
    tuple so repeated dispatches for the same cell skip the Jinja render.
    """

    __slots__ = ("cell_type", "__name__", "_per_spec")

    def __init__(self, cell_type: str):
        if cell_type not in _RNN_CELL_TYPES:
            raise ValueError(
                f"Unknown RNN cell_type '{cell_type}'. "
                f"Must be one of: {_RNN_CELL_TYPES}"
            )
        self.cell_type = cell_type
        self.__name__ = f"slang_rnn_{cell_type}"
        self._per_spec: dict[tuple, tuple[str, str]] = {}

    def _src_and_key(
        self, hidden_size: int, input_size: int, dtype: str
    ) -> tuple[str, str]:
        spec_key = (hidden_size, input_size, dtype)
        cached = self._per_spec.get(spec_key)
        if cached is not None:
            return cached

        src = _render_rnn_cell(
            cell_type=self.cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            dtype=dtype,
        )
        cache_key = f"slang_rnn_{self.cell_type}_h{hidden_size}_i{input_size}_{dtype}"
        cached = (src, cache_key)
        self._per_spec[spec_key] = cached
        return cached

    def __call__(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor | None,
        w_ih: torch.Tensor,
        w_hh: torch.Tensor,
        b_ih: torch.Tensor | None,
        b_hh: torch.Tensor | None,
        h_t: torch.Tensor,
        c_t: torch.Tensor | None,
    ) -> None:
        """Dispatch one RNN cell step.

        Args:
            x_t: Input [batch, input_size].
            h_prev: Previous hidden state [batch, hidden_size].
            c_prev: Previous cell state (LSTM only).
            w_ih: Input-to-hidden weight.
            w_hh: Hidden-to-hidden weight.
            b_ih: Input-to-hidden bias (or None).
            b_hh: Hidden-to-hidden bias (or None).
            h_t: Output hidden state [batch, hidden_size].
            c_t: Output cell state (LSTM only).
        """
        batch_size = x_t.shape[0]
        hidden_size = h_prev.shape[-1]
        input_size = x_t.shape[-1]
        dtype_s = _dtype_to_slang(x_t.dtype)
        has_bias = b_ih is not None and b_hh is not None

        src, cache_key = self._src_and_key(hidden_size, input_size, dtype_s)

        _dispatch_rnn_cell(
            cell_type=self.cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            batch_size=batch_size,
            has_bias=has_bias,
            dtype=dtype_s,
            x_t=x_t,
            h_prev=h_prev,
            c_prev=c_prev,
            w_ih=w_ih,
            w_hh=w_hh,
            b_ih=b_ih,
            b_hh=b_hh,
            h_t=h_t,
            c_t=c_t,
            src=src,
            cache_key=cache_key,
        )

    def __reduce__(self):
        return (_SlangTileRNN, (self.cell_type,))


# Maximum hidden_size for the fused template (groupshared memory budget).
# 2 × float[1024] = 8 KB, well within the Vulkan minimum of 32 KB.
_FUSED_RNN_MAX_HIDDEN_SIZE = 1024

_rnn_cell_fused_cache: dict[tuple, str] = {}


def _render_rnn_cell_fused(
    cell_type: str,
    hidden_size: int,
    input_size: int,
    seq_len: int,
    dtype: str = "float",
) -> str:
    """Render the rnn_cell_fused Jinja2 template for multi-time-step dispatch.

    B2: like `_render_rnn_cell`, the rendered source is identical for
    every `cell_type` — only `hidden_size` / `input_size` / `dtype` vary
    the output. `cell_type` selects the slangc entry point (see
    `_rnn_cell_entry`), not the Slang source.

    Args:
        cell_type: ``"lstm"``, ``"gru"``, ``"rnn_tanh"``, or ``"rnn_relu"``.
        hidden_size: Number of hidden units (≤ 1024).
        input_size: Input feature dimension.
        seq_len: Number of time steps to fuse into one kernel.
        dtype: Slang type string — ``"float"`` (f32) or ``"half"`` (f16).

    Returns:
        Rendered Slang source string ready for SPIR-V compilation.
    """
    from jinja2 import Environment

    if cell_type not in _RNN_CELL_TYPES:
        raise ValueError(
            f"Fused RNN template only supports {_RNN_CELL_TYPES}, got '{cell_type}'"
        )
    if hidden_size > _FUSED_RNN_MAX_HIDDEN_SIZE:
        raise ValueError(
            f"Fused RNN template requires hidden_size ≤ {_FUSED_RNN_MAX_HIDDEN_SIZE}, "
            f"got {hidden_size}"
        )

    key = (hidden_size, dtype)
    if key in _rnn_cell_fused_cache:
        return _rnn_cell_fused_cache[key]

    src = _load_slang_template("rnn_cell_fused")
    if not src:
        raise RuntimeError("rnn_cell_fused.slang template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        hidden_size=hidden_size,
        input_size=input_size,
        gate_size_max=_GATE_SIZE_MAX,
        dtype=dtype,
    )
    _rnn_cell_fused_cache[key] = rendered
    return rendered


def _dispatch_rnn_cell_fused(
    cell_type: str,
    hidden_size: int,
    input_size: int,
    seq_len: int,
    batch_size: int,
    has_bias: bool,
    dtype: str,
    x_seq: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor | None,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor | None,
    b_hh: torch.Tensor | None,
    out_seq: torch.Tensor,
    h_last: torch.Tensor,
    c_last: torch.Tensor | None,
    direction: int = 0,
    src: str | None = None,
    cache_key: str | None = None,
) -> None:
    """Dispatch a fused multi-time-step RNN cell computation.

    Processes ALL ``seq_len`` time steps for ALL batch elements in ONE kernel
    dispatch.  One workgroup per batch element, internal loop over time steps.

    Args:
        cell_type: ``"lstm"``, ``"gru"``, ``"rnn_tanh"``, or ``"rnn_relu"``.
        hidden_size: Number of hidden units.
        input_size: Input feature dimension.
        seq_len: Number of time steps.
        batch_size: Number of batch elements.
        has_bias: Whether bias tensors are provided.
        dtype: Slang type string.
        x_seq: Input sequence [seq_len, batch, input_size].
        h0: Initial hidden state [batch, hidden_size].
        c0: Initial cell state [batch, hidden_size] (LSTM only, else None).
        w_ih: Input-to-hidden weight [gate_size*hidden_size, input_size].
        w_hh: Hidden-to-hidden weight [gate_size*hidden_size, hidden_size].
        b_ih: Input-to-hidden bias [gate_size*hidden_size] (or None).
        b_hh: Hidden-to-hidden bias [gate_size*hidden_size] (or None).
        out_seq: Output sequence [batch, seq_len, hidden_size] (pre-allocated).
        h_last: Final hidden state [batch, hidden_size] (pre-allocated).
        c_last: Final cell state [batch, hidden_size] (LSTM only, else None).
        direction: 0=forward, 1=reverse (M10 IRnnDirection).
        src: Pre-rendered Slang source (rendered if None).
        cache_key: SPIR-V cache key (computed if None).
    """
    from ...runtime import compile_and_dispatch

    is_lstm = cell_type == "lstm"

    if hidden_size > _FUSED_RNN_MAX_HIDDEN_SIZE:
        raise ValueError(
            f"Fused RNN template requires hidden_size ≤ {_FUSED_RNN_MAX_HIDDEN_SIZE}, "
            f"got {hidden_size}"
        )

    if src is None or cache_key is None:
        src = _render_rnn_cell_fused(
            cell_type=cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            seq_len=seq_len,
            dtype=dtype,
        )
        # Note: seq_len is a push constant, NOT embedded in the Slang source,
        # so the source is identical for all sequence lengths.  B2: the
        # source is also identical across cell types now (cell_type only
        # selects the slangc entry point — see `_rnn_cell_entry`), so the
        # cache key tracks only source-varying parameters (hidden_size,
        # input_size, dtype).
        cache_key = f"slang_rnn_fused_h{hidden_size}_i{input_size}_{dtype}"

    # Push constants layout (11 uint32_t fields — same for all cell types).
    # PC: hidden_size, input_size, seq_len, stride_w_ih, stride_w_hh,
    #     stride_x_tbatch, stride_x_batch, stride_h_batch,
    #     stride_out_tbatch, stride_out_batch, has_bias
    stride_w_ih = input_size
    stride_w_hh = hidden_size
    stride_x_tbatch = batch_size * input_size
    stride_x_batch = input_size
    stride_h_batch = hidden_size
    stride_out_tbatch = hidden_size
    stride_out_batch = seq_len * hidden_size

    pc = struct.pack(
        "12I",
        hidden_size,
        input_size,
        seq_len,
        stride_w_ih,
        stride_w_hh,
        stride_x_tbatch,
        stride_x_batch,
        stride_h_batch,
        stride_out_tbatch,
        stride_out_batch,
        1 if has_bias else 0,
        direction,  # M10: 0=forward, 1=reverse — runtime gate (anti-goal #5)
    )

    # Ensure all inputs are contiguous.
    if not x_seq.is_contiguous():
        x_seq = x_seq.contiguous()
    if not h0.is_contiguous():
        h0 = h0.contiguous()
    if is_lstm and c0 is not None and not c0.is_contiguous():
        c0 = c0.contiguous()
    if not w_ih.is_contiguous():
        w_ih = w_ih.contiguous()
    if not w_hh.is_contiguous():
        w_hh = w_hh.contiguous()
    if not out_seq.is_contiguous():
        out_seq = out_seq.contiguous()
    if not h_last.is_contiguous():
        h_last = h_last.contiguous()
    if is_lstm and c_last is not None and not c_last.is_contiguous():
        c_last = c_last.contiguous()

    # Build buffer list matching KernelArgs field order:
    # [x_seq, h0, c0_or_dummy, w_ih, w_hh, b_ih, b_hh, out_seq, h_last, c_last_or_dummy]
    # For non-LSTM cells, slot 2 (c0) and slot 9 (c_last) are unused placeholders.
    c0_buf: torch.Tensor
    c_last_buf: torch.Tensor
    if is_lstm and c0 is not None and c_last is not None:
        c0_buf = c0
        c_last_buf = c_last
    else:
        c0_buf = torch.empty(0, device=x_seq.device)
        c_last_buf = torch.empty(0, device=x_seq.device)

    buffers: list[torch.Tensor] = [x_seq, h0, c0_buf, w_ih, w_hh]
    if has_bias and b_ih is not None:
        buffers.append(b_ih.contiguous() if not b_ih.is_contiguous() else b_ih)
    else:
        buffers.append(torch.empty(0, device=x_seq.device))
    if has_bias and b_hh is not None:
        buffers.append(b_hh.contiguous() if not b_hh.is_contiguous() else b_hh)
    else:
        buffers.append(torch.empty(0, device=x_seq.device))
    buffers.extend([out_seq, h_last, c_last_buf])

    # B2: KernelArgs always declares out_seq/h_last/c_last as RW bindings
    # (struct layout is cell-type-independent now), so all three trailing
    # buffer slots are outputs regardless of cell type.  For non-LSTM,
    # c_last is a zero-sized dummy — marking it dirty is a no-op.
    num_outputs = 3

    grid_x = batch_size
    grid_y = 1
    grid_z = 1

    compile_and_dispatch(
        src,
        buffers,
        grid_x,
        grid_y,
        grid_z,
        push_constants=pc,
        num_outputs=num_outputs,
        entry=_rnn_cell_entry(cell_type),
        cache_key=cache_key,
    )


def _can_use_fused_rnn_template(cell_type: str, hidden_size: int) -> bool:
    """Check whether the fused template can be used for these parameters.

    T.10-fast supports LSTM, GRU, RNN-tanh, and RNN-relu.
    """
    return cell_type in _RNN_CELL_TYPES and hidden_size <= _FUSED_RNN_MAX_HIDDEN_SIZE


class _SlangTileRNNFused:
    """Picklable callable for fused multi-time-step RNN cell dispatch.

    Each instance is configured for a specific cell_type.  Unlike
    :class:`_SlangTileRNN`, which dispatches once per time step, this
    callable processes the entire sequence in one kernel dispatch.

    Supports LSTM, GRU, RNN-tanh, and RNN-relu.

    Caches the rendered Slang source per (hidden_size, input_size, dtype) tuple.
    """

    __slots__ = ("cell_type", "__name__", "_per_spec")

    def __init__(self, cell_type: str):
        if cell_type not in _RNN_CELL_TYPES:
            raise ValueError(
                f"Fused RNN template only supports {_RNN_CELL_TYPES}, got '{cell_type}'"
            )
        self.cell_type = cell_type
        self.__name__ = f"slang_rnn_fused_{cell_type}"
        self._per_spec: dict[tuple, tuple[str, str]] = {}

    def _src_and_key(
        self, hidden_size: int, input_size: int, seq_len: int, dtype: str
    ) -> tuple[str, str]:
        spec_key = (hidden_size, input_size, dtype)
        cached = self._per_spec.get(spec_key)
        if cached is not None:
            return cached

        src = _render_rnn_cell_fused(
            cell_type=self.cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            seq_len=seq_len,
            dtype=dtype,
        )
        cache_key = (
            f"slang_rnn_fused_{self.cell_type}_h{hidden_size}_i{input_size}_{dtype}"
        )
        cached = (src, cache_key)
        self._per_spec[spec_key] = cached
        return cached

    def __call__(
        self,
        x_seq: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor | None,
        w_ih: torch.Tensor,
        w_hh: torch.Tensor,
        b_ih: torch.Tensor | None,
        b_hh: torch.Tensor | None,
        out_seq: torch.Tensor,
        h_last: torch.Tensor,
        c_last: torch.Tensor | None,
        direction: int = 0,
    ) -> None:
        """Dispatch one fused multi-time-step RNN cell.

        Args:
            x_seq: Input sequence [seq_len, batch, input_size].
            h0: Initial hidden state [batch, hidden_size].
            c0: Initial cell state [batch, hidden_size] (LSTM only, else None).
            w_ih: Input-to-hidden weight.
            w_hh: Hidden-to-hidden weight.
            b_ih: Input-to-hidden bias (or None).
            b_hh: Hidden-to-hidden bias (or None).
            out_seq: Output buffer [batch, seq_len, hidden_size].
            h_last: Final hidden state buffer [batch, hidden_size].
            c_last: Final cell state buffer [batch, hidden_size] (LSTM only, else None).
            direction: 0=forward, 1=reverse (M10 IRnnDirection).
        """
        batch_size = x_seq.shape[1]
        seq_len = x_seq.shape[0]
        hidden_size = h0.shape[-1]
        input_size = x_seq.shape[-1]
        dtype_s = _dtype_to_slang(x_seq.dtype)
        has_bias = b_ih is not None and b_hh is not None

        src, cache_key = self._src_and_key(hidden_size, input_size, seq_len, dtype_s)

        _dispatch_rnn_cell_fused(
            cell_type=self.cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            seq_len=seq_len,
            batch_size=batch_size,
            has_bias=has_bias,
            dtype=dtype_s,
            x_seq=x_seq,
            h0=h0,
            c0=c0,
            w_ih=w_ih,
            w_hh=w_hh,
            b_ih=b_ih,
            b_hh=b_hh,
            out_seq=out_seq,
            h_last=h_last,
            c_last=c_last,
            direction=direction,
            src=src,
            cache_key=cache_key,
        )

    def __reduce__(self):
        return (_SlangTileRNNFused, (self.cell_type,))

def install_external_rnn() -> None:
    """Register RNN cell template as the lowering route for RNN ops.

    Called from ``lowerings/__init__.py`` at backend init.  Replaces the
    CPU-roundtrip fallback path in ``lowerings/rnn.py`` with a Vulkan-native
    template dispatch that keeps data on-device.

    Safe to call multiple times — only installs once.
    """
    global _rnn_installed
    if _rnn_installed:
        return
    _rnn_installed = True

    # Pre-render all cell types for the two common dtypes at standard sizes
    # so the first dispatch doesn't block on Jinja rendering.
    #
    # B2: each cell type now compiles via a distinct generic entry point
    # (``computeMain<LstmCellImpl>`` etc., see `_rnn_cell_entry`) rather
    # than a distinct Slang source.  `prewarm_compile` is entry-blind (it
    # always assumes plain ``computeMain``), so using it here would either
    # silently fail to find the generic entry point or — worse — cache a
    # wrong-entry result under the same cache_key string the real dispatch
    # looks up by exact match.  Use the entry-aware
    # `batch_compile_slang_to_spirv` instead.
    from ...runtime import _slangc_available
    from ...runtime.slangc import batch_compile_slang_to_spirv

    if not _slangc_available():
        return

    rnn_specs: list[tuple[str, str, str, tuple[str, ...]]] = []
    for cell_type in _RNN_CELL_TYPES:
        entry = _rnn_cell_entry(cell_type)
        for dt in ("float",):
            for hidden_size in (128, 256, 512):
                for input_size in (128, 256, 512):
                    cache_key = (
                        f"slang_rnn_{cell_type}_h{hidden_size}_i{input_size}_{dt}"
                    )
                    src = _render_rnn_cell(
                        cell_type=cell_type,
                        hidden_size=hidden_size,
                        input_size=input_size,
                        dtype=dt,
                    )
                    rnn_specs.append((src, entry, cache_key, ()))

    # T.10-fast: prewarm fused RNN templates for all cell types at common sizes.
    for dt in ("float",):
        for hidden_size in (128, 256, 512):
            for input_size in (128, 256, 512):
                for cell_type in _RNN_CELL_TYPES:
                    entry = _rnn_cell_entry(cell_type)
                    cache_key = (
                        f"slang_rnn_fused_{cell_type}_h{hidden_size}_i{input_size}_{dt}"
                    )
                    src = _render_rnn_cell_fused(
                        cell_type=cell_type,
                        hidden_size=hidden_size,
                        input_size=input_size,
                        seq_len=64,
                        dtype=dt,
                    )
                    rnn_specs.append((src, entry, cache_key, ()))

    import threading

    def _bg_prewarm() -> None:
        try:
            batch_compile_slang_to_spirv(rnn_specs)
        except Exception:
            pass  # best-effort: individual failures are non-fatal

    threading.Thread(target=_bg_prewarm, daemon=True).start()
