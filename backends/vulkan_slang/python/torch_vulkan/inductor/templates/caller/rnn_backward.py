"""RNN cell backward template callers (split from rnn.py, M22d, anti-goal #7)."""

from __future__ import annotations

import struct

import torch

from ...buffer_pool import pool_acquire
from ...vulkan_template import _load_slang_template
from ...vulkan_template_caller import _dtype_to_slang
from .rnn import _RNN_CELL_TYPES

# OP.25 — RNN cell backward template rendering and dispatch
# ═══════════════════════════════════════════════════════════════════════

_rnn_cell_bwd_cache: dict[tuple, str] = {}


def _render_rnn_cell_bwd(
    cell_type: str,
    hidden_size: int,
    input_size: int,
    dtype: str = "float",
) -> str:
    """Render the rnn_cell_bwd Jinja2 template.

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

    key = (cell_type, hidden_size, input_size, dtype)
    if key in _rnn_cell_bwd_cache:
        return _rnn_cell_bwd_cache[key]

    src = _load_slang_template("rnn_cell_bwd")
    if not src:
        raise RuntimeError("rnn_cell_bwd.py.jinja template not found")

    env = Environment()
    tmpl = env.from_string(src)
    rendered = tmpl.render(
        cell_type=cell_type,
        hidden_size=hidden_size,
        input_size=input_size,
        dtype=dtype,
    )
    _rnn_cell_bwd_cache[key] = rendered
    return rendered


def _dispatch_rnn_cell_bwd(
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
    grad_h: torch.Tensor,
    grad_c: torch.Tensor | None,
    grad_x: torch.Tensor,
    grad_h_prev: torch.Tensor,
    grad_c_prev: torch.Tensor | None,
    grad_w_ih: torch.Tensor,
    grad_w_hh: torch.Tensor,
    grad_b_ih: torch.Tensor | None,
    grad_b_hh: torch.Tensor | None,
    src: str | None = None,
    cache_key: str | None = None,
) -> None:
    """Dispatch a single RNN cell backward computation.

    Computes ALL gradients for one time step in one kernel dispatch:
    grad_x, grad_h_prev, grad_c_prev (LSTM), grad_w_ih, grad_w_hh,
    grad_b_ih, grad_b_hh.

    All gradient tensors are accumulated in-place (read-modify-write),
    so callers must zero them before the first time step.

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
        grad_h: Gradient of loss w.r.t. h_t [batch, hidden_size].
        grad_c: Gradient of loss w.r.t. c_t [batch, hidden_size] (LSTM only).
        grad_x: Accumulated input gradient [batch, input_size].
        grad_h_prev: Accumulated hidden state gradient [batch, hidden_size].
        grad_c_prev: Accumulated cell state gradient [batch, hidden_size] (LSTM).
        grad_w_ih: Accumulated W_ih gradient.
        grad_w_hh: Accumulated W_hh gradient.
        grad_b_ih: Accumulated b_ih gradient (or None).
        grad_b_hh: Accumulated b_hh gradient (or None).
        src: Pre-rendered Slang source (rendered if None).
        cache_key: SPIR-V cache key (computed if None).
    """
    from ...runtime import compile_and_dispatch

    is_lstm = cell_type == "lstm"

    if src is None or cache_key is None:
        src = _render_rnn_cell_bwd(
            cell_type=cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            dtype=dtype,
        )
        cache_key = f"slang_rnn_bwd_{cell_type}_h{hidden_size}_i{input_size}_{dtype}"

    stride_w_ih = input_size
    stride_w_hh = hidden_size
    stride_x = input_size
    stride_h = hidden_size
    stride_c = hidden_size

    if is_lstm:
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
    else:
        pc = struct.pack(
            "7I",
            hidden_size,
            input_size,
            stride_w_ih,
            stride_w_hh,
            stride_x,
            stride_h,
            1 if has_bias else 0,
        )

    # Ensure all inputs are contiguous.
    for t_name in (
        "x_t",
        "h_prev",
        "w_ih",
        "w_hh",
        "grad_h",
        "grad_x",
        "grad_h_prev",
        "grad_w_ih",
        "grad_w_hh",
    ):
        t = locals()[t_name]
        if not t.is_contiguous():
            locals()[t_name] = t.contiguous()
    for t_name in ("c_prev", "grad_c", "grad_c_prev"):
        t = locals().get(t_name)
        if t is not None and not t.is_contiguous():
            locals()[t_name] = t.contiguous()

    # Build buffer list matching KernelArgs field order.
    # [x_t, h_prev, [c_prev], w_ih, w_hh, b_ih, b_hh, grad_h, [grad_c],
    #  grad_x, grad_h_prev, [grad_c_prev], grad_w_ih, grad_w_hh,
    #  grad_b_ih, grad_b_hh]
    buffers: list[torch.Tensor] = [x_t, h_prev]
    if is_lstm:
        buffers.append(
            c_prev if c_prev is not None else torch.empty(0, device=x_t.device)
        )
    buffers.extend([w_ih, w_hh])
    if has_bias and b_ih is not None:
        buffers.append(b_ih.contiguous() if not b_ih.is_contiguous() else b_ih)
    else:
        buffers.append(torch.empty(0, device=x_t.device))
    if has_bias and b_hh is not None:
        buffers.append(b_hh.contiguous() if not b_hh.is_contiguous() else b_hh)
    else:
        buffers.append(torch.empty(0, device=x_t.device))
    buffers.append(grad_h)
    if is_lstm:
        buffers.append(
            grad_c if grad_c is not None else torch.empty(0, device=x_t.device)
        )
    buffers.append(grad_x)
    buffers.append(grad_h_prev)
    if is_lstm:
        buffers.append(
            grad_c_prev
            if grad_c_prev is not None
            else torch.empty(0, device=x_t.device)
        )
    buffers.append(grad_w_ih)
    buffers.append(grad_w_hh)
    if has_bias and grad_b_ih is not None:
        buffers.append(
            grad_b_ih.contiguous() if not grad_b_ih.is_contiguous() else grad_b_ih
        )
    else:
        buffers.append(torch.empty(0, device=x_t.device))
    if has_bias and grad_b_hh is not None:
        buffers.append(
            grad_b_hh.contiguous() if not grad_b_hh.is_contiguous() else grad_b_hh
        )
    else:
        buffers.append(torch.empty(0, device=x_t.device))

    # All outputs are RW (accumulated in-place), but we still need to tell
    # compile_and_dispatch how many are outputs.
    # Outputs: grad_x, grad_h_prev, [grad_c_prev], grad_w_ih, grad_w_hh,
    #          grad_b_ih, grad_b_hh
    num_outputs = 5 if has_bias else 3
    if is_lstm:
        num_outputs += 1  # grad_c_prev
    if has_bias:
        num_outputs += 0  # already counted

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
        cache_key=cache_key,
    )


class _SlangTileRNNBackward:
    """Picklable callable for RNN cell backward template dispatch.

    Each instance is configured for a specific cell_type.  The callable
    interface accepts saved forward tensors and output gradients for one
    time step and accumulates all parameter gradients in-place.

    Caches the rendered Slang source per (dtype, hidden_size, input_size) tuple.
    """

    __slots__ = ("cell_type", "__name__", "_per_spec")

    def __init__(self, cell_type: str):
        if cell_type not in _RNN_CELL_TYPES:
            raise ValueError(
                f"Unknown RNN cell_type '{cell_type}'. "
                f"Must be one of: {_RNN_CELL_TYPES}"
            )
        self.cell_type = cell_type
        self.__name__ = f"slang_rnn_bwd_{cell_type}"
        self._per_spec: dict[tuple, tuple[str, str]] = {}

    def _src_and_key(
        self, hidden_size: int, input_size: int, dtype: str
    ) -> tuple[str, str]:
        spec_key = (hidden_size, input_size, dtype)
        cached = self._per_spec.get(spec_key)
        if cached is not None:
            return cached

        src = _render_rnn_cell_bwd(
            cell_type=self.cell_type,
            hidden_size=hidden_size,
            input_size=input_size,
            dtype=dtype,
        )
        cache_key = (
            f"slang_rnn_bwd_{self.cell_type}_h{hidden_size}_i{input_size}_{dtype}"
        )
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
        grad_h: torch.Tensor,
        grad_c: torch.Tensor | None,
        grad_x: torch.Tensor,
        grad_h_prev: torch.Tensor,
        grad_c_prev: torch.Tensor | None,
        grad_w_ih: torch.Tensor,
        grad_w_hh: torch.Tensor,
        grad_b_ih: torch.Tensor | None,
        grad_b_hh: torch.Tensor | None,
    ) -> None:
        """Dispatch one RNN cell backward step.

        All gradient tensors are accumulated in-place.
        """
        batch_size = x_t.shape[0]
        hidden_size = h_prev.shape[-1]
        input_size = x_t.shape[-1]
        dtype_s = _dtype_to_slang(x_t.dtype)
        has_bias = b_ih is not None and b_hh is not None

        src, cache_key = self._src_and_key(hidden_size, input_size, dtype_s)

        _dispatch_rnn_cell_bwd(
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
            grad_h=grad_h,
            grad_c=grad_c,
            grad_x=grad_x,
            grad_h_prev=grad_h_prev,
            grad_c_prev=grad_c_prev,
            grad_w_ih=grad_w_ih,
            grad_w_hh=grad_w_hh,
            grad_b_ih=grad_b_ih,
            grad_b_hh=grad_b_hh,
            src=src,
            cache_key=cache_key,
        )

    def __reduce__(self):
        return (_SlangTileRNNBackward, (self.cell_type,))
