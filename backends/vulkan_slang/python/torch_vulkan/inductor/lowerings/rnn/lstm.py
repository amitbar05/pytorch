"""T.10 / CP.3 — LSTM cell dispatch via eager intercepts.

Registers ``aten.lstm.input`` and ``aten.lstm.data`` on the
``AutogradPrivateUse1`` dispatch key.
"""

from __future__ import annotations

from ..rnn_template import run_rnn_via_template as _run_rnn_via_template
from .common import _RNN_CUSTOM_OPS, _use_cpu_fallback


def register_lstm_intercepts(rnn_lib) -> None:
    """Register LSTM ``AutogradPrivateUse1`` intercepts on *rnn_lib*."""

    lstm_op = _RNN_CUSTOM_OPS["lstm"]

    def _lstm_input_impl(
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    ):
        if not _use_cpu_fallback():
            return _run_rnn_via_template(
                "lstm",
                input,
                hx,
                params,
                has_biases,
                num_layers,
                bidirectional,
                batch_first,
                dropout=dropout,
                train=train,
            )
        tensors = [input, hx[0], hx[1], *params]
        out = lstm_op(
            tensors,
            len(params),
            False,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )
        return tuple(out)

    def _lstm_data_impl(
        data,
        batch_sizes,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
    ):
        tensors = [data, batch_sizes, hx[0], hx[1], *params]
        out = lstm_op(
            tensors,
            len(params),
            True,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            False,
        )
        return tuple(out)

    rnn_lib.impl("lstm.input", _lstm_input_impl, "AutogradPrivateUse1")
    rnn_lib.impl("lstm.data", _lstm_data_impl, "AutogradPrivateUse1")
