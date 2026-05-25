"""T.10 / CP.3 — GRU / RNN-tanh / RNN-relu cell dispatch via eager intercepts.

Registers ``aten.gru.input``, ``aten.gru.data``, ``aten.rnn_tanh.input``,
``aten.rnn_tanh.data``, ``aten.rnn_relu.input``, and ``aten.rnn_relu.data``
on the ``AutogradPrivateUse1`` dispatch key.
"""

from __future__ import annotations

from .common import _RNN_CUSTOM_OPS, _use_cpu_fallback
from .rnn_autograd import apply_vulkan_rnn


def register_gru_intercepts(rnn_lib) -> None:
    """Register GRU / RNN-tanh / RNN-relu ``AutogradPrivateUse1`` intercepts on *rnn_lib*."""

    gru_op = _RNN_CUSTOM_OPS["gru"]
    rnn_tanh_op = _RNN_CUSTOM_OPS["rnn_tanh"]
    rnn_relu_op = _RNN_CUSTOM_OPS["rnn_relu"]

    def _gru_input_impl(
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
            out = apply_vulkan_rnn(
                "gru",
                False,
                input,
                hx,
                list(params),
                has_biases,
                num_layers,
                dropout,
                train,
                bidirectional,
                batch_first,
            )
            # out = (output, h_n)
            return out[0], out[1]
        tensors = [input, hx, *params]
        out = gru_op(
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

    def _gru_data_impl(
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
        tensors = [data, batch_sizes, hx, *params]
        out = gru_op(
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

    def _rnn_tanh_input_impl(
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
            out = apply_vulkan_rnn(
                "rnn_tanh",
                False,
                input,
                hx,
                list(params),
                has_biases,
                num_layers,
                dropout,
                train,
                bidirectional,
                batch_first,
            )
            return out[0], out[1]
        tensors = [input, hx, *params]
        out = rnn_tanh_op(
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

    def _rnn_relu_input_impl(
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
            out = apply_vulkan_rnn(
                "rnn_relu",
                False,
                input,
                hx,
                list(params),
                has_biases,
                num_layers,
                dropout,
                train,
                bidirectional,
                batch_first,
            )
            return out[0], out[1]
        tensors = [input, hx, *params]
        out = rnn_relu_op(
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

    def _rnn_tanh_data_impl(
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
        tensors = [data, batch_sizes, hx, *params]
        out = rnn_tanh_op(
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

    def _rnn_relu_data_impl(
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
        tensors = [data, batch_sizes, hx, *params]
        out = rnn_relu_op(
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

    rnn_lib.impl("gru.input", _gru_input_impl, "AutogradPrivateUse1")
    rnn_lib.impl("gru.data", _gru_data_impl, "AutogradPrivateUse1")
    rnn_lib.impl("rnn_tanh.input", _rnn_tanh_input_impl, "AutogradPrivateUse1")
    rnn_lib.impl("rnn_tanh.data", _rnn_tanh_data_impl, "AutogradPrivateUse1")
    rnn_lib.impl("rnn_relu.input", _rnn_relu_input_impl, "AutogradPrivateUse1")
    rnn_lib.impl("rnn_relu.data", _rnn_relu_data_impl, "AutogradPrivateUse1")
