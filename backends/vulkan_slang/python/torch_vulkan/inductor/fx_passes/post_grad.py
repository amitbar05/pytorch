"""Post-grad FX passes — relu-clamp_min, prewarm, debug dump."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_DUMP_FX_DIR: str = ""
_DUMP_FX_COUNTER: int = 0


_VULKAN_OP_PREWARM_REGISTRY: dict[str, tuple[str, str]] = {
    # aten target full-name → (cache_key, slang_src)
    # Pointwise activations.
    "aten.relu.default": (
        "fx_prewarm_relu_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = max(in_x[tid.x], 0.0f);\n"
        "}\n",
    ),
    "aten.sigmoid.default": (
        "fx_prewarm_sigmoid_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = 1.0f / (1.0f + exp(-in_x[tid.x]));\n"
        "}\n",
    ),
    "aten.tanh.default": (
        "fx_prewarm_tanh_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = tanh(in_x[tid.x]);\n"
        "}\n",
    ),
    # Pointwise binary.
    "aten.add.Tensor": (
        "fx_prewarm_add_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_a,\n"
        "                 uniform StructuredBuffer<float> in_b,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = in_a[tid.x] + in_b[tid.x];\n"
        "}\n",
    ),
    "aten.mul.Tensor": (
        "fx_prewarm_mul_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_a,\n"
        "                 uniform StructuredBuffer<float> in_b,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = in_a[tid.x] * in_b[tid.x];\n"
        "}\n",
    ),
    "aten.sub.Tensor": (
        "fx_prewarm_sub_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_a,\n"
        "                 uniform StructuredBuffer<float> in_b,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = in_a[tid.x] - in_b[tid.x];\n"
        "}\n",
    ),
    "aten.div.Tensor": (
        "fx_prewarm_div_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_a,\n"
        "                 uniform StructuredBuffer<float> in_b,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = in_a[tid.x] / in_b[tid.x];\n"
        "}\n",
    ),
    "aten.gelu.default": (
        "fx_prewarm_gelu_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    float x = in_x[tid.x];\n"
        "    float k = 0.7978845608028654f;\n"
        "    float c = 0.044715f * x * x * x;\n"
        "    out_y[tid.x] = 0.5f * x * (1.0f + tanh(k * (x + c)));\n"
        "}\n",
    ),
    "aten.silu.default": (
        "fx_prewarm_silu_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    float x = in_x[tid.x];\n"
        "    out_y[tid.x] = x / (1.0f + exp(-x));\n"
        "}\n",
    ),
    "aten.neg.default": (
        "fx_prewarm_neg_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = -in_x[tid.x];\n"
        "}\n",
    ),
    "aten.abs.default": (
        "fx_prewarm_abs_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = abs(in_x[tid.x]);\n"
        "}\n",
    ),
    "aten.exp.default": (
        "fx_prewarm_exp_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = exp(in_x[tid.x]);\n"
        "}\n",
    ),
    "aten.log.default": (
        "fx_prewarm_log_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = log(in_x[tid.x]);\n"
        "}\n",
    ),
    "aten.sqrt.default": (
        "fx_prewarm_sqrt_f32",
        '[shader("compute")][numthreads(256, 1, 1)]\n'
        "void computeMain(uint3 tid : SV_DispatchThreadID,\n"
        "                 uniform StructuredBuffer<float> in_x,\n"
        "                 uniform RWStructuredBuffer<float> out_y,\n"
        "                 uniform uint numel) {\n"
        "    if (tid.x >= numel) return;\n"
        "    out_y[tid.x] = sqrt(in_x[tid.x]);\n"
        "}\n",
    ),
}


def _aten_target_name(target) -> str:
    """Canonical full-name of an aten OpOverload (e.g. ``aten.relu.default``)
    for the prewarm registry lookup. Returns empty string for non-aten
    targets so callers can skip them cheaply.
    """
    try:
        ns = target.namespace  # type: ignore[attr-defined]
        name = target._opname  # type: ignore[attr-defined]
        overload = target._overloadname  # type: ignore[attr-defined]
        return f"{ns}.{name}.{overload}"
    except AttributeError:
        return ""


def prewarm_from_fx_graph(gm: "torch.fx.GraphModule") -> int:
    """Walk the FX graph, find Vulkan-bound aten ops, submit canonical
    kernel sources to ``runtime.prewarm_compile`` so slangc starts
    compiling before the wrapper codegen requests them.

    Best-effort: any unrecognized op is silently skipped, and the
    function tolerates a missing ``torch_vulkan.inductor.runtime`` (e.g.
    when the FX pass is invoked from a non-Vulkan context).

    Returns the count of ``(cache_key, src)`` pairs actually scheduled
    by ``prewarm_compile`` (already-cached entries don't count).
    """
    try:
        from torch_vulkan.inductor.runtime import prewarm_compile
    except ImportError:
        return 0

    # Deduplicate registry hits — many graphs use the same op multiple
    # times but we only need to warm the cache once per source.
    pending: dict[str, str] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        # Vulkan-only: skip nodes whose `meta['val']` indicates a non-
        # Vulkan device. If `val` is missing (early FX pass), assume
        # Vulkan since this pass only runs in the Vulkan custom-pass
        # composite.
        val = node.meta.get("val") if hasattr(node, "meta") else None
        if val is not None and hasattr(val, "device"):
            try:
                if val.device.type != "vulkan":
                    continue
            except (AttributeError, TypeError):
                pass
        full_name = _aten_target_name(node.target)
        if not full_name or full_name not in _VULKAN_OP_PREWARM_REGISTRY:
            continue
        key, src = _VULKAN_OP_PREWARM_REGISTRY[full_name]
        pending[key] = src

    if not pending:
        return 0
    return prewarm_compile(list(pending.items()), sync=False)


def _maybe_dump_fx(gm: "torch.fx.GraphModule", phase: str) -> None:
    """When ``TORCH_VULKAN_DUMP_FX=<dir>`` is set, write the FX graph to disk.

    Phase is ``"pre"`` (before our passes run) or ``"post"`` (after). Files
    are numbered by global counter so multiple compiled functions don't
    overwrite each other.
    """
    import os

    global _DUMP_FX_DIR, _DUMP_FX_COUNTER
    dump_dir = os.environ.get("TORCH_VULKAN_DUMP_FX", "")
    if not dump_dir:
        return
    if _DUMP_FX_DIR != dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        _DUMP_FX_DIR = dump_dir
    if phase == "pre":
        _DUMP_FX_COUNTER += 1
    idx = _DUMP_FX_COUNTER
    path = os.path.join(dump_dir, f"graph_{idx:04d}_{phase}.txt")
    try:
        with open(path, "w") as f:
            f.write(gm.print_readable(False))
    except OSError:
        pass


def _replace_relu_with_clamp_min(gm: "torch.fx.GraphModule") -> None:
    """Replace ``relu`` with pointwise primitives in any call form.

    The built-in ``ReluBackward0`` fails under torch.compile with Vulkan
    tensors because it saves the forward output and calls
    ``aten.threshold_backward`` against a meta-cascaded saved tensor
    during AOTAutograd's joint trace, which then collapses to a ``[]``-
    shape gradient.

    Decompose relu(x) → where(x > 0, x, 0) using only pointwise
    primitives whose backwards also decompose into pointwise ops — no
    threshold-backward dispatch, no saved-output shenanigans.

    Handles every form Dynamo emits before AOT lowering:
    - ``call_method`` ``relu`` (``x.relu()``)
    - ``call_function torch.relu`` (``torch.relu(x)``)
    - ``call_function torch.nn.functional.relu`` (``F.relu(x)``)
    - ``call_function aten.relu.default`` (post-AOT form)
    - ``call_module`` of an ``nn.ReLU`` instance
    """
    import torch

    aten = torch.ops.aten

    def _is_relu_target(node) -> bool:
        if node.op == "call_method" and node.target in ("relu", "relu_"):
            return True
        if node.op == "call_function":
            t = node.target
            if t is torch.relu or t is torch.relu_:
                return True
            if t is torch.nn.functional.relu:
                return True
            if t is aten.relu.default or t is aten.relu_.default:
                return True
        if node.op == "call_module":
            sub = (
                gm.get_submodule(node.target) if hasattr(gm, "get_submodule") else None
            )
            if isinstance(sub, torch.nn.ReLU):
                return True
        return False

    replaced = 0
    for node in list(gm.graph.nodes):
        if not _is_relu_target(node):
            continue
        # call_method relu: args = (self,)
        # call_function aten.relu.default: args = (self,)
        # call_module ReLU: args = (self,)
        inp = node.args[0]
        with gm.graph.inserting_before(node):
            # relu(x) = where(x > 0, x, full_like(x, 0))
            zero = gm.graph.call_function(
                aten.full_like.default,
                (inp, 0.0),
            )
            cond = gm.graph.call_function(
                aten.gt.Scalar,
                (inp, 0),
            )
            # NOTE: GPU.1 — on RDNA1 hardware, compiled backward for
            # gt+where produces inconsistent results (sometimes zero gradients).
            # Adding aten.to.dtype(cond, float32) helps when graph breaks allow
            # eager fallback but doesn't fix the fullgraph path.  Root cause
            # appears to be a Vulkan buffer synchronization or driver issue.
            new_node = gm.graph.call_function(
                aten.where.self,
                (cond, inp, zero),
            )
        new_node.meta = node.meta.copy() if hasattr(node, "meta") else {}
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
        replaced += 1
    if replaced:
        gm.graph.lint()
        gm.recompile()
    # TR.15: rewrite SDPA call_function nodes to our opaque custom_op so
    # AOTAutograd's fake-trace doesn't dispatch into ``_C._nn.scaled_dot_product_attention``
    # on FakeTensor inputs (which calls ``data_ptr()`` and crashes when
    # SDPA is preceded by reshape/transpose chains that produce non-contiguous
    # FakeTensors). Co-located in the same pre-grad pass entry-point as the
    # relu rewrite so a single Group H pass handles all pre-AOT Vulkan fixups.
    _replace_sdpa_with_custom_op(gm)
    return gm


def _replace_sdpa_with_custom_op(gm: "torch.fx.GraphModule") -> None:
    """TR.15 — Decompose ``F.scaled_dot_product_attention`` /
    ``aten.scaled_dot_product_attention.default`` in the pre-grad FX graph
    into pure aten primitives (matmul + softmax + matmul + scaling).

    Why: Dynamo captures ``F.scaled_dot_product_attention(q, k, v, ...)`` as
    a call_function node whose target is ``torch._C._nn.scaled_dot_product_attention``.
    AOTAutograd's ``run_functionalized_fw_and_collect_metadata`` re-runs the
    captured graph against FakeTensors via ``fx.Interpreter``. The C function
    has no ``register_fake`` shim, so when the SDPA inputs come from a
    ``reshape().transpose()`` chain (non-contiguous FakeTensors), the C
    fast-path tries to read ``data_ptr()`` and trips ``RuntimeError: Cannot
    access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor)``.

    A custom_op replacement (e.g. ``flash_attention_fused`` or
    ``sdpa_with_optional_mask``) would let fake-trace succeed but introduces a
    second blocker — joint-trace runs the registered autograd backward
    concretely on Vulkan tensors and trips an engine stream assertion. The
    decomposition path sidesteps both: AOT joint-traces a graph of aten
    primitives whose backwards are all FakeTensor-safe, and the existing
    post-grad SDPA / scaled_bmm patterns can re-fuse the chain when the
    envelope qualifies (head_dim ∈ {32, 64, 128} for flash, etc.).

    Decomposition (no mask / no dropout):
        scores  = (q @ k.transpose(-2, -1)) * scale
        if is_causal: scores = scores + triu(full(..., -inf), 1)
        attn    = softmax(scores, dim=-1)
        output  = attn @ v
    Default scale = 1 / sqrt(head_dim) per ``F.sdpa`` semantics.

    Skipped: attn_mask provided (let upstream handle), dropout_p>0, GQA.

    Vulkan-gating is handled by the caller (the pre-grad hook only invokes
    this rewrite when the graph's example inputs are on Vulkan).
    """
    import torch

    aten = torch.ops.aten

    sdpa_targets: set = {torch._C._nn.scaled_dot_product_attention}
    sdpa_aten = getattr(aten, "scaled_dot_product_attention", None)
    if sdpa_aten is not None and hasattr(sdpa_aten, "default"):
        sdpa_targets.add(sdpa_aten.default)

    def _node_val(node):
        """Return the FakeTensor stored on a pre-grad FX node.

        Dynamo's pre-grad graph stores it under ``example_value``; AOT-decomposed
        graphs use ``val``. Check both.
        """
        if not hasattr(node, "meta"):
            return None
        return node.meta.get("example_value", node.meta.get("val"))

    def _q_head_dim(node) -> int | None:
        """Best-effort head-dim recovery from node meta or input-graph traversal."""
        val = _node_val(node)
        if val is None:
            return None
        try:
            return int(val.shape[-1])
        except Exception:  # noqa: BLE001
            return None

    replaced = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in sdpa_targets:
            continue
        if len(node.args) < 3:
            continue
        q, k, v = node.args[:3]
        kwargs = dict(node.kwargs or {})
        # Positional args after (q, k, v) follow the F.sdpa signature:
        # (attn_mask, dropout_p, is_causal, scale, enable_gqa).
        positional = list(node.args[3:])

        def _pop(idx, key, default):
            if len(positional) > idx:
                return positional[idx]
            return kwargs.get(key, default)

        attn_mask = _pop(0, "attn_mask", None)
        dropout_p = _pop(1, "dropout_p", 0.0)
        is_causal = _pop(2, "is_causal", False)
        scale = _pop(3, "scale", None)
        enable_gqa = _pop(4, "enable_gqa", False)

        try:
            if bool(enable_gqa):
                continue
            if attn_mask is not None:
                continue
            if float(dropout_p) > 0.0:
                continue
        except Exception:  # noqa: BLE001
            continue

        # Resolve scale. If meta is missing, fall back to the runtime form
        # via ``aten.size + reciprocal_sqrt`` — but Dynamo's pre-grad graph
        # usually has shape info; if not we bail to keep the rewrite safe.
        head_dim = _q_head_dim(q)
        if scale is None and head_dim is None:
            # Cannot infer scale safely; leave the node alone (the original
            # data_ptr error will still surface, but we don't introduce a
            # subtly-wrong scale).
            continue
        if scale is None:
            scale_val = 1.0 / (head_dim**0.5)
        else:
            try:
                scale_val = float(scale)
            except (TypeError, ValueError):
                continue

        with gm.graph.inserting_before(node):
            # k_t = k.transpose(-2, -1)
            k_t = gm.graph.call_function(aten.transpose.int, args=(k, -2, -1))
            # scores = q @ k_t
            scores = gm.graph.call_function(aten.matmul, args=(q, k_t))
            # scores = scores * scale
            scores = gm.graph.call_function(
                aten.mul.Tensor, args=(scores, scale_val)
            )
            if bool(is_causal):
                # Pull seq_len from the captured FakeTensor (pre-grad graphs
                # store it under ``example_value``).
                q_val = _node_val(q)
                if q_val is not None and q_val.dim() >= 2:
                    seq_len = int(q_val.shape[-2])
                    val = q_val
                    full_node = gm.graph.call_function(
                        aten.full.default,
                        args=([seq_len, seq_len], float("-inf")),
                        kwargs={
                            "dtype": val.dtype,
                            "device": val.device,
                            "pin_memory": False,
                        },
                    )
                    triu_node = gm.graph.call_function(
                        aten.triu.default, args=(full_node, 1)
                    )
                    scores = gm.graph.call_function(
                        aten.add.Tensor, args=(scores, triu_node)
                    )
                else:
                    # No shape info — leave the node alone rather than
                    # generate an incorrect causal mask.
                    continue
            attn = gm.graph.call_function(aten._softmax.default, args=(scores, -1, False))
            output = gm.graph.call_function(aten.matmul, args=(attn, v))
            output.meta = node.meta.copy() if hasattr(node, "meta") else {}

        node.replace_all_uses_with(output)
        gm.graph.erase_node(node)
        replaced += 1

    if replaced:
        gm.graph.lint()
        gm.recompile()
