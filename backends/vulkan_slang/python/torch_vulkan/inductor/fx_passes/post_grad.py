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
    # TR.15: SDPA nodes are NO LONGER decomposed here — OP.26 registers a
    # native lowering for aten.scaled_dot_product_attention that routes
    # directly to the FlashAttention template.  The meta_table registration
    # (_register_sdpa_meta) handles FakeTensorMode dispatch so AOT's
    # metadata collection succeeds without the data_ptr crash.
    return gm


# M22.4 (2026-05-18): `_replace_sdpa_with_custom_op` deleted.
#
# This was a 160-line pre-grad FX pass that decomposed
# ``F.scaled_dot_product_attention`` / ``torch._C._nn.scaled_dot_product_attention``
# into a chain of pure aten primitives (matmul + softmax + matmul + scaling +
# optional causal triu mask). It was originally written for TR.15 to dodge a
# ``data_ptr()`` crash during AOTAutograd's metadata collection on FakeTensors
# that came from non-contiguous ``reshape().transpose()`` chains.
#
# OP.26 obsoleted it: a native ``aten.scaled_dot_product_attention`` lowering
# in ``lowerings/attention.py`` now routes directly to the FlashAttention
# Slang template, and a companion ``_register_sdpa_meta`` shim provides the
# missing FakeTensor dispatch. The decomposition pass was kept around for a
# while but was never wired into any post-grad pass list (TR.15 closeout
# explicitly notes "SDPA nodes are NO LONGER decomposed here").
#
# The TR.15 invariant — SDPA compiles end-to-end through the attention block
# pattern — is locked by
# ``TestAttentionBlockReshapeTransposeCompile.test_attention_block_compile_tr_15``
# in the regression suite, which exercises the native OP.26 path. The
# structural floor that previously called this function directly
# (``test_tr15_pre_grad_sdpa_rewrite_lands``) is removed alongside it.


# ═══════════════════════════════════════════════════════════════════════════
# M18.8.b — enhanced conv → GN → ReLU fusion that matches Dynamo-emitted forms
# ═══════════════════════════════════════════════════════════════════════════
#
# The legacy fusion in ``meta_patches/decomposition_passes.py:_fuse_conv_gn_relu``
# matches ``aten.relu.default`` ← ``aten.native_group_norm.default`` ←
# ``aten.convolution.default`` (or ``torch_vulkan::conv2d_with_optional_bias``).
# Empirically, Dynamo emits a *different* shape for the monkey-patched
# ``nn.Sequential(Conv, GN, ReLU)`` pattern: the GN node is the closure
# ``_register_optional_tensor_workarounds.<locals>._patched_group_norm`` and
# the ReLU node is the raw ``torch.nn.functional.relu`` function reference
# (NOT ``aten.relu.default``).  The legacy matcher therefore never fires for
# the most common eager-Vulkan model topology.
#
# This pass walks the same pre-grad graph but recognises ALL three forms each
# op takes after Dynamo trace.  Installed by
# ``fx_passes/eager/__init__.py:register_eager_patch_custom_ops`` as an
# additional pre-grad pass — runs first (outermost) so the legacy pass and
# the relu→clamp_min rewrite see the rewritten chain (i.e. fewer nodes to
# process).


def _is_patched_group_norm_target(target) -> bool:
    """True if ``target`` is one of the GN forms Dynamo emits.

    Recognises:

    1. ``torch_vulkan.__init__._register_optional_tensor_workarounds.<locals>._patched_group_norm``
       — closure produced by the monkey-patch in ``python/torch_vulkan/__init__.py``.
       Identified by ``__qualname__`` since the closure object id is unique
       per-process and we can't import it without circular issues.
    2. ``torch.nn.functional.group_norm`` — the un-patched form (some
       trace paths see this when ``_is_vulkan`` returns False or the
       trace happens before the monkey-patch installs).
    3. ``torch.ops.aten.native_group_norm.default`` — the post-decomp form
       (for completeness; the legacy fusion already handles this).
    """
    import torch

    if target is torch.ops.aten.native_group_norm.default:
        return True
    if target is torch.nn.functional.group_norm:
        return True
    qn = getattr(target, "__qualname__", "")
    if "_patched_group_norm" in qn:
        return True
    return False


def _is_conv_with_bias_target(target) -> bool:
    """True if ``target`` is one of the conv forms Dynamo emits for the
    Vulkan-patched ``F.conv2d``."""
    import torch

    try:
        if target is torch.ops.torch_vulkan.conv2d_with_optional_bias.default:
            return True
    except AttributeError:
        pass
    if target is torch.ops.aten.convolution.default:
        return True
    return False


def _is_relu_target_for_fusion(target) -> bool:
    """True if ``target`` is one of the ReLU forms Dynamo emits."""
    import torch

    if target is torch.ops.aten.relu.default:
        return True
    if target is torch.ops.aten.relu_.default:
        return True
    if target is torch.nn.functional.relu:
        return True
    if target is torch.relu:
        return True
    return False


def _fuse_conv_patched_gn_relu(gm: "torch.fx.GraphModule") -> int:
    """M18.8.b enhanced fusion: replace ``conv → GN → ReLU`` chains with
    the fused ``torch_vulkan::conv2d_gn_relu_fused`` custom op.

    Matches the Dynamo-emitted forms (see ``_is_patched_group_norm_target``,
    ``_is_conv_with_bias_target``, ``_is_relu_target_for_fusion``).

    Returns the number of chains fused.
    """
    import operator

    import torch

    aten = torch.ops.aten
    graph = gm.graph
    fused_count = 0
    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if not _is_relu_target_for_fusion(node.target):
                continue
            relu_node = node

            # Trace back to the GN node (look through operator.getitem(gn, 0)).
            gn_node = None
            relu_in = relu_node.args[0] if len(relu_node.args) > 0 else None
            if isinstance(relu_in, torch.fx.Node) and relu_in.op == "call_function":
                if _is_patched_group_norm_target(relu_in.target):
                    gn_node = relu_in
                elif (
                    relu_in.target == operator.getitem
                    and len(relu_in.args) >= 1
                    and isinstance(relu_in.args[0], torch.fx.Node)
                    and relu_in.args[0].op == "call_function"
                    and _is_patched_group_norm_target(relu_in.args[0].target)
                ):
                    gn_node = relu_in.args[0]
            if gn_node is None:
                continue

            # Extract GN args.  Different call forms have different signatures:
            #   _patched_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5)
            #   F.group_norm(input, num_groups, weight=None, bias=None, eps=1e-5)
            #   aten.native_group_norm(input, weight, bias, N, C, HxW, group, eps)
            if gn_node.target is aten.native_group_norm.default:
                gn_args_pos = gn_node.args
                if len(gn_args_pos) < 8:
                    continue
                gn_input = gn_args_pos[0]
                gn_weight = gn_args_pos[1]
                gn_bias = gn_args_pos[2]
                num_groups = gn_args_pos[6]
                eps = gn_args_pos[7]
            else:
                # _patched_group_norm / F.group_norm form.
                gn_args_pos = gn_node.args
                gn_kwargs = gn_node.kwargs or {}
                if len(gn_args_pos) < 2:
                    continue
                gn_input = gn_args_pos[0]
                num_groups = gn_args_pos[1]
                gn_weight = (
                    gn_args_pos[2]
                    if len(gn_args_pos) > 2
                    else gn_kwargs.get("weight")
                )
                gn_bias = (
                    gn_args_pos[3]
                    if len(gn_args_pos) > 3
                    else gn_kwargs.get("bias")
                )
                eps = (
                    gn_args_pos[4]
                    if len(gn_args_pos) > 4
                    else gn_kwargs.get("eps", 1e-5)
                )

            if not isinstance(gn_input, torch.fx.Node):
                continue
            if gn_input.op != "call_function":
                continue

            # Conv must precede GN.
            conv_node = gn_input
            if not _is_conv_with_bias_target(conv_node.target):
                continue

            conv_args = conv_node.args
            if conv_node.target is aten.convolution.default:
                # aten.convolution(input, weight, bias, stride, padding,
                #                  dilation, transposed, output_padding, groups)
                if len(conv_args) < 9:
                    continue
                conv_input = conv_args[0]
                conv_weight = conv_args[1]
                conv_bias = conv_args[2]
                conv_stride = conv_args[3]
                conv_padding = conv_args[4]
                conv_dilation = conv_args[5]
                conv_groups = conv_args[8]
            else:
                # torch_vulkan::conv2d_with_optional_bias(input, weight,
                #                                        bias, stride, padding,
                #                                        dilation, groups)
                if len(conv_args) < 7:
                    continue
                conv_input = conv_args[0]
                conv_weight = conv_args[1]
                conv_bias = conv_args[2]
                conv_stride = conv_args[3]
                conv_padding = conv_args[4]
                conv_dilation = conv_args[5]
                conv_groups = conv_args[6]

            try:
                fused_op = torch.ops.torch_vulkan.conv2d_gn_relu_fused.default
            except AttributeError:
                continue

            if not isinstance(conv_stride, (list, tuple)):
                conv_stride = [conv_stride, conv_stride]
            if not isinstance(conv_padding, (list, tuple)):
                conv_padding = [conv_padding, conv_padding]
            if not isinstance(conv_dilation, (list, tuple)):
                conv_dilation = [conv_dilation, conv_dilation]

            with graph.inserting_before(relu_node):
                fused = graph.call_function(
                    fused_op,
                    args=(
                        conv_input,
                        conv_weight,
                        conv_bias,
                        list(conv_stride),
                        list(conv_padding),
                        list(conv_dilation),
                        int(conv_groups),
                        gn_weight,
                        gn_bias,
                        int(num_groups) if num_groups is not None else 1,
                        float(eps) if eps is not None else 1e-5,
                    ),
                )
                fused.meta = dict(relu_node.meta)

            relu_node.replace_all_uses_with(fused)
            graph.erase_node(relu_node)
            # gn_node and conv_node may still be referenced by other consumers
            # (e.g. debug inspection); only erase if no remaining users.
            if not gn_node.users:
                graph.erase_node(gn_node)
            if not conv_node.users:
                graph.erase_node(conv_node)
            graph.lint()
            gm.recompile()
            fused_count += 1
            changed = True
            break  # restart scan after graph mutation
    return fused_count


# Cumulative counter for the test suite to assert that the pass fires.
_FUSE_CONV_PATCHED_GN_RELU_COUNTER = [0]


def install_conv_patched_gn_relu_fusion() -> None:
    """Install the M18.8.b enhanced fusion as a pre-grad pass.

    Idempotent — guarded on a sentinel attribute on
    ``torch._inductor.compile_fx``.
    """
    import torch
    import torch._inductor.compile_fx as _cfx

    if getattr(_cfx, "_vulkan_conv_patched_gn_relu_fusion_patched", False):
        return

    _orig = _cfx.run_pre_grad_passes

    def _detect_vulkan(example_inputs_, model_) -> bool:
        inputs = example_inputs_ or ()
        for t in inputs:
            if not isinstance(t, torch.Tensor):
                continue
            try:
                if t.device.type in ("vulkan", "privateuseone"):
                    return True
            except Exception:
                pass
            try:
                fd = getattr(t, "fake_device", None)
                if fd is not None and fd.type in ("vulkan", "privateuseone"):
                    return True
            except Exception:
                pass
        if isinstance(model_, torch.fx.GraphModule):
            try:
                for node in model_.graph.nodes:
                    if node.op != "placeholder":
                        continue
                    val = node.meta.get("val") if hasattr(node, "meta") else None
                    if val is None:
                        continue
                    for v in val if isinstance(val, (list, tuple)) else [val]:
                        if not isinstance(v, torch.Tensor):
                            continue
                        try:
                            if v.device.type in ("vulkan", "privateuseone"):
                                return True
                        except Exception:
                            pass
                        try:
                            fd = getattr(v, "fake_device", None)
                            if fd is not None and fd.type in (
                                "vulkan",
                                "privateuseone",
                            ):
                                return True
                        except Exception:
                            pass
            except Exception:
                pass
        return False

    def _patched(model_, example_inputs_):
        if _detect_vulkan(example_inputs_, model_) and isinstance(
            model_, torch.fx.GraphModule
        ):
            try:
                fused = _fuse_conv_patched_gn_relu(model_)
                _FUSE_CONV_PATCHED_GN_RELU_COUNTER[0] += fused
            except Exception as e:  # pragma: no cover
                import logging

                logging.getLogger(__name__).warning(
                    "M18.8.b conv→GN→ReLU fusion failed: %s", e
                )
        return _orig(model_, example_inputs_)

    _cfx.run_pre_grad_passes = _patched
    _cfx._vulkan_conv_patched_gn_relu_fusion_patched = True
