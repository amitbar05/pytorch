# mypy: allow-untyped-defs
import functools

import torch
from torch._dynamo.utils import counters
from torch._ops import OpOverload, OpOverloadPacket
from torch.utils._ordered_set import OrderedSet

from ..pattern_matcher import fwd_only, register_replacement

aten = torch.ops.aten


@functools.cache
def _misc_patterns_init(input_device: torch.device | None = None):
    from .joint_graph import patterns as joint_graph_patterns
    from .post_grad import pass_patterns as post_grad_patterns_all

    post_grad_patterns = post_grad_patterns_all[1]  # medium priority

    if input_device:
        device = str(input_device)
    else:
        if torch.cuda.is_available():
            # workaround https://github.com/pytorch/pytorch/issues/97894
            device = "cuda"
        else:
            device = "cpu"

    # These patterns do 2 things
    # 1. Since we know that index is completely unique, we can codegen it using
    # stores instead of atomic adds, which is quite a bit faster.
    # 2. Also, since we are guaranteed that they are completely within bounds,
    # we can use unsafe indexing and skip debug asserts
    def randperm_index_add_pattern(x, y):
        index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return torch.index_add(x, dim=0, source=y, index=index), index

    def randperm_index_add_replacement(x, y):
        index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return (
            torch.ops.aten._unsafe_index_put(
                x, (index,), aten._unsafe_index(x, (index,)) + y, accumulate=False
            ),
            index,
        )

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_add_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_add_replacement,
        [torch.empty(4, 8, device=device), torch.empty(2, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        skip_duplicates=True,
    )

    def randperm_index_pattern(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten.index(x, (index,)), index

    def randperm_index_replacement(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten._unsafe_index(x, (index,)), index

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_replacement,
        [torch.empty(4, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        scalar_workaround={"slice_shape": 42},
        skip_duplicates=True,
    )

    # Pattern: e8m0 extraction with ceiling rounding (for MX format scaling)
    # Only register on SM100+ where the PTX instruction is available
    if device == "cuda" and torch.cuda.get_device_capability() >= (10, 0):
        from .. import inductor_prims

        # Pattern 1: Bit manipulation approach
        def e8m0_rceil_pattern(inp):
            inp_bits = inp.view(torch.int32)
            biased_exp = (inp_bits >> 23) & 0xFF
            mantissa = inp_bits & 0x7FFFFF
            needs_round_up = mantissa != 0
            e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
            e8m0_biased = torch.clamp(e8m0_biased, 0, 255)
            return e8m0_biased.to(torch.uint8)

        def e8m0_rceil_replacement(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        def e8m0_extra_check(match):
            inp = match.kwargs.get("inp")
            if inp is None:
                return False
            inp_val = inp.meta.get("val")
            return (
                inp_val is not None
                and inp_val.device.type == "cuda"
                and inp_val.dtype == torch.float32
            )

        register_replacement(
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_pattern,
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_replacement,
            [torch.randn(32, device="cuda", dtype=torch.float32)],
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            [post_grad_patterns],
            extra_check=e8m0_extra_check,
        )

        # Pattern 2: log2 + ceil approach (used by torchao MX formats)
        # Matches: (clamp(ceil(log2(x)), -127, 127) + 127).to(uint8)
        E8M0_BIAS = 127

        def e8m0_rceil_log2_pattern(inp):
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        def e8m0_rceil_log2_replacement(inp):
            # The PTX instruction expects the raw float value, not log2
            # So we need to convert: if inp is log2(x), then 2^inp is x
            # But actually our pattern matches on the value before log2
            return inductor_prims.cvt_e8m0_rceil(inp)

        register_replacement(
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_log2_pattern,
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_log2_replacement,
            [torch.randn(32, device="cuda", dtype=torch.float32).abs() + 1e-10],
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            [post_grad_patterns],
            extra_check=e8m0_extra_check,
        )

    # =========================================================================
    # RMSNorm decomposition pattern:
    # Matches: pow(x,2)->mean->add(eps)->rsqrt->mul(x)->mul(weight)
    # Replaces with: aten.rms_norm (fused kernel)
    # =========================================================================
    def _rms_norm_decomp_pattern(x, weight, eps):
        var = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
        rstd = torch.rsqrt(torch.add(var, eps))
        normed = torch.mul(x, rstd)
        return torch.mul(normed, weight)

    def _rms_norm_decomp_replacement(x, weight, eps):
        counters["inductor"]["fuse_rms_norm"] += 1
        return aten.rms_norm(x, [x.shape[-1]], weight, eps)

    def _rms_norm_extra_check(match):
        x = match.kwargs.get("x")
        if x is None:
            return False
        x_val = x.meta.get("val")
        if x_val is None:
            return False
        weight = match.kwargs.get("weight")
        if weight is not None:
            w_val = weight.meta.get("val")
            if w_val is not None and w_val.dtype != x_val.dtype:
                return False
            # weight should be 1D and match the last dim of x
            if w_val is not None and w_val.dim() != 1:
                return False
        # Only match for float dtypes and at least 2D tensors
        return (
            x_val.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and x_val.dim() >= 2
        )

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        _rms_norm_decomp_pattern,
        # pyrefly: ignore [bad-argument-type]
        _rms_norm_decomp_replacement,
        [
            torch.randn(4, 8, device=device),
            torch.randn(8, device=device),
            1e-5,
        ],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        extra_check=_rms_norm_extra_check,
        scalar_workaround={"eps": 1e-5},
        skip_duplicates=True,
    )

    # =========================================================================
    # RMSNorm + Linear (matmul) fusion pattern:
    # Matches: rms_norm(x, weight, eps) -> linear(normed, w_lin) or
    # decomposed rms_norm -> matmul/addmm
    # Replaces with: fused operation via rms_norm + addmm
    # =========================================================================
    def _rms_norm_linear_pattern(x, rms_weight, eps, w_lin, bias):
        var = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
        rstd = torch.rsqrt(torch.add(var, eps))
        normed = torch.mul(x, rstd)
        normed_weighted = torch.mul(normed, rms_weight)
        return torch.addmm(bias, normed_weighted, w_lin)

    def _rms_norm_linear_replacement(x, rms_weight, eps, w_lin, bias):
        counters["inductor"]["fuse_rms_norm_linear"] += 1
        normed = aten.rms_norm(x, [x.shape[-1]], rms_weight, eps)
        return torch.addmm(bias, normed, w_lin)

    def _rms_norm_linear_extra_check(match):
        x = match.kwargs.get("x")
        if x is None:
            return False
        x_val = x.meta.get("val")
        if x_val is None:
            return False
        w_lin = match.kwargs.get("w_lin")
        if w_lin is None:
            return False
        w_lin_val = w_lin.meta.get("val")
        if w_lin_val is None:
            return False
        # x shape: [..., D], w_lin shape: [D, D_out] or [D_out, D]
        if x_val.dim() < 2 or w_lin_val.dim() != 2:
            return False
        return x_val.dtype in (torch.float32, torch.float16, torch.bfloat16)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        _rms_norm_linear_pattern,
        # pyrefly: ignore [bad-argument-type]
        _rms_norm_linear_replacement,
        [
            torch.randn(4, 8, device=device),
            torch.randn(8, device=device),
            1e-5,
            torch.randn(8, 16, device=device),
            torch.randn(16, device=device),
        ],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        extra_check=_rms_norm_linear_extra_check,
        scalar_workaround={"eps": 1e-5},
        skip_duplicates=True,
    )

    # =========================================================================
    # SiLU + Mul fusion (SwiGLU component):
    # Matches: sigmoid(x) -> mul(x) -> mul(y)  which is silu(x) * y
    # This is a key component of SwiGLU FFN layers in LLaMA/Mistral/etc.
    # Replaces with: aten.silu(x) * y
    #
    # Also matches the inductor-specific decomposition:
    # neg(x)->exp->add(1)->div(x)->mul(y)  which is silu(x) * y
    # =========================================================================
    def _silu_mul_pattern_1(x, y):
        return torch.mul(torch.mul(x, torch.sigmoid(x)), y)

    def _silu_mul_replacement_1(x, y):
        counters["inductor"]["fuse_silu_mul"] += 1
        return aten.silu(x).mul(y)

    def _silu_mul_extra_check(match):
        x = match.kwargs.get("x")
        if x is None:
            return False
        x_val = x.meta.get("val")
        if x_val is None:
            return False
        y = match.kwargs.get("y")
        if y is None:
            return False
        y_val = y.meta.get("val")
        if y_val is None:
            return False
        # Both inputs should be float dtypes
        return x_val.dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ) and y_val.dtype in (torch.float32, torch.float16, torch.bfloat16)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        _silu_mul_pattern_1,
        # pyrefly: ignore [bad-argument-type]
        _silu_mul_replacement_1,
        [
            torch.randn(4, 8, device=device),
            torch.randn(4, 8, device=device),
        ],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        extra_check=_silu_mul_extra_check,
        skip_duplicates=True,
    )

    # =========================================================================
    # RoPE (Rotary Position Embedding) fusion pattern:
    # Matches the common RoPE computation:
    #   split(x, 2, dim=-1) -> [x1, x2]
    #   neg(x2) -> cat([neg_x2, x1], dim=-1)  (the rotated half)
    #   mul(x, cos) + mul(rotated, sin)
    #
    # Replaces with a fused RoPE operation.
    # Pattern: x * cos + cat([-x[..., half:], x[..., :half]], dim=-1) * sin
    # =========================================================================
    def _rope_pattern(x, cos, sin):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        neg_x2 = torch.neg(x2)
        rotated = torch.cat([neg_x2, x1], dim=-1)
        return torch.add(torch.mul(x, cos), torch.mul(rotated, sin))

    def _rope_replacement(x, cos, sin):
        counters["inductor"]["fuse_rope"] += 1
        # Use a fused RoPE implementation that avoids the split/cat overhead
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        # Compute directly: x * cos + [-x2, x1] * sin
        # This keeps the computation structured for further fusion
        rotated_x1 = x1 * cos[..., :half] + (-x2) * sin[..., :half]
        rotated_x2 = x2 * cos[..., half:] + x1 * sin[..., half:]
        return torch.cat([rotated_x1, rotated_x2], dim=-1)

    def _rope_extra_check(match):
        x = match.kwargs.get("x")
        if x is None:
            return False
        x_val = x.meta.get("val")
        if x_val is None:
            return False
        cos = match.kwargs.get("cos")
        sin = match.kwargs.get("sin")
        if cos is None or sin is None:
            return False
        cos_val = cos.meta.get("val")
        sin_val = sin.meta.get("val")
        if cos_val is None or sin_val is None:
            return False
        # x should be at least 2D with even last dimension
        if x_val.dim() < 2 or x_val.shape[-1] % 2 != 0:
            return False
        # cos and sin should be broadcastable with x
        return x_val.dtype in (torch.float32, torch.float16, torch.bfloat16)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        _rope_pattern,
        # pyrefly: ignore [bad-argument-type]
        _rope_replacement,
        [
            torch.randn(2, 4, 64, device=device),
            torch.randn(2, 4, 64, device=device),
            torch.randn(2, 4, 64, device=device),
        ],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        extra_check=_rope_extra_check,
        skip_duplicates=True,
    )

    # TODO: Add pattern for cvt.rn.bf16x2.ue8m0x2 (e8m0 -> bf16 conversion)
    # This is the inverse operation for MX format dequantization


class NumpyCompatNormalization:
    numpy_compat: dict[str, tuple[str, ...]] = {
        "dim": ("axis",),
        "keepdim": ("keepdims",),
        "input": ("x", "a", "x1"),
        "other": ("x2",),
    }
    inverse_mapping: dict[str, str]
    cache: dict["torch.fx.graph.Target", OrderedSet[str]]

    def __init__(self) -> None:
        self.cache = {}  # callable -> tuple of replaceable args e.g. ["axis"]
        self.inverse_mapping = {}
        for actual_kwarg, numpy_kwargs in self.numpy_compat.items():
            for numpy_kwarg in numpy_kwargs:
                assert numpy_kwarg not in self.inverse_mapping
                self.inverse_mapping[numpy_kwarg] = actual_kwarg

    def __call__(self, graph: torch.fx.Graph):
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if isinstance(node.target, (OpOverload, OpOverloadPacket)):
                # only applies to torch ops; e.g. torch.stack(axis=1) works, torch.ops.aten.stack(axis=1) doesn't.
                continue
            kwargs = node.kwargs

            if node.target in self.cache:
                replaceable_kwargs = self.cache[node.target]
            else:
                signatures = torch.fx.operator_schemas.get_signature_for_torch_op(
                    node.target
                )
                signatures = () if signatures is None else signatures
                replaceable_kwargs = OrderedSet()
                for sig in signatures:
                    for param_name in sig.parameters:
                        if param_name in self.numpy_compat:
                            replaceable_kwargs.update(self.numpy_compat[param_name])

                self.cache[node.target] = replaceable_kwargs

            if not replaceable_kwargs:
                continue

            new_kwargs = {}
            kwargs_changed = False
            for k, v in kwargs.items():
                if k in replaceable_kwargs:
                    kwargs_changed = True
                    new_kwargs[self.inverse_mapping[k]] = v
                else:
                    new_kwargs[k] = v

            if kwargs_changed:
                node.kwargs = torch.fx.immutable_collections.immutable_dict(new_kwargs)
                counters["inductor"]["numpy_compat_normalization"] += 1


numpy_compat_normalization = NumpyCompatNormalization()
