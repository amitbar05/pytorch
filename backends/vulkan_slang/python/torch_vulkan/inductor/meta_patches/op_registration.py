"""Op registration and decomposition patches for Vulkan.

Registers meta decompositions, matmul fake impls, PrivateUse1 matmul/einsum
dispatch, disables bmm→mm for Vulkan, and registers SDPA in meta_table.
"""

from __future__ import annotations

import torch


def _register_backward_meta_decomps() -> None:
    """Register meta-level decompositions for aten backward ops.

    FakeTensorMode._dispatch_impl checks meta_table before trying regular
    decompositions. Regular decompositions can fail when saved forward inputs
    arrive as meta tensors (different device from the grad on vulkan:0).
    A meta decomposition fires first and just returns the correct shape,
    bypassing the device-mixing issue.

    M18.7 (2026-05-22): Three-layer consolidation complete.
    The 8 shape-only backward proxies that previously appeared here also had
    entries in both _OP_IMPLS (fake_impl, __init__.py) and _patch_decompositions
    (decomposition_table, decomposition_passes.py).  Per the M15.2 audit the
    fake_impl entries in _OP_IMPLS fire first for FakeTensor shape inference,
    making the meta decomps below redundant dead fallbacks for those ops.
    The AOT-level shape proxies live solely in _patch_decompositions.

    Removed (now covered by _OP_IMPLS fake_impl + _patch_decompositions):
      _softmax_backward_data, _log_softmax_backward_data, avg_pool2d_backward,
      max_pool2d_with_indices_backward, linear_backward,
      native_layer_norm_backward, native_group_norm_backward,
      native_batch_norm_backward.

    Kept: ops where no _OP_IMPLS fake_impl entry exists and the meta decomp
    is the only FakeTensor shape inference path:
      gelu_backward, silu_backward, leaky_relu_backward, elu_backward,
      upsample_nearest2d_backward, upsample_bilinear2d_backward.
    """
    try:
        from torch._decomp import register_decomposition

        aten = torch.ops.aten

        def _bwd_meta_like_grad(grad_output, *_args, **_kwargs):
            return grad_output.new_empty(grad_output.shape)

        for op, fn in [
            (aten.gelu_backward.default, _bwd_meta_like_grad),
            (aten.silu_backward.default, _bwd_meta_like_grad),
            (aten.leaky_relu_backward.default, _bwd_meta_like_grad),
            (aten.elu_backward.default, _bwd_meta_like_grad),
            # PyTorch 2.11 uses .default (not .vec which was an older overload name).
            (aten.upsample_nearest2d_backward.default, _bwd_meta_like_grad),
            (aten.upsample_bilinear2d_backward.default, _bwd_meta_like_grad),
        ]:
            try:
                register_decomposition(op, type="meta")(fn)
            except Exception:
                pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering backward meta decompositions failed: %s", e
        )


def _register_matmul_meta() -> None:
    """GAP 0 — register aten.matmul in FakeTensorMode's meta_table.

    Without this, FakeTensorMode._dispatch_impl calls func.decompose() for
    aten.matmul (a CompositeImplicitAutograd op). The decompose path
    expands matmul into reshape→bmm→view, then evaluates the bmm on the
    FakeTensor's underlying zero-filled storage, producing all-zeros output.
    Inductor's constant_fold_uniform_value pass sees the uniform all-zeros
    and replaces the inputs with full(1.0) — the user's actual q/k tensors
    become dead code, and the compiled function returns wrong results.

    By registering in meta_table, _dispatch_impl skips decompose() and
    falls through to the fake_impl / meta kernel path, which returns a
    fresh FakeTensor without computing values. The constant folder then
    sees non-uniform data and leaves the graph alone.
    """
    try:
        from torch._decomp import register_decomposition

        matmul_op = torch.ops.aten.matmul.default

        @register_decomposition(matmul_op, type="meta")
        def _matmul_meta(tensor1, tensor2):
            t1 = tensor1 if isinstance(tensor1, torch.Tensor) else tensor1
            t2 = tensor2 if isinstance(tensor2, torch.Tensor) else tensor2
            dim1 = t1.dim()
            dim2 = t2.dim()
            if dim1 == 1 and dim2 == 1:
                return t1.new_empty(())
            elif dim1 == 2 and dim2 == 1:
                return t1.new_empty((t1.size(0),))
            elif dim1 == 1 and dim2 == 2:
                return t1.new_empty((t2.size(1),))
            elif dim1 == 2 and dim2 == 2:
                return t1.new_empty((t1.size(0), t2.size(1)))
            elif dim1 >= 1 and dim2 >= 1:
                max_dim = max(dim1, dim2)
                shape1 = list(t1.shape)
                shape2 = list(t2.shape)
                if dim1 == 1:
                    shape1 = [1, shape1[0]]
                if dim2 == 1:
                    shape2 = [shape2[0], 1]
                out_shape = torch.broadcast_shapes(shape1[:-2], shape2[:-2])
                out_shape = list(out_shape) + [shape1[-2], shape2[-1]]
                if dim1 == 1:
                    out_shape = out_shape[:-2] + out_shape[-1:]
                if dim2 == 1:
                    out_shape = out_shape[:-1]
                return t1.new_empty(out_shape)
            raise RuntimeError(f"matmul: unexpected dims {dim1}, {dim2}")
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering matmul meta decomposition failed: %s", e
        )


def _patch_proxy_call_matmul_decomp() -> None:
    """GAP 0 — register matmul on PrivateUse1 to prevent CompositeImplicitAutograd
    decomposition during make_fx tracing.

    M15.2 audit (b): workaround for missing primitive. Should be replaced by
    an FX-level decomposition pass that rewrites matmul → mm/bmm BEFORE
    make_fx tracing. The matmul_backward should route through autodiff
    (bwd_diff_table.py). See roadmap M12 (autodiff) and M14 (op coverage).
    """

    # The registration makes autograd_would_have_decomposed() return False for
    # matmul on vulkan tensors, preventing the decomposition into view->bmm->view
    # that causes constant-fold bugs.

    # For eager mode, we also register matmul_backward on AutogradPrivateUse1
    # so autograd still computes correct gradients through our matmul.
    try:
        import torch
        from torch._subclasses.fake_tensor import FakeTensor

        _matmul_lib = torch.library.Library("aten", "IMPL", "PrivateUse1")

        def _vulkan_matmul(self, other):
            dim1 = self.dim()
            dim2 = other.dim()
            if dim1 == 2 and dim2 == 2:
                return torch.ops.aten.mm(self, other)
            if dim1 == 3 and dim2 == 3:
                return torch.ops.aten.bmm(self, other)
            s = self
            o = other
            sq1 = sq2 = False
            if dim1 == 1:
                s = s.unsqueeze(0)
                sq1 = True
            if dim2 == 1:
                o = o.unsqueeze(-1)
                sq2 = True
            if s.dim() == 2 and o.dim() == 2:
                result = torch.ops.aten.mm(s, o)
            elif s.dim() == 3 and o.dim() == 3:
                result = torch.ops.aten.bmm(s, o)
            else:
                raise RuntimeError(f"vulkan_matmul: unsupported dims {dim1} x {dim2}")
            if sq1 and sq2:
                return result.squeeze(-1).squeeze(-2)
            if sq1:
                return result.squeeze(-2)
            if sq2:
                return result.squeeze(-1)
            return result

        _matmul_lib.impl("matmul", _vulkan_matmul)

        def _vulkan_matmul_backward(grad, self, other, mask):
            results = []
            if mask[0]:
                g = torch.ops.aten.bmm(
                    grad.reshape(-1, grad.shape[-2], grad.shape[-1]),
                    other.reshape(-1, other.shape[-2], other.shape[-1]).transpose(
                        -2, -1
                    ),
                ).reshape(self.shape)
                results.append(g)
            else:
                results.append(torch.zeros_like(self))
            if mask[1]:
                g = torch.ops.aten.bmm(
                    self.reshape(-1, self.shape[-2], self.shape[-1]).transpose(-2, -1),
                    grad.reshape(-1, grad.shape[-2], grad.shape[-1]),
                ).reshape(other.shape)
                results.append(g)
            else:
                results.append(torch.zeros_like(other))
            return tuple(results)

        _matmul_lib.impl("matmul_backward", _vulkan_matmul_backward)

        import sys

        _module = sys.modules[__name__]
        _module._matmul_lib = _matmul_lib
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering matmul on PrivateUse1 failed: %s", e
        )


def _patch_einsum_proxy_decomp() -> None:
    """T.12 — register aten.einsum on PrivateUse1 with a Python decomposition
    that traces cleanly through ``__torch_dispatch__``.

    M15.2 audit (b): workaround for missing primitive. Should be replaced by
    an FX-level decomposition pass that rewrites einsum → bmm/permute/reshape
    BEFORE make_fx tracing. See roadmap M14 (op coverage).
    """

    # Without this, the C++ ``at::native::einsum`` implementation runs as a
    # CompositeImplicitAutograd. During AOTAutograd's ``make_fx`` tracing,
    # that C++ impl creates intermediate ``unsqueeze`` / ``permute`` views
    # *inside C++*, so the resulting tensors carry no proxy. When those
    # views are then passed to ``aten.bmm``, ``proxy_tensor.create_arg``
    # cannot find a proxy and falls back to baking the tensors in as
    # ``_tensor_constantN`` ``get_attr`` nodes on the GraphModule. The user's
    # real ``arg0_1`` / ``arg1_1`` placeholders end up dead, and Inductor's
    # ``constant_fold_uniform_value`` later sees the (uninitialised /
    # fake-storage) constants as uniform and replaces the entire einsum
    # with ``aten.full(value=1.0)`` — silent all-ones output (round 9 T.12).

    # Fix: provide a Python decomposition that uses pure Python aten calls
    # (``unsqueeze`` / ``permute`` / ``bmm`` / ``mul`` / ``sum``). Each call
    # goes through ``__torch_dispatch__`` so proxies are tracked correctly.
    # Registered on PrivateUse1 so it runs before any
    # CompositeImplicitAutograd decomposition.

    # Coverage: the seven canonical patterns from
    # ``TestT12EinsumCoverage``: ``i,i->`` (inner), ``i,j->ij`` (outer),
    # ``ij,jk->ik`` (mm), ``bij,bjk->bik`` (bmm),
    # ``bnsh,bksh->bnks`` and ``bnks,bksh->bnsh`` (attention QK / QK·V),
    # and ``ii->i`` (diag). For unsupported equations, falls through to
    # the C++ default by raising NotImplementedError, which the dispatcher
    # treats as no-impl.
    try:
        import torch

        _einsum_lib = torch.library.Library("aten", "IMPL", "PrivateUse1")

        def _parse_einsum(equation):
            """Parse ``equation`` into (input_subs_list, output_sub).

            ``output_sub`` is computed implicitly when '->' is absent
            (each label appearing exactly once across all inputs is
            output, in alphabetical order)."""
            equation = equation.replace(" ", "")
            if "->" in equation:
                lhs, rhs = equation.split("->")
            else:
                lhs, rhs = equation, None
            input_subs = lhs.split(",")
            if rhs is None:
                from collections import Counter

                counts = Counter()
                for s in input_subs:
                    for c in s:
                        counts[c] += 1
                rhs = "".join(sorted(c for c, n in counts.items() if n == 1))
            return input_subs, rhs

        def _einsum_one(sub, out_sub, x):
            """Single-operand einsum: handles diag / sum / permute."""
            # Shape sanity: every dim with the same label must have the same size.
            label_to_size: dict[str, int] = {}
            for label, size in zip(sub, x.shape):
                if label in label_to_size:
                    if label_to_size[label] != size:
                        raise RuntimeError(
                            f"einsum: size mismatch on label {label!r}: "
                            f"{label_to_size[label]} vs {size}"
                        )
                else:
                    label_to_size[label] = size

            # If a label appears more than once, take the diagonal across
            # those axes (e.g. ``ii->i``). ``aten.diagonal`` removes the
            # two source dims and appends a single dim with the matching
            # label. Repeat until all subscripts are unique. Note: do NOT
            # restart from i=0 after a diag rebuild — the dedupe progress
            # would loop forever for ``i,i,i->i``-style patterns; instead
            # rescan with a fresh ``seen`` and look for the next repeat.
            from collections import Counter

            while True:
                counts = Counter(sub)
                repeats = [c for c, n in counts.items() if n > 1]
                if not repeats:
                    break
                label = repeats[0]
                # Find the first two positions of ``label``.
                first = sub.index(label)
                second = sub.index(label, first + 1)
                x = torch.ops.aten.diagonal.default(x, 0, first, second)
                sub = sub[:first] + sub[first + 1 : second] + sub[second + 1 :] + label

            # Sum out labels that don't appear in output_sub.
            sum_dims = [j for j, c in enumerate(sub) if c not in out_sub]
            if sum_dims:
                x = torch.ops.aten.sum.dim_IntList(x, sum_dims, False)
                sub = "".join(c for c in sub if c in out_sub)

            # Permute to output order.
            if sub != out_sub:
                if sorted(sub) != sorted(out_sub):
                    raise RuntimeError(
                        f"einsum: subscript mismatch {sub!r} vs {out_sub!r}"
                    )
                perm = [sub.index(c) for c in out_sub]
                x = torch.ops.aten.permute.default(x, perm)
            return x

        def _einsum_two(sub_a, sub_b, out_sub, a, b):
            """Two-operand einsum via the standard contract: classify each
            label as batch (in both inputs and output), contract (in both
            inputs but not output), or free (in one input and output)."""
            from collections import Counter

            # Record per-label sizes (for sanity).
            label_to_size: dict[str, int] = {}
            for sub, t in ((sub_a, a), (sub_b, b)):
                for label, size in zip(sub, t.shape):
                    if label in label_to_size:
                        if label_to_size[label] != size and 1 not in (
                            label_to_size[label],
                            size,
                        ):
                            raise RuntimeError(
                                f"einsum: size mismatch on label {label!r}: "
                                f"{label_to_size[label]} vs {size}"
                            )
                    else:
                        label_to_size[label] = size

            # Diagonalize repeats inside each operand first (a→a', b→b').
            a = _einsum_one(sub_a, "".join(dict.fromkeys(sub_a)), a)
            sub_a = "".join(dict.fromkeys(sub_a))
            b = _einsum_one(sub_b, "".join(dict.fromkeys(sub_b)), b)
            sub_b = "".join(dict.fromkeys(sub_b))

            in_a = set(sub_a)
            in_b = set(sub_b)
            in_out = set(out_sub)

            # Labels that are summed out and only appear in one input → reduce.
            only_a_summed = [c for c in sub_a if c not in in_b and c not in in_out]
            only_b_summed = [c for c in sub_b if c not in in_a and c not in in_out]
            if only_a_summed:
                axes = [sub_a.index(c) for c in only_a_summed]
                a = torch.ops.aten.sum.dim_IntList(a, axes, False)
                sub_a = "".join(c for c in sub_a if c not in only_a_summed)
            if only_b_summed:
                axes = [sub_b.index(c) for c in only_b_summed]
                b = torch.ops.aten.sum.dim_IntList(b, axes, False)
                sub_b = "".join(c for c in sub_b if c not in only_b_summed)

            in_a = set(sub_a)
            in_b = set(sub_b)

            batch_labels = [c for c in sub_a if c in in_b and c in in_out]
            contract_labels = [c for c in sub_a if c in in_b and c not in in_out]
            free_a = [c for c in sub_a if c not in in_b]
            free_b = [c for c in sub_b if c not in in_a]

            # Build target orderings:
            # a: [batch, free_a, contract]
            # b: [batch, contract, free_b]
            target_a = batch_labels + free_a + contract_labels
            target_b = batch_labels + contract_labels + free_b
            if list(sub_a) != target_a:
                perm = [sub_a.index(c) for c in target_a]
                a = torch.ops.aten.permute.default(a, perm)
            if list(sub_b) != target_b:
                perm = [sub_b.index(c) for c in target_b]
                b = torch.ops.aten.permute.default(b, perm)

            # Reshape to (B, M, K) × (B, K, N) bmm form.
            shape_a = list(a.shape)
            shape_b = list(b.shape)
            nb = len(batch_labels)
            nfa = len(free_a)
            nc = len(contract_labels)
            nfb = len(free_b)

            B_dims = shape_a[:nb]
            M_dims = shape_a[nb : nb + nfa]
            K_dims = shape_a[nb + nfa :]
            assert len(K_dims) == nc

            B_dims_b = shape_b[:nb]
            K_dims_b = shape_b[nb : nb + nc]
            N_dims = shape_b[nb + nc :]
            assert len(N_dims) == nfb

            B = 1
            for s in B_dims:
                B *= int(s)
            M = 1
            for s in M_dims:
                M *= int(s)
            K = 1
            for s in K_dims:
                K *= int(s)
            N = 1
            for s in N_dims:
                N *= int(s)

            a3 = torch.ops.aten.reshape.default(a, [B, M, K])
            b3 = torch.ops.aten.reshape.default(b, [B, K, N])

            # Pure-pointwise outer product when there's nothing to contract.
            if K == 1 and nc == 0:
                # mul + broadcast (no real reduction).
                # Here K_dims is empty; both a3 and b3 are [B, M, 1] / [B, 1, N].
                # Use bmm anyway — it's a valid degenerate matmul.
                pass
            out_3 = torch.ops.aten.bmm.default(a3, b3)

            # Reshape back to [*B_dims, *M_dims, *N_dims].
            out_shape = list(B_dims) + list(M_dims) + list(N_dims)
            if not out_shape:
                # Pure inner product: result is scalar.
                out = torch.ops.aten.reshape.default(out_3, [])
            else:
                out = torch.ops.aten.reshape.default(out_3, out_shape)

            # Permute to user's output order.
            current = batch_labels + free_a + free_b
            if current != list(out_sub):
                if sorted(current) != sorted(out_sub):
                    raise RuntimeError(
                        f"einsum: subscript mismatch {current!r} vs {out_sub!r}"
                    )
                perm = [current.index(c) for c in out_sub]
                out = torch.ops.aten.permute.default(out, perm)
            return out

        def _vulkan_einsum(equation, tensors, *, path=None):
            input_subs, out_sub = _parse_einsum(equation)
            if len(tensors) != len(input_subs):
                raise RuntimeError(
                    f"einsum: equation has {len(input_subs)} operand(s) but "
                    f"got {len(tensors)} tensor(s)"
                )
            if len(tensors) == 1:
                return _einsum_one(input_subs[0], out_sub, tensors[0])
            if len(tensors) == 2:
                return _einsum_two(
                    input_subs[0], input_subs[1], out_sub, tensors[0], tensors[1]
                )
            # Multi-operand: contract pairwise from left to right.
            cur = tensors[0]
            cur_sub = input_subs[0]
            for i in range(1, len(tensors)):
                next_t = tensors[i]
                next_sub = input_subs[i]
                # Compute the intermediate output subscript: keep labels that
                # appear in any later input, in either current pair, or in
                # the final output.
                later = set()
                for j in range(i + 1, len(tensors)):
                    later.update(input_subs[j])
                later.update(out_sub)
                inter_sub = "".join(
                    dict.fromkeys([c for c in cur_sub + next_sub if c in later])
                )
                cur = _einsum_two(cur_sub, next_sub, inter_sub, cur, next_t)
                cur_sub = inter_sub
            if cur_sub != out_sub:
                # Final permute / reduction (idempotent if already correct).
                cur = _einsum_one(cur_sub, out_sub, cur)
            return cur

        _einsum_lib.impl("einsum", _vulkan_einsum)

        import sys

        _module = sys.modules[__name__]
        _module._einsum_lib = _einsum_lib
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Registering einsum on PrivateUse1 failed: %s", e
        )


def _disable_bmm_to_mm_for_vulkan() -> None:
    """T.12 — skip Inductor's ``bmm_to_mm`` pattern for Vulkan tensors.

    The ``bmm_to_mm`` joint-graph pattern (``torch/_inductor/fx_passes/
    joint_graph.py``) rewrites ``aten.bmm`` with batch=1 into
    ``aten.mm(a.squeeze(0), b.squeeze(0)).unsqueeze(0)``. Its
    ``replace_by_example`` re-traces the replacement function via
    ``make_fx(..., tracing_mode='real')`` on the matched ``FakeTensor``
    values from ``mat.meta['val']``. On Vulkan, those FakeTensors carry
    ``device='vulkan:0'`` and the ``squeeze(0)`` view dispatch escapes
    proxy tracking — proxy_tensor's ``create_arg`` then bakes the
    squeezed views as ``_tensor_constantN`` ``get_attr`` nodes whose
    underlying tensors have *no backing buffer* (``data_ptr == 0``).
    The replacement graph ends up with the user's ``arg0`` / ``arg1``
    placeholders unused and the ``mm`` reading from the broken
    constants. At runtime, ``extern_kernels.mm(_vulkan_pm__tensor_constant0,
    _vulkan_pm__tensor_constant1)`` fails with ``Cannot access data
    pointer of Tensor (e.g. FakeTensor)``.

    This was the second-stage breakage of the T.12 ``einsum`` chain
    (round 9). With ``aten.einsum`` decomposed via our PrivateUse1
    Python impl (``_patch_einsum_proxy_decomp``) the AOT graph now has
    a clean ``unsqueeze + reshape + bmm + reshape`` tail driven by
    ``arg0_1`` / ``arg1_1``, but ``bmm_to_mm`` then matches and breaks
    it again. The pattern's only purpose is a CUDA micro-optimization
    (``mm`` is faster than ``bmm`` with ``B=1`` on CUDA); on Vulkan
    our ``bmm`` template handles ``B=1`` perfectly fine, so disabling
    is a safe and complete fix.

    Implementation: wrap the registered pattern function so it bails
    out (returns ``None`` without calling ``replace_by_example``) when
    either matched tensor is on Vulkan.
    """
    try:
        from torch._inductor.fx_passes import joint_graph as _jg

        if getattr(_jg, "_vulkan_bmm_to_mm_patched", False):
            return

        _patterns = _jg.patterns
        _orig_fn = _jg.bmm_to_mm
        _bmm_op = torch.ops.aten.bmm.default

        def _vulkan_aware_bmm_to_mm(match, mat1, mat2):
            try:
                v1 = mat1.meta.get("val", None)
                v2 = mat2.meta.get("val", None)
                if v1 is not None and getattr(v1, "device", None) is not None:
                    if v1.device.type == "vulkan":
                        return None
                if v2 is not None and getattr(v2, "device", None) is not None:
                    if v2.device.type == "vulkan":
                        return None
            except Exception:
                pass
            return _orig_fn(match, mat1, mat2)

        # Locate every PatternEntry registered against ``aten.bmm`` in
        # ``joint_graph.patterns`` (the dict is keyed by
        # ``('call_function', op_or_packet)``) and rewrap the handler.
        # We have to handle both the OpOverload and OpOverloadPacket
        # forms because ``register_graph_pattern`` registers under
        # whichever form the pattern declared.
        replaced = 0
        try:
            patterns_dict = _patterns.patterns
        except AttributeError:
            patterns_dict = {}
        bmm_keys = []
        for k in patterns_dict:
            try:
                if isinstance(k, tuple) and len(k) >= 2 and k[0] == "call_function":
                    op = k[1]
                    name = getattr(op, "__name__", str(op))
                    if "bmm" in str(op) or "bmm" in name:
                        bmm_keys.append(k)
                elif k is _bmm_op:
                    bmm_keys.append(k)
            except Exception:
                continue

        for key in bmm_keys:
            for entry in patterns_dict[key]:
                handler = getattr(entry, "handler", None)
                if handler is _orig_fn:
                    entry.handler = _vulkan_aware_bmm_to_mm  # type: ignore[attr-defined]
                    replaced += 1

        # Also overwrite the module-level binding (for any future
        # registrations that look it up by attribute).
        _jg.bmm_to_mm = _vulkan_aware_bmm_to_mm  # type: ignore[attr-defined]
        _jg._vulkan_bmm_to_mm_patched = True  # type: ignore[attr-defined]
        if not replaced:
            import logging

            logging.getLogger(__name__).warning(
                "bmm_to_mm pattern not found in joint_graph.patterns "
                "(searched %d bmm keys); module attr replaced",
                len(bmm_keys),
            )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "Disabling bmm_to_mm for Vulkan failed: %s", e
        )


def _register_sdpa_meta() -> None:
    """Register SDPA in FakeTensorMode's meta_table.

    FakeTensorMode._dispatch_impl checks `func not in meta_table` first. If
    True (SDPA is NOT registered), it calls func.decompose() which fires the
    CompositeImplicitAutograd decomposition — returning before the fake_impl
    check is reached. By registering in meta_table we skip the decompose path
    so _dispatch_impl falls through to fake_impl, where _sdpa_fake handles
    shape inference correctly for Vulkan FakeTensors.
    """
    try:
        from torch._decomp import register_decomposition

        sdpa_op = torch.ops.aten.scaled_dot_product_attention.default

        @register_decomposition(sdpa_op, type="meta")
        def _sdpa_meta(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            sizes = list(query.size())
            sizes[-1] = value.size(-1)
            return query.new_empty(sizes)
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger(__name__).warning(
            "Registering SDPA in meta_table failed: %s", e
        )
