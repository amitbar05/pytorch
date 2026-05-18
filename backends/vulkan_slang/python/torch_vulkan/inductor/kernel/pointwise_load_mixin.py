"""Pointwise load mixin — dtype dispatch, packed16 decisions, and buffer loading.

Extracted from ``PointwiseMixin`` (M15.1.d — Track 1 anti-goal #7 split).
"""

import sympy
import torch
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.virtualized import V

from .symbolic import is_dynamic_stride


# dtype → (emit_fn, header_tag) dispatch table used by PointwiseLoadMixin.load().
# ``emit_fn(var, idx_str)`` returns a Slang expression that loads one
# element from ``var`` at index ``idx_str`` into a float register.
# ``header_tag``, if non-None, is added to ``self.headers`` so
# ``slang_helpers.emit_helpers`` emits the supporting function.
_LOAD_DISPATCH: dict = {}  # populated lazily for torch.dtype imports


def _init_load_dispatch() -> None:
    import torch

    global _LOAD_DISPATCH
    if _LOAD_DISPATCH:
        return
    # M18.4-followup-C: narrow integer dtypes (bool / int8 / uint8 / int16
    # / uint16) now bind at their NATIVE element width — ``int8_t`` /
    # ``uint8_t`` / ``int16_t`` / ``uint16_t`` ``RWStructuredBuffer<T>``
    # — once the Vulkan ``shaderInt{8,16}`` + 8/16-bit storage features
    # are enabled in ``csrc/vulkan/Context.cpp``. With element widths
    # matching PyTorch's ``c10::elementSize(dtype)``, the M17.8.d.3
    # tail-corruption bug class is structurally CLOSED for the integer
    # half.
    #
    # The load expression is now the natural ``(float)(v[i])`` cast for
    # every signed/unsigned narrow int — Slang's implicit widening from
    # ``int8_t`` / ``int16_t`` already sign-extends correctly when the
    # next op is a float cast (the bit-twiddle sign extends we used
    # before M18.4-followup-C were stopgaps for the
    # ``StructuredBuffer<uint>`` 4B-slot binding).
    #
    # ``bfloat16`` still binds as a 32-bit ``uint`` slot — see the
    # ``M18.4-followup-bfloat16`` comment in ``overrides.py``. The
    # ``packed16_bf16`` load path covers eligible fusion shapes; the
    # bare-load fallback below stays at ``(float)(v[i])`` for now.
    _LOAD_DISPATCH.update(
        {
            torch.bool: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.uint8: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.int8: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.int16: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.uint16: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.float16: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.bfloat16: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.int32: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.uint32: (lambda v, i: f"((float)({v}[{i}]))", None),
            torch.int64: (lambda v, i: f"((float)(int)({v}[{i}].x))", None),
        }
    )


class PointwiseLoadMixin:
    """Mixin providing pointwise buffer-load codegen.

    Handles dtype dispatch, packed16 decisions, buffer-path resolution,
    and the main ``load()`` method that the Inductor scheduler calls
    to emit a buffer read into a CSE variable.
    """

    def _buf_path(self, inner_name: str) -> str:
        """Return the buffer access path for codegen emission.

        When ParameterBlock is enabled (P3.1/M9), buffer accesses use
        args.in_ptr0[idx] instead of in_ptr0[idx]. This helper
        returns args.{inner_name} or just {inner_name} based on
        the current mode.
        """
        if getattr(self, "_use_parameter_block", False):
            return f"args.{inner_name}"
        return inner_name

    def _decide_packed16(self, dtype: torch.dtype) -> bool:
        """Lazily decide whether this kernel uses packed16 mode.

        Called on every load/store.  Returns True only when eligible AND the
        dtype of the new buffer matches the dtype already locked in.  Flips
        self._packed16 to False permanently on the first disqualifying event.

        Eligibility rules (all must hold):
        - No reduction, multistage, or welford (unless _packed16_load_only).
        - All I/O buffers share the same half dtype (f16 or bf16).
        - Innermost non-reduction axis has even numel (so pairs of adjacent
          elements can be packed into one uint32 word).
        - For small persistent reductions (rnumel <= simd_group_size, even
          rnumel): packed16 is load-only — stores remain f32.
        """
        from .. import config

        if self._packed16 is False:
            return False

        if (
            self.has_welford
            or self.multistage_reduction_entry
            or config.no_packed16()
            or not config.prefer_packed16()
        ):
            self._packed16 = False
            return False

        if dtype not in (torch.float16, torch.bfloat16):
            self._packed16 = False
            return False

        if self._packed16 is None:
            axes = self.active_range_trees()
            non_red = [t for t in axes if not t.is_reduction]
            red = [t for t in axes if t.is_reduction]

            if self.inside_reduction and red:
                rnumel, _ = self._compute_red_numel()
                if (
                    not is_dynamic_stride(rnumel)
                    and int(rnumel) > 0
                    and int(rnumel) <= self.simd_group_size
                    and int(rnumel) % 2 == 0
                ):
                    self._packed16_load_only = True
                else:
                    self._packed16 = False
                    return False

            if not non_red:
                self._packed16 = False
                return False

            innermost = non_red[-1]
            if is_dynamic_stride(innermost.numel) or int(innermost.numel) % 2 != 0:
                self._packed16 = False
                return False

            self._packed16 = True
            self._packed16_dtype = dtype

            non_red_trees = [t for t in axes if not t.is_reduction]
            if (
                not self.inside_reduction
                and len(non_red_trees) == 1
                and not is_dynamic_stride(non_red_trees[0].numel)
                and int(non_red_trees[0].numel) % (self.max_threadgroup_size * 4) == 0
            ):
                self._packed16_vw_active = True

            return True

        if dtype != self._packed16_dtype:
            self._packed16 = False
            return False
        return True

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        dtype = V.graph.get_dtype(name)
        idx_str = self.index_to_str(index)

        # Track 5.7: Record sympy index for BlockPatternMatcher analysis.
        self._pw_index_records.append((var, index, True))

        if self._decide_packed16(dtype):
            self._pw_uses_subbyte_packing = True
            self._packed16_bufs.add(var)
            suffix = "f16" if dtype == torch.float16 else "bf16"
            self.headers.add(f"packed16_{suffix}")
            line = f"_vk_unpack_{suffix}({self._buf_path(var)}[({idx_str}) >> 1u], ({idx_str}) & 1u)"
            dtype = torch.float32
            cse_var = self.cse.generate(self.loads, line, dtype=dtype)
            self._p16_load_records.append((str(cse_var), var, suffix))
            return cse_var
        else:
            if (
                self._vec_width > 1
                and self.inside_reduction
                and self.multistage_reduction_entry
                and dtype in (torch.float32, torch.float16, torch.bfloat16)
                and self._reduction_type in ("sum", "prod", "max", "min")
            ):
                self.headers.add(f"vec4_reduce_{self._reduction_type}")
                rt = self._reduction_type
                if rt == "sum":
                    line = f"vk_vec4_hsum({self._buf_path(var)}, {idx_str})"
                elif rt == "max":
                    line = f"vk_vec4_hmax({self._buf_path(var)}, {idx_str})"
                elif rt == "min":
                    line = f"vk_vec4_hmin({self._buf_path(var)}, {idx_str})"
                else:
                    line = f"vk_vec4_hprod({self._buf_path(var)}, {idx_str})"
                if dtype != torch.float32:
                    line = f"((float)({line}))"
                    dtype = torch.float32
            else:
                _init_load_dispatch()
                # OP.1.b — external bool inputs are byte-packed.  PyTorch eager
                # allocates 1 byte per ``torch.bool`` element; the
                # ``dispatch_copy_buffer`` upload path (4 B/element) packs four
                # consecutive bool bytes into each uint32 slot of the SSBO.
                # Reading the slot as a single uint (the compile-internal
                # contract) returns 0x00010001 for ``[T,F,T,F]`` instead of the
                # per-element value.  For graph-input bool buffers we unpack
                # the byte at ``idx`` via ``_vk_unpack_u8`` (helpers.slang).
                # Compile-internal bool buffers (produced by a prior pointwise
                # STORE that writes 1 uint/element) keep the legacy
                # 1-uint-per-element read.
                if dtype == torch.bool and name in V.graph.graph_inputs:
                    self._pw_uses_subbyte_packing = True
                    self.headers.add("subdtype_unpack")
                    line = f"_vk_unpack_u8({self._buf_path(var)}, {idx_str})"
                    dtype = torch.float32
                else:
                    spec = _LOAD_DISPATCH.get(dtype)
                    if spec is not None:
                        emit_fn, hdr = spec
                        if hdr is not None:
                            self._pw_uses_subbyte_packing = True
                            self.headers.add(hdr)
                        line = emit_fn(self._buf_path(var), idx_str)
                        dtype = torch.float32
                    else:
                        line = f"{self._buf_path(var)}[{idx_str}]"

        from .. import config

        if (
            self.inside_reduction
            and self.multistage_reduction_entry
            and not getattr(self, "_partitioned_2d_active", False)
            and not self.has_welford
            and not config.no_load_hoist()
        ):
            key = (var, idx_str)
            cached = self.multistage_load_cache.get(key)
            if cached is not None:
                return self.cse.generate(self.loads, cached, dtype=dtype)
            root = self.multistage_reduction_entry[0].root
            if isinstance(root.numel, sympy.Integer):
                _hoist_stride = self.max_threadgroup_size * self._vec_width
                loop_size = (int(root.numel) + _hoist_stride - 1) // _hoist_stride
                # P5.4: Load-hoist threshold keyed on dtype and simd size.
                # Smaller dtypes → smaller per-element register footprint →
                # larger cache tolerated.  Smaller simd → more per-lane
                # registers available → larger cache.  Cap at 256 to prevent
                # pathological register spilling on RDNA1 (64 VGPRs/SIMD).
                _elt_bytes = 2 if self._packed16 else 4
                if dtype == torch.float64:
                    _elt_bytes = 8
                _dtype_scale = 4.0 / _elt_bytes
                _simd_scale = 64.0 / self.simd_group_size
                _base_limit = int(64 * _dtype_scale * _simd_scale)
                hoist_limit = min(max(_base_limit, 32), 256)
                if loop_size > hoist_limit:
                    return self.cse.generate(self.loads, line, dtype=dtype)
                arr_name = f"_ml_cache_{next(self.multistage_load_seq)}"
                cnt_name = f"{root.prefix}_cnt"
                self.indexing_code.writeline(f"float {arr_name}[{loop_size}];")
                cse_var = self.cse.generate(self.loads, line, dtype=dtype)
                self.loads.writeline(f"{arr_name}[{cnt_name}] = {cse_var};")
                self.multistage_load_cache[key] = f"{arr_name}[{cnt_name}]"
                return cse_var

        return self.cse.generate(self.loads, line, dtype=dtype)
