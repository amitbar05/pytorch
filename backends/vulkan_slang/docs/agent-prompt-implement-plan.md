# Agent Prompt: Implement the Codegen Optimization Plan

> Copy this entire prompt into a new agent session.
> The agent reads the plan, picks the highest-priority unblocked item, and ships it.
> Repeat until the plan is exhausted.

---

## Session Prompt

```
You are implementing the Vulkan/Slang Inductor backend optimization plan.
Read the three plan documents FIRST, then start implementing.

═══════════════════════════════════════════════════════════════
READ THESE IN ORDER:
═══════════════════════════════════════════════════════════════

1. docs/10-inductor-backend.md          ← Roadmap: all tracks, active TRAIN.* items
2. docs/codegen-optimization-roadmap.md ← Gap analysis: 5 tiers, priority matrix, model coverage
3. docs/how-to-compile-and-codegen.md   ← Reference: compile APIs, lowering pipeline, backend dispatch

═══════════════════════════════════════════════════════════════
PRIORITY ORDER — work on the highest unblocked item:
═══════════════════════════════════════════════════════════════

P0 — FIX CORRECTNESS (blocks ALL models):
  P0.1  C1: ReLU backward returns [] gradient
         → Diagnose why AOT Autograd's ReluBackward0 doesn't emit gradients
           for PrivateUse1 tensors. Check AutogradPrivateUse1 registration,
           FunctionalTensorMode interaction, and FakeTensor propagation.
         → Test: training step with ReLU model produces non-zero gradients.
  P0.2  C2: Conv backward has no autograd formula
         → Register convolution_backward in AutogradPrivateUse1.
         → Depends on P0.1 (same autograd registration subsystem).
  P0.3  TRAIN.2: Implement mm/bmm/addmm Jinja backward templates
         → Create templates/slang_mm_bwd.py.jinja with tile-loop backward.
         → Wire bwd_template_registry.py entries into bwd_diff_dispatch.py.
         → The 3 placeholder entries (mm/bmm/addmm BackwardKind.TEMPLATE_JINJA)
           need actual template bodies.
  P0.4  TRAIN.3: Migrate remaining hand-written backward lowerings
         → Route the 12 remaining aten.*_backward lowerings through
           _ensure_unary_bwd_diff_op() and delete hand-coded impls.
         → Target ops: elu_backward, hardswish_backward, hardsigmoid_backward,
           mish_backward, softplus_backward, selu_backward, and 6 norm backwards.
  P0.5  TRAIN.6: Fix combo-kernel wave-mask uniformity
         → Batched same-shape outer-axis reductions produce zero outputs.
         → The gtid-to-output model in vulkan_combo_kernel.py conflicts
           with reduction kernels needing per-WG output assignment.

P1 — EXPAND MODEL COVERAGE (unlocks new model classes):
  P1.1  TRAIN.10: Dynamic shapes — implement dispatch grid + buffer binding
         → Finish kernel/symbolic.py beyond the 57-line stub.
         → Dynamic dispatch grid in kernel/header.py.
         → Dynamic buffer binding from Slang reflection in runtime.py.
  P1.2  TRAIN.7: AMP/fp16 autocast codegen path
         → Zero matches for autocast/fp16/bf16 in scheduling.py or codegen.py.
         → Need packed16 path plumbed through prefer_packed16.
         → Dtype propagation through joint graph.
  P1.3  TRAIN.8: Extern-kernel pool allocator hook
         → mm/conv/SDPA outputs bypass the buffer pool.
         → These are the largest training tensors.
         → Wire extern-kernel alloc through vulkan_pool_acquire.
  P1.4  P4.4: Un-gate Flash attention template
         → flash_attention.py.jinja (261 lines) is complete but gated with
           NotImplementedError. Ungate it, add _render_flash_attention,
           create FlashAttentionTemplate class, install_external_flash_attention().
         → Also add shaders/lib/attention.slang stub if needed.
  P1.5  T4.5: Implement scatter/gather/index_put template
         → Currently unsupported — falls back to eager dispatch.
         → Blocks NLP (embeddings) and GNN models.
  P1.6  N.1/N.2: Implement SCAN and SORT via wave primitives
         → Currently ADVERTISED in get_backend_features() but NOT IMPLEMENTED.
         → If Inductor fuses a scan/sort pattern, it WILL produce broken kernels.
         → Implement via WavePrefixSum (scan) and bitonic sort (sort) in
           kernel/reduction.py, using reduction.slang generics.

P2 — GPU PERFORMANCE OPTIMIZATION (generated code is correct, make it fast):
  P2.1  M1: Wave primitives — use WavePrefixSum, WavePrefixProduct,
         WaveReadLaneFirst, WaveReadLaneAt in reduction codegen.
         → Currently only WaveActiveSum/Max/Min/Prod used.
         → Enables single-WG scan without shared memory.
  P2.2  M2: SPIR-V specialization constants
         → Replace push-constant numel/size vars with [[vk::constant_id(N)]]
           when values are statically known at compile time.
         → Lets slangc constant-fold loops and eliminate dead branches.
  P2.3  M3: Descriptor indexing (VK_EXT_descriptor_indexing)
         → Remove the 16-buffer cap by enabling UPDATE_AFTER_BIND.
         → Enables larger fusion groups.
  P2.4  M4: Register-pressure-aware workgroup sizing
         → _pick_threadgroup_size currently keys on rnumel only.
         → Use dtype + estimated VGPRs (from Slang reflection or heuristics)
           to avoid oversubscribing registers.
  P2.5  M20: Bank conflict padding in shared memory
         → Add +32 padding to groupshared arrays where access stride
           would cause bank conflicts on RDNA1's 32 banks.
  P2.6  M21: Parallel slangc compilation
         → slangc subprocess per kernel is serial today.
         → Use TORCH_VULKAN_ASYNC_COMPILE=1 ThreadPoolExecutor for
           template variant pre-warming.

P3 — PRODUCTIONIZATION (deploy without Python):
  P3.1  P4.6: ParameterBlock<T> migration
         → Replace manual [[vk::binding(N)]] slot += 1 with ParameterBlock.
         → Enables VkPipelineLayout auto-derivation from SPIR-V reflection.
  P3.2  P4.7: Link-time specialization for templates
         → extern static const int TILE_M; in precompiled module.
         → 112 slangc invocations → 2 (one per dtype module).
  P3.3  P4.8: Slang reflection integration
         → slangc -dump-reflection → VGPR, shared_mem, loop_depth.
         → Feed into WG sizing and SPIR-V perf regression tracker.
  P3.4  AOTI e2e: Python-less fwd+bwd+optimizer under AOTInductor.
  P3.5  CI/CD + model zoo regression suite.

═══════════════════════════════════════════════════════════════
RULES:
═══════════════════════════════════════════════════════════════

1. Pick ONE item, ship it completely (code + test), then move to the next.
2. Tests go in tests/test_inductor_regression.py.
3. Every fix gets an xfail(strict=True) test first, then flip it to passing.
4. Correctness before performance. Gradients must match CPU at rtol=1e-3.
5. Verify with: python -m pytest tests/test_inductor_regression.py -x --timeout=120
6. After each item, update docs/10-inductor-backend.md checkboxes.
7. If blocked, skip and note why. Come back after the blocker is resolved.
8. For C++ changes: TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=8 python setup.py build_ext --inplace
9. For Slang lib changes: precompile_shader_libs(force=True)
10. NEVER edit generated files in build/.

═══════════════════════════════════════════════════════════════
FILE SCOPES FOR PARALLEL WORK:
═══════════════════════════════════════════════════════════════

Disjoint file groups — items from different groups can run in parallel:

Group A (autograd/C++):       csrc/backend/Registration.cpp, csrc/autocast/
Group B (lowerings/bwd):      lowerings/*.py, bwd_diff_dispatch.py, bwd_diff_table.py
Group C (codegen/kernel):     kernel/*.py, expr_printer.py, overrides.py
Group D (templates):          vulkan_template.py, vulkan_template_caller.py, templates/
Group E (scheduler/fusion):   scheduling.py, vulkan_combo_kernel.py
Group F (runtime):            runtime.py, buffer_pool.py, wrapper.py
Group G (Slang lib):          shaders/lib/*.slang
Group H (fx passes):          fx_passes/

Never dispatch agents that both touch the same group concurrently.
Rebuild C++ only once after all C++ edits are done.

═══════════════════════════════════════════════════════════════
START HERE: Read the 3 plan docs, then implement P0.1 first.
═══════════════════════════════════════════════════════════════
```
