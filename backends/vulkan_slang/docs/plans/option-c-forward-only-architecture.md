# Option C: Forward-Only Compile Architecture

## Status
**EXPLORATION** (2026-05-28)

## Overview
A hybrid approach to bypass the AOT partitioner blocker for TRAIN.8 by splitting compilation into two phases: forward-only Inductor compilation and eager backward passes.

## Problem
The AOT autograd partitioner fails when `target` (class indices) enters the compiled sub-graph as a backward-only input:
```
AssertionError: Node div was invalid, but is output
```

This happens because Dynamo's graph break at `loss.backward()` causes `target` to be classified as backward-only, making forward nodes that depend on it invalid.

## Investigated Approaches (All Failed)

### Approach 1: Custom nll_loss_forward decomp
- **Method**: Replace AOT's `nll_loss_forward` decomp with one using `target.numel()` (constant)
- **Result**: Partitioner error on `div` node
- **Why failed**: Decomp still creates div operation that partitioner sees

### Approach 2: Manual cross_entropy via log_softmax+gather+neg+sum
- **Method**: Decompose `F.cross_entropy` into primitive ops
- **Result**: Partitioner error on `neg` node  
- **Why failed**: Forward nodes still depend on `target` (marked backward-only)

### Approach 3: Remove nll_loss_forward from AOT decomp tables
- **Method**: Force opaque op handling
- **Result**: Partitioner error on `getitem_2`
- **Why failed**: Same root cause - forward/backward classification issue

## Option C: Three-Phase Hybrid

### Key Finding (2026-05-28): Forward-Only Compile Works!

**✓ Validated**: `torch.compile(backend="inductor")` on a forward-only function bypasses the AOT partitioner entirely. Tested with a Conv-only model (no Linear/addmm):
- ✓ Forward compilation succeeds (Slang codegen works)
- ✓ Eager loss computation (`F.cross_entropy`) works after compiled forward
- ✓ Eager backward (`loss.backward()`) works, gradients flow to parameters
- No "Node X was invalid" partitioner error

**Remaining blocker**: `aten.addmm` autotune fails with `mm_tile.slang-module not found` (separate issue from partitioner). Workaround: use 1×1 Conv2d instead of Linear (already done in SimpleCNN test).

### Phase 1: Forward-Only Compile (Ready to Implement)
**Goal**: Bypass AOT autograd entirely by compiling only the forward pass

**Implementation**:
```python
@torch.compile(backend="inductor")
def forward_only(model, x):
    return model(x)

# Training loop (backward outside compiled function)
for batch in dataloader:
    x, target = batch
    
    optimizer.zero_grad()
    logits = forward_only(model, x)     # Compiled forward
    loss = F.cross_entropy(logits, target)  # Eager loss  
    loss.backward()                         # Eager backward
    optimizer.step()                        # Eager optimizer
```
    # Eager loss + backward
    loss = F.cross_entropy(logits, target)
    loss.backward()
    optimizer.step()
```

**Key insight**: Using `torch.no_grad()` prevents Dynamo from tracing backward paths, avoiding AOT autograd entirely.

**Pros**:
- Bypasses partitioner completely
- Leverages existing eager backward ops (most have Vulkan support)
- Minimal code changes

**Cons**:
- No fusion between forward and backward
- Performance penalty from eager backward (20-30% estimated)
- Need to verify all backward ops have Vulkan registration

**Risks**:
- May still trigger compilation if Dynamo traces loss computation
- Eager backward ops may be incomplete

**Validation plan**:
1. Create minimal PoC with `torch.no_grad()` wrapper
2. Test on SimpleCNN with cross_entropy
3. Measure compilation vs execution time
4. Add to regression tests if successful

### Phase 2: Incremental Slang Backward (1-2 weeks)
**Goal**: Optimize hot backward paths using Slang `[BackwardDerivative]`

**Steps**:
1. **Profile** eager backward: Identify slowest 3-5 backward ops
2. **Prioritize**: Focus on ops with high call frequency (e.g., `mm.backward`, `conv2d.backward`)
3. **Implement**: Add Slang backward kernels with `[BackwardDerivative]`
4. **Integrate**: Replace eager calls with compiled backward where beneficial

**Candidate ops** (likely hot paths):
- `aten.mm.backward` (linear layers)
- `aten.conv2d_backward` (conv layers)
- `aten.log_softmax_backward` (loss computation)
- `aten._log_softmax_backward_data` (variant)

**Pros**:
- Gradual performance improvement
- Low risk - can fall back to eager if Slang backward fails
- Leverages existing 45 backward derivatives

**Cons**:
- Doesn't solve partitioner for all cases
- Still no forward/backward fusion

### Phase 3: Full Slang Backward Evaluation (3-4 weeks)
**Goal**: Determine if full migration to Slang backward is worthwhile

**Evaluation criteria**:
1. **Performance gain**: >40% training speedup vs Phase 1
2. **Complexity**: <2 weeks implementation effort
3. **Maintenance**: Can leverage existing Slang derivatives without new infrastructure

**Implementation sketch**:
```python
@torch.compile(backend='vulkan_inductor')
def compiled_train_step(x, target):
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    # Slang-generated backward
    loss.backward()
    return loss
```

**Decision point**: After Phase 2 profiling, evaluate if the performance gap justifies the complexity.

## Implementation Details

### Phase 1: Minimal viable implementation

**File**: `backends/vulkan_slang/python/torch_vulkan/inductor/compile.py`
```python
def compile_forward_only(model, backend='vulkan_inductor'):
    """Compile forward pass only, bypassing AOT autograd."""
    @torch.compile(backend=backend, fullgraph=False)
    def forward_fn(x, *args, **kwargs):
        with torch.no_grad():
            return model(x, *args, **kwargs)
    return forward_fn
```

**Test harness**: `tests/test_inductor_regression.py:TestTrain8ConvTrainingSweep`
```python
def test_simple_cnn_training(self):
    model = SimpleCNN().to('vulkan')
    compiled_forward = compile_forward_only(model)
    
    losses = []
    for x, target in dataloader:
        logits = compiled_forward(x, target)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # Verify training works
    assert losses[-1] < losses[0]
```

### Profiling plan (Phase 2)

**Tools**:
- `torch.profiler` with CUDA/Vulkan events
- Custom backward timing hooks

**Metrics**:
- Backward op latency (ms)
- Call frequency per training step
- GPU utilization during backward

**Output**: Ranked list of backward ops by total time spent

## Trade-offs

| Approach | Performance | Complexity | Risk |
|----------|-------------|------------|------|
| **Phase 1 only** | -20-30% vs optimal | Low | Low |
| **Phase 1+2** | -5-15% vs optimal | Medium | Low |
| **Phase 1+2+3** | Optimal | High | Medium |

## Success Criteria

### Phase 1 (1 day)
- [ ] `torch.no_grad()` wrapper prevents AOT autograd
- [ ] SimpleCNN trains successfully on Vulkan
- [ ] No partitioner errors
- [ ] Performance within 2x of CPU baseline

### Phase 2 (1-2 weeks)
- [ ] Profile identifies top 3 slow backward ops
- [ ] Slang backward implemented for at least 1 op
- [ ] >20% speedup from Slang backward
- [ ] Regression tests pass

### Phase 3 (3-4 weeks, optional)
- [ ] >40% total training speedup vs Phase 1
- [ ] Implementation estimate <2 weeks
- [ ] No new infrastructure required

## Related Work

### Existing infrastructure we can leverage
- **45 Slang backward derivatives** in `python/torch_vulkan/inductor/codegen.py`
- **Custom autograd functions** pattern in `python/torch_vulkan/autograd.py`
- **Eager fallback mechanism** in `python/torch_vulkan/inductor/lowerings/`

### Upstream PyTorch context
- Dynamo's graph break logic: `torch/_dynamo/symbolic_convert.py`
- AOT autograd partitioner: `torch/_functorch/partitioners.py`
- Joint graph construction: `torch/_functorch/aot_autograd.py`

## Open Questions

1. **Does `torch.no_grad()` in compiled function prevent AOT autograd?**
   - Need to verify with PoC
   - Alternative: `torch.inference_mode()`

2. **Which eager backward ops lack Vulkan registration?**
   - Audit `backends/vulkan_slang/csrc/Register.cpp`
   - Compare against `F.cross_entropy` backward graph

3. **Can we detect forward-only compilation automatically?**
   - Analyze Dynamo graph for gradient operations
   - Fall back to Phase 1 if gradients detected

4. **Is there a cleaner way to bypass AOT?**
   - Custom Dynamo backend that skips `aot_dispatch_autograd`
   - Modify `torch.compile` flags (if supported)

## References

- TRAIN.8 blocker analysis: `docs/10-inductor-backend.md` (TRAIN.8 section)
- AOT partitioner code: `.venv/lib/python3.12/site-packages/torch/_functorch/partitioners.py`
- Slang backward examples: `python/torch_vulkan/inductor/codegen.py:L45-120`
- Custom autograd pattern: `python/torch_vulkan/autograd.py`
