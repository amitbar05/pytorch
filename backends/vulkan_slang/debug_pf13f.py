"""Compare backends for relu."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
os.environ["TORCH_VULKAN_TRUST_INDUCTOR"] = "0"
import torch
torch._dynamo.reset()

torch.manual_seed(42)

print("=== Eager (baseline) ===")
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
x_cpu = x.detach().cpu().requires_grad_()
y = x.relu().sum()
y.backward()
ref_cpu = x_cpu.relu().sum()
ref_cpu.backward()
diff_ref = (ref_cpu.grad - x.grad.cpu()).abs().max().item()
print(f"  x.grad (eager): shape={x.grad.shape}, diff vs CPU: {diff_ref:.6f}")

print("\n=== aot_eager ===")
torch._dynamo.reset()
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
try:
    fn = torch.compile(lambda a: a.relu().sum(), backend="aot_eager", fullgraph=True)
    out = fn(x)
    out.backward()
    if x.grad is not None:
        diff = (ref_cpu.grad - x.grad.cpu()).abs().max().item()
        print(f"  x.grad: shape={x.grad.shape}, diff vs CPU: {diff:.6f}")
    else:
        print(f"  x.grad: None")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n=== inductor (with TORCH_LOGS) ===")
# Try without slang (just compile AOTAutograd part)
torch._dynamo.reset()
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
try:
    fn = torch.compile(lambda a: a.relu().sum(), backend="inductor", fullgraph=True)
    # Don't call - just compile
    print("  Compiled successfully (forward only)")
except Exception as e:
    print(f"  Compile FAILED: {e}")

# Also test relu via F.relu
print("\n=== F.relu (aot_eager) ===")
import torch.nn.functional as F
torch._dynamo.reset()
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
try:
    fn = torch.compile(lambda a: F.relu(a).sum(), backend="aot_eager", fullgraph=True)
    out = fn(x)
    out.backward()
    if x.grad is not None:
        diff = (ref_cpu.grad - x.grad.cpu()).abs().max().item()
        print(f"  x.grad: shape={x.grad.shape}, diff vs CPU: {diff:.6f}")
    else:
        print(f"  x.grad: None")
except Exception as e:
    print(f"  FAILED: {e}")

# Check what relu decomposes to
print("\n=== ReLU decomposition under compile ===")
with torch.no_grad():
    x = torch.randn(4, 16, device="vulkan:0")
    try:
        # Export the graph to see the decomposition
        gm, guards = torch._dynamo.export(
            lambda a: torch.relu(a),
            aten_graph=True,
        )(x)
        print("  Dynamo export of relu:")
        for node in gm.graph.nodes:
            print(f"    {node.target} {node.args} {node.kwargs}")
    except Exception as e:
        print(f"  Export failed: {e}")
