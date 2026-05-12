"""PF.13 — trace FX graphs to find where requires_grad is lost."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
os.environ["TORCH_VULKAN_TRUST_INDUCTOR"] = "0"
os.environ["TORCH_LOGS"] = "graph_code"
import torch
torch._dynamo.reset()

# Check: do vulkan tensors propagate requires_grad correctly in eager?
print("=== Eager requires_grad check ===")
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
y = x.sigmoid()
z = y.sum()
print(f"x.requires_grad: {x.requires_grad}")
print(f"y.requires_grad: {y.requires_grad}, y.grad_fn: {y.grad_fn}")
print(f"z.requires_grad: {z.requires_grad}, z.grad_fn: {z.grad_fn}")
z.backward()
print(f"x.grad: {x.grad is not None} (shape {x.grad.shape if x.grad is not None else None})")

# Check: does torch.compile explain anything?
print("\n=== torch.compile breakdown ===")
torch._dynamo.reset()
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)

def f(a):
    return a.sigmoid().sum()

# Try to get the compiled graph
try:
    compiled = torch.compile(f, backend="inductor", fullgraph=True)
    out = compiled(x)
    print(f"Output: {out.cpu().item():.4f}")
    print(f"x.grad: {x.grad is not None}")
except Exception as e:
    print(f"Compile error: {e}")

# Test: is it a Dynano or AOTAutograd issue?
print("\n=== Dynamo only (aot_eager) ===")
torch._dynamo.reset()
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
try:
    compiled_aot = torch.compile(f, backend="aot_eager", fullgraph=True)
    out = compiled_aot(x)
    print(f"Output: {out.cpu().item():.4f}")
    print(f"x.grad: {x.grad is not None}")
except Exception as e:
    print(f"aot_eager error: {e}")

# Check: trace with torch._dynamo.export to see the FX graph
print("\n=== Dynamo export graph ===")
torch._dynamo.reset()
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
try:
    from torch._dynamo.testing import normalize_gm
    gm, guards = torch._dynamo.export(
        f,
        aten_graph=True,
    )(x)
    gm.print_readable()
except Exception as e:
    print(f"Export error: {e}")

# Check AOTAutograd  
print("\n=== AOTAutograd graph ===")
torch._dynamo.reset()
x = torch.randn(4, 16, device="vulkan:0", requires_grad=True)
from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler
from functorch.compile import nop
try:
    # Run AOTAutograd with nop compiler to see the joint graph
    from torch._functorch import aot_autograd
    from torch._inductor.debug import aot_logger
    
    # Use aot_function to see the backward
    import torch.nn as nn
    
    def g(x):
        return f(x)
    
    # Trace the function through AOTAutograd
    from functorch.compile import memory_efficient_fusion
    # Try aot_function
    import torch._functorch.aot_autograd as aot
    from torch._functorch.aot_autograd import aot_function
    
    def joint_graph_printer(gm, fw_inputs, bw_inputs):
        print("=== Joint graph ===")
        gm.print_readable()
        return gm
    
    traced = aot_function(f, joint_graph_printer, joint_graph_printer)
    result = traced(x)
    print(f"Traced output: {result.cpu().item():.4f}")
except Exception as e:
    print(f"AOT graph error: {e}")
    import traceback
    traceback.print_exc(limit=5)
