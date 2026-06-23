"""Diagnose relu backward failure point."""
import os, sys, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
os.environ["TORCH_VULKAN_TRUST_INDUCTOR"] = "0"
import torch
torch._dynamo.reset()

torch.manual_seed(42)

# Step 1: Try separate compile and run 
print("Step 1: Compile relu...")
x = torch.randn(4, 16, device='vulkan:0', requires_grad=True)
try:
    fn = torch.compile(lambda a: a.relu().sum(), backend='inductor', fullgraph=True)
    print("  compile OK")
    
    print("Step 2: Run forward...")
    out = fn(x)
    print(f"  forward OK: {out.cpu().item():.4f}")
    
    print("Step 3: Run backward...")
    out.backward()
    print(f"  backward OK")
    if x.grad is not None:
        print(f"  grad shape: {x.grad.shape}")
except Exception as e:
    print(f"  FAILED at: {e}")
    tb = traceback.format_exc()
    for line in tb.split('\n'):
        if 'File "' in line and ('functorch' in line or 'inductor' in line or '_aot' in line):
            pass
        elif 'RuntimeError' in line or 'ReluBackward' in line or 'returned an invalid' in line:
            print(f"  => {line.strip()}")
