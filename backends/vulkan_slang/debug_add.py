"""Debug x+y backward storage issue."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
os.environ["TORCH_VULKAN_TRUST_INDUCTOR"] = "0"
os.environ["TORCH_LOGS"] = "output_code"
import torch
torch._dynamo.reset()

# Test x + y
a = torch.randn(4, 16, device='vulkan:0', requires_grad=True)
b = torch.randn(4, 16, device='vulkan:0', requires_grad=True)

try:
    fn = torch.compile(lambda x, y: (x + y).sum(), backend='inductor', fullgraph=True)
    out = fn(a, b)
    print('\n=== Forward output ===')
    print(f'out: {out.cpu().item():.4f}')
    print('\n=== Backward ===')
    out.backward()
    print('backward OK')
    if a.grad is not None:
        print(f'a.grad shape: {a.grad.shape}')
    if b.grad is not None:
        print(f'b.grad shape: {b.grad.shape}')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc(limit=5)
