"""Check whether fla's RWKV7 (Triton) forward+backward works on this GPU.
Run: uv run python scripts/probe_gpu.py
"""
import torch, traceback
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available to torch -> wrong torch build? use the cu126 wheel")
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import triton; print("triton", triton.__version__)
from fla.models import RWKV7Config, RWKV7ForCausalLM
cfg = RWKV7Config(hidden_size=128, num_hidden_layers=2, head_dim=64, vocab_size=1000,
                  fuse_norm=False, fuse_cross_entropy=False)
m = RWKV7ForCausalLM(cfg).cuda().float()
x = torch.randint(0, 1000, (2, 256), device="cuda")
try:
    out = m(input_ids=x, labels=x)
    out.loss.backward()
    torch.cuda.synchronize()
    print("OK fwd+bwd, loss", out.loss.item())
except Exception:
    traceback.print_exc()
    print("\nFAILED: Triton kernels don't work here; we need the PyTorch fallback.")
