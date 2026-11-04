"""Pure-PyTorch replacements for fla's Triton RWKV7 kernels.

For GPUs Triton can't target (e.g. Pascal / sm_61) and for CPU smoke tests.
Training-oriented: no varlen (cu_seqlens), caches only minimally supported.

    from rwkv_enhance.torch_fallback import patch; patch()
    # and load the model with fuse_norm=False, fuse_cross_entropy=False
"""
import torch
from torch.utils.checkpoint import checkpoint

CHUNK = 32  # timesteps per checkpointed segment (memory/recompute tradeoff)


def _wkv_segment(h, r, w, k, v, a, b):
    """h: [B,H,K,V]; r,w,k,a,b: [B,t,H,K]; v: [B,t,H,V]. Returns (o [B,t,H,V], h)."""
    outs = []
    for t in range(r.shape[1]):
        # h = exp(w) * h + b (a^T h) + k v^T
        ah = torch.einsum("bhk,bhkv->bhv", a[:, t], h)
        h = torch.exp(w[:, t]).unsqueeze(-1) * h + b[:, t].unsqueeze(-1) * ah.unsqueeze(-2)
        h = h + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
        outs.append(torch.einsum("bhk,bhkv->bhv", r[:, t], h))
    return torch.stack(outs, 1), h


def wkv7(r, w, k, v, a, b, scale=1.0, initial_state=None, output_final_state=False,
         cu_seqlens=None, chunk=CHUNK):
    assert cu_seqlens is None, "varlen not supported in torch fallback"
    B, T, H, K = r.shape
    V = v.shape[-1]
    dt = r.dtype
    ft = torch.promote_types(dt, torch.float32)  # compute in >= fp32
    r, w, k, v, a, b = (x.to(ft) for x in (r * scale, w, k, v, a, b))
    h = (torch.zeros(B, H, K, V, device=r.device, dtype=ft) if initial_state is None
         else initial_state.to(ft))
    needs_grad = torch.is_grad_enabled() and any(
        x.requires_grad for x in (r, w, k, v, a, b, h))
    outs = []
    for s in range(0, T, chunk):
        sl = slice(s, s + chunk)
        args = (h, r[:, sl], w[:, sl], k[:, sl], v[:, sl], a[:, sl], b[:, sl])
        if needs_grad:
            o, h = checkpoint(_wkv_segment, *args, use_reentrant=False)
        else:
            o, h = _wkv_segment(*args)
        outs.append(o)
    o = torch.cat(outs, 1).to(dt)
    return o, (h if output_final_state else None)


def chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=None, output_final_state=False,
                cu_seqlens=None, **_ignored):
    return wkv7(r, w, k, v, a, b, scale, initial_state, output_final_state, cu_seqlens)


def fused_mul_recurrent_rwkv7(r, w, k, v, kk, a, scale=1.0, initial_state=None,
                              output_final_state=False, cu_seqlens=None, **_ignored):
    return wkv7(r, w, k, v, -kk, kk * a, scale, initial_state, output_final_state, cu_seqlens)


def token_shift(x, cu_seqlens=None, cache=None, output_cache=False, chunk_indices=None):
    assert cu_seqlens is None, "varlen not supported in torch fallback"
    if cache is None:
        first = torch.zeros_like(x[:, :1])
    else:
        first = cache.reshape(x.shape[0], 1, x.shape[-1]).to(x.dtype)
    delta = torch.cat([first, x[:, :-1]], 1) - x
    return (delta, x[:, -1]) if output_cache else delta


def fused_addcmul_rwkv7(hidden_states, delta, xr, xw, xk, xv, xa, xg=None):
    f = lambda m: torch.addcmul(hidden_states, delta, m)
    return f(xr), f(xw), f(xk), f(xv), f(xa), (f(xg) if xg is not None else None)


def fused_k_rwkv7(k, a, ka, cu_seqlens=None, cu_seqlens_cpu=None):
    return k.addcmul(k * (a - 1), ka)


def gate_output_correction(o, r, k, r_k, v, g):
    corr = ((r * k * r_k.unsqueeze(0).unsqueeze(0)).sum(-1, keepdim=True) * v).view(o.shape)
    return (o + corr) * g


def sqrelu(x):
    return torch.relu(x).square()


def patch():
    """Swap fla's Triton-backed names in the RWKV7 layer/model modules."""
    # fla wraps many small ops in @torch.compile (inductor needs Triton). Make them eager.
    torch._dynamo.config.disable = True
    from fla.modules.activations import ACT2FN
    ACT2FN["sqrelu"] = sqrelu
    import fla.layers.rwkv7 as layer
    import fla.models.rwkv7.modeling_rwkv7 as model
    for name in ("chunk_rwkv7", "fused_mul_recurrent_rwkv7", "token_shift",
                 "fused_addcmul_rwkv7", "fused_k_rwkv7", "gate_output_correction"):
        setattr(layer, name, globals()[name]) if hasattr(layer, name) else None
    model.token_shift = token_shift
