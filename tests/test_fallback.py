import torch
from rwkv_enhance import torch_fallback as fb


def naive(r, w, k, v, a, b, scale=1.0):
    B, T, H, K = r.shape; V = v.shape[-1]
    h = torch.zeros(B, H, K, V, dtype=torch.float64); o = []
    for t in range(T):
        for_h = torch.exp(w[:, t])[..., None] * h
        for_h = for_h + b[:, t][..., None] * (a[:, t][..., None] * h).sum(2, keepdim=True)
        h = for_h + k[:, t][..., None] * v[:, t][..., None, :]
        o.append((h * (r[:, t] * scale)[..., None]).sum(2))
    return torch.stack(o, 1), h


def test_wkv_matches_naive_incl_grads():
    torch.manual_seed(0)
    B, T, H, K = 2, 70, 3, 8   # T not a multiple of chunk
    mk = lambda: torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=True)
    r, k, v, a, b = mk(), mk(), mk(), mk(), mk()
    w = (-0.6 * torch.rand(B, T, H, K, dtype=torch.float64)).requires_grad_()
    ins = (r, w, k, v, a, b)
    o1, h1 = naive(*ins, 0.7)
    o2, h2 = fb.wkv7(*ins, scale=0.7, output_final_state=True)
    assert torch.allclose(o1, o2.double(), atol=1e-9) and torch.allclose(h1, h2.double(), atol=1e-9)
    g1 = torch.autograd.grad(o1.square().sum(), ins)
    g2 = torch.autograd.grad(o2.square().sum(), ins)
    for x, y in zip(g1, g2):
        assert torch.allclose(x, y, atol=1e-8)


def test_streaming_state_equals_full():
    torch.manual_seed(1)
    B, T, H, K = 1, 40, 2, 8
    r, k, v, a, b = (torch.randn(B, T, H, K) for _ in range(5))
    w = -0.6 * torch.rand(B, T, H, K)
    full, hf = fb.wkv7(r, w, k, v, a, b, output_final_state=True)
    o1, h1 = fb.wkv7(*(x[:, :17] for x in (r, w, k, v, a, b)), output_final_state=True)
    o2, h2 = fb.wkv7(*(x[:, 17:] for x in (r, w, k, v, a, b)), initial_state=h1, output_final_state=True)
    assert torch.allclose(full, torch.cat([o1, o2], 1), atol=1e-5) and torch.allclose(hf, h2, atol=1e-5)


def test_full_model_trains_on_cpu():
    fb.patch()
    from fla.models import RWKV7Config, RWKV7ForCausalLM
    cfg = RWKV7Config(hidden_size=64, num_hidden_layers=3, head_dim=16, vocab_size=200,
                      fuse_norm=False, fuse_cross_entropy=False)
    m = RWKV7ForCausalLM(cfg)
    x = torch.randint(0, 200, (2, 70))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(25):
        loss = torch.nn.functional.cross_entropy(m(input_ids=x).logits[:, :-1].reshape(-1, 200), x[:, 1:].reshape(-1))
        loss.backward(); opt.step(); opt.zero_grad(); losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5, losses
    m.eval()  # causality: changing a later token must not change earlier logits
    with torch.no_grad():
        y = x.clone(); y[:, 50:] = (y[:, 50:] + 1) % 200
        assert torch.allclose(m(input_ids=x).logits[:, :50], m(input_ids=y).logits[:, :50], atol=1e-5)


def test_patch_disables_compile_and_sqrelu_matches():
    fb.patch()
    from fla.modules.activations import ACT2FN, sqrelu as fla_sqrelu
    assert torch._dynamo.config.disable
    assert ACT2FN["sqrelu"] is fb.sqrelu
    x = torch.randn(5, 7, requires_grad=True)
    y = fb.sqrelu(x); y.sum().backward()
    assert torch.allclose(y, torch.relu(x) ** 2)
    assert torch.allclose(x.grad, 2 * torch.relu(x))
