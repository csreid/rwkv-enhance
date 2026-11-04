import pytest
import torch
from rwkv_enhance import torch_fallback as fb


def realistic(B, T, H, K, dtype=torch.float32, grad=False):
    """Inputs shaped like the real layer's: a=-kk, b=kk*gate, ||kk||=1, w in [-0.6065, 0]."""
    kk = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=dtype), dim=-1)
    ts = [torch.randn(B, T, H, K, dtype=dtype), -0.6065 * torch.sigmoid(torch.randn(B, T, H, K, dtype=dtype)),
          0.3 * torch.randn(B, T, H, K, dtype=dtype), torch.randn(B, T, H, K, dtype=dtype),
          -kk, kk * torch.rand(B, T, H, K, dtype=dtype)]
    if grad:
        ts = [x.detach().requires_grad_() for x in ts]
    return ts


def naive(r, w, k, v, a, b, scale=1.0):
    B, T, H, K = r.shape; V = v.shape[-1]
    h = torch.zeros(B, H, K, V, dtype=torch.float64); o = []
    for t in range(T):
        for_h = torch.exp(w[:, t])[..., None] * h
        for_h = for_h + b[:, t][..., None] * (a[:, t][..., None] * h).sum(2, keepdim=True)
        h = for_h + k[:, t][..., None] * v[:, t][..., None, :]
        o.append((h * (r[:, t] * scale)[..., None]).sum(2))
    return torch.stack(o, 1), h


@pytest.mark.parametrize('chunk', [1, 7, 32, 64])
def test_wkv_matches_naive_incl_grads(chunk):
    torch.manual_seed(0)
    B, T, H, K = 2, 70, 3, 8   # T not a multiple of chunk
    ins = tuple(realistic(B, T, H, K, torch.float64, grad=True))
    o1, h1 = naive(*ins, 0.7)
    o2, h2 = fb.wkv7(*ins, scale=0.7, output_final_state=True, chunk=chunk)
    assert torch.allclose(o1, o2.double(), atol=1e-8, rtol=1e-6) and torch.allclose(h1, h2.double(), atol=1e-8, rtol=1e-6)
    g1 = torch.autograd.grad(o1.square().sum(), ins)
    g2 = torch.autograd.grad(o2.square().sum(), ins)
    for x, y in zip(g1, g2):
        assert torch.allclose(x, y, atol=1e-6, rtol=1e-5)


def test_streaming_state_equals_full():
    torch.manual_seed(1)
    B, T, H, K = 1, 40, 2, 8
    r, w, k, v, a, b = realistic(B, T, H, K)
    full, hf = fb.wkv7(r, w, k, v, a, b, output_final_state=True, chunk=8)
    o1, h1 = fb.wkv7(*(x[:, :17] for x in (r, w, k, v, a, b)), output_final_state=True, chunk=8)
    o2, h2 = fb.wkv7(*(x[:, 17:] for x in (r, w, k, v, a, b)), initial_state=h1, output_final_state=True, chunk=8)
    assert torch.allclose(full, torch.cat([o1, o2], 1), atol=1e-4, rtol=1e-4) and torch.allclose(hf, h2, atol=1e-4, rtol=1e-4)


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


def test_loop_and_chunked_agree_fp32_realistic_scale():
    torch.manual_seed(2)
    B, T, H, K = 2, 256, 4, 64
    kk = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
    a_gate = torch.rand(B, T, H, K)
    args = (torch.randn(B, T, H, K), -0.6065 * torch.sigmoid(torch.randn(B, T, H, K)),
            0.3 * torch.randn(B, T, H, K), torch.randn(B, T, H, K), -kk, kk * a_gate)
    o1, _ = fb.wkv7_loop(*args)
    o2, _ = fb.wkv7(*args)
    assert torch.allclose(o1, o2, atol=2e-4, rtol=1e-3), (o1 - o2).abs().max()


def _tiny(seed):
    fb.patch()
    from fla.models import RWKV7Config, RWKV7ForCausalLM
    torch.manual_seed(seed)
    m = RWKV7ForCausalLM(RWKV7Config(hidden_size=64, num_hidden_layers=2, head_dim=16, vocab_size=300,
                                     fuse_norm=False, fuse_cross_entropy=False))
    # o_proj / value are zero-initialised; randomise so every parameter gets gradient
    for p in m.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.05)
    return m


@pytest.mark.parametrize("with_teacher", [False, True])
def test_assistant_only_loss_matches_full_logits(with_teacher):
    from rwkv_enhance.losses import distill_loss
    import torch.nn.functional as F
    student, teacher = _tiny(0), (_tiny(1).eval() if with_teacher else None)
    ids = torch.randint(0, 300, (3, 40))
    labels = ids.clone()
    labels[:, :11] = -100; labels[0, 30:] = -100; labels[2, 5:20] = -100   # mixed masks
    alpha, T = 0.4, 2.0

    # reference: full [B,T,V] logits, as the old implementation did
    logits = student(input_ids=ids).logits[:, :-1].float()
    y = labels[:, 1:]; m = y != -100
    ce = F.cross_entropy(logits.reshape(-1, 300), y.reshape(-1), ignore_index=-100)
    ref = ce
    if with_teacher:
        with torch.no_grad():
            tl = teacher(input_ids=ids).logits[:, :-1].float()
        kl = F.kl_div(F.log_softmax(logits[m] / T, -1), F.log_softmax(tl[m] / T, -1),
                      log_target=True, reduction="batchmean") * T * T
        ref = (1 - alpha) * ce + alpha * kl
    g_ref = torch.autograd.grad(ref, list(student.parameters()), allow_unused=True)

    got, stats = distill_loss(student, ids, labels, teacher, alpha, T, chunk=16)  # several chunks
    g_got = torch.autograd.grad(got, list(student.parameters()), allow_unused=True)
    assert stats["tokens"] == int(m.sum())
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), (ref.item(), got.item())
    for a, b in zip(g_ref, g_got):
        if a is None:
            assert b is None
        else:
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_session_incremental_matches_full_forward():
    from rwkv_enhance.state import Session
    m = _tiny(0).eval()
    ids = torch.randint(0, 300, (60,)).tolist()
    with torch.no_grad():
        full = m(input_ids=torch.tensor([ids])).logits[0]
    s = Session(m)
    s.feed(ids[:25]);                  assert torch.allclose(s.logits, full[24], atol=1e-4)   # parallel prefill
    for i in range(25, 40):            # one token at a time (decode path)
        s.feed([ids[i]]);              assert torch.allclose(s.logits, full[i], atol=1e-4), i
    s.feed(ids[40:]);                  assert torch.allclose(s.logits, full[59], atol=1e-4)   # mixed chunk
    assert s.n_tokens == 60
    tok, lp = s.sample(1.0)
    assert 0 <= tok < 300 and lp <= 0


def test_segmented_forward_with_detached_state_matches_full():
    from rwkv_enhance.state import detach_cache
    m = _tiny(0)
    ids = torch.randint(0, 300, (1, 90))
    with torch.no_grad():
        full = m.model(input_ids=ids).last_hidden_state
    cache, outs = None, []
    for s in range(0, 90, 32):                       # segments 32,32,26
        o = m.model(input_ids=ids[:, s:s + 32], past_key_values=cache, use_cache=True)
        outs.append(o.last_hidden_state)
        cache = detach_cache(o.past_key_values)
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-4)
    # gradient reaches parameters from a late segment, but not through the detached state
    outs[-1].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
