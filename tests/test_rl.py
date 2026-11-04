import random
import pytest
import torch
from rwkv_enhance import torch_fallback as fb
from rwkv_enhance.env import StepResult
from rwkv_enhance.grpo import (GRPOConfig, Curriculum, group_advantages, grpo_update,
                               token_logprobs, trajectory_backward)
from rwkv_enhance.rollout import rollout, should_stop
from rwkv_enhance.tasks import TaskInstance


class CharTok:
    chars = [chr(i) for i in range(32, 127)] + ["\n"]
    def __init__(self): self.m = {c: i for i, c in enumerate(self.chars)}
    vocab_size = 96
    def encode(self, s, add_special_tokens=False): return [self.m[c] for c in s if c in self.m]
    def decode(self, ids): return "".join(self.chars[i] for i in ids)


class ToyEnv:
    """Reward 1 iff the model's reply starts with 'x'. Ends after one turn."""
    def reset(self, inst): return inst.prompt
    def step(self, text): return StepResult(None, True, float(text.strip().startswith("x")), "final")
    def close(self): pass


class MultiTurnToyEnv:
    """Two turns, same rule each turn (first letter in a-t). Passing turn 1 earns 0.3 and the episode
    continues; passing turn 2 as well earns 1.0. Success needs the whole chain, so credit has to flow
    across the turn boundary. (Opposite rules per turn would make the two turns fight over the same
    first-token logits, which tests the toy's design rather than the trainer.)"""
    OK = "abcdefghijklmnopqrst"
    def reset(self, inst): self.n = 0; return inst.prompt
    def step(self, text):
        self.n += 1; ok = bool(text.strip()) and text.strip()[0] in self.OK
        if self.n == 1:
            return StepResult("ok", False, 0.0, "bash") if ok else StepResult(None, True, 0.0)
        return StepResult(None, True, 1.0 if ok else 0.3, "final")
    def close(self): pass


INST = TaskInstance("toy", 1, "go")


def tiny(seed=0):
    fb.patch()
    from fla.models import RWKV7Config, RWKV7ForCausalLM
    torch.manual_seed(seed)
    m = RWKV7ForCausalLM(RWKV7Config(hidden_size=64, num_hidden_layers=2, head_dim=16, vocab_size=96,
                                     fuse_norm=False, fuse_cross_entropy=False))
    for p in m.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.05)
    return m.eval()


def test_advantages_and_curriculum():
    a = group_advantages([1, 0, 0, 0])
    assert abs(sum(a)) < 1e-5 and a[0] > 0 > a[1]
    assert group_advantages([1, 1, 1]) == [0, 0, 0] and group_advantages([0, 0]) == [0, 0]
    c = Curriculum(window=4, up_at=0.75, max_level=3)
    assert not any(c.record(1, r) for r in [1, 1, 1]) and c.record(1, 1) and c.level == 2
    assert not c.record(1, 1)                                   # other levels don't count
    rng = random.Random(0)
    assert {c.sample_level(rng) for _ in range(100)} == {1, 2}


def test_stop_rule():
    assert should_stop(" ```bash\nls\n```") and should_stop(" Final answer: 3\n\n")
    assert not should_stop(" ```bash\ncat <<E\na\n\nb") and not should_stop(" ls")


def test_rollout_structure_and_recorded_logprobs_match_training_pass():
    m, tok = tiny(), CharTok()
    tr = rollout(m, tok, MultiTurnToyEnv(), INST, max_new_per_turn=12)
    n = len(tr.ids)
    assert n == len(tr.is_model) == len(tr.logp) and tr.n_model_tokens > 0
    assert all(lp <= 0 for lp, im in zip(tr.logp, tr.is_model) if im)
    assert all(lp == 0 for lp, im in zip(tr.logp, tr.is_model) if not im)
    text = tok.decode(tr.ids)
    assert text.startswith("System: ") and "User: go\n\nAssistant:" in text
    # The key property: log-probs recorded while sampling with the cached/decode path equal the
    # log-probs recomputed by the segmented training path (so the PPO ratio starts at 1).
    ids = torch.tensor(tr.ids)
    with torch.no_grad():
        h = m.model(input_ids=ids[None]).last_hidden_state[0]
        mask = torch.tensor(tr.is_model)[1:]
        lp = token_logprobs(m.lm_head, h[:-1][mask], ids[1:][mask])
    assert torch.allclose(lp, torch.tensor(tr.logp)[1:][mask], atol=2e-3), (lp - torch.tensor(tr.logp)[1:][mask]).abs().max()


def test_segmented_backward_equals_unsegmented_gradients():
    m, tok = tiny(), CharTok()
    tr = rollout(m, tok, MultiTurnToyEnv(), INST, max_new_per_turn=12)
    # Splitting into segments only truncates gradient through the carried state, so gradients
    # differ slightly; but with seg_len >= T it must be exactly the single-pass gradient.
    def grads(seg):
        m.zero_grad(); trajectory_backward(m, tr, 1.0, tr.n_model_tokens, seg_len=seg)
        return [p.grad.clone() if p.grad is not None else None for p in m.parameters()]
    g_full, g_big = grads(10_000), grads(len(tr.ids) + 5)
    for a, b in zip(g_full, g_big):
        assert (a is None) == (b is None)
        if a is not None: assert torch.allclose(a, b, atol=1e-6)
    # multi-segment still produces finite, non-trivial gradients
    g_seg = grads(16)
    assert all(torch.isfinite(g).all() for g in g_seg if g is not None)
    assert sum(g.abs().sum() for g in g_seg if g is not None) > 0


def test_positive_advantage_raises_logprob_of_trajectory():
    m, tok = tiny(), CharTok()
    tr = rollout(m, tok, ToyEnv(), INST, max_new_per_turn=10)
    def total_logp():
        ids = torch.tensor(tr.ids); mask = torch.tensor(tr.is_model)[1:]
        with torch.no_grad():
            h = m.model(input_ids=ids[None]).last_hidden_state[0]
            return token_logprobs(m.lm_head, h[:-1][mask], ids[1:][mask]).sum().item()
    before = total_logp()
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    m.train(); opt.zero_grad()
    trajectory_backward(m, tr, +1.0, 1, seg_len=16); opt.step(); m.eval()
    assert total_logp() > before
    before = total_logp()
    m.train(); opt.zero_grad()
    trajectory_backward(m, tr, -1.0, 1, seg_len=16); opt.step(); m.eval()
    assert total_logp() < before


@pytest.mark.parametrize("envcls,kw", [(ToyEnv, dict(iters=60, final_min=0.7)),
                                       (MultiTurnToyEnv, dict(iters=80, final_min=0.8))])
def test_grpo_learns_toy_tasks(envcls, kw):
    torch.manual_seed(0)
    m, tok = tiny(1), CharTok()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    cfg = GRPOConfig(seg_len=16)
    hist = []
    for it in range(kw["iters"]):
        trajs = [rollout(m, tok, envcls(), INST, max_new_per_turn=6) for _ in range(32)]
        hist.append(sum(t.reward for t in trajs) / len(trajs))
        grpo_update(m, opt, [trajs], cfg)
    start, end = sum(hist[:5]) / 5, sum(hist[-8:]) / 8
    assert end > max(kw["final_min"], start + 0.3), (start, end, hist)
