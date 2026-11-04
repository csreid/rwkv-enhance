"""GRPO over multi-turn trajectories with truncated BPTT through the recurrent state."""
import math
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from .state import detach_cache


def group_advantages(rewards, normalize_std=True, eps=1e-4):
    r = torch.tensor(rewards, dtype=torch.float32)
    a = r - r.mean()
    if normalize_std:
        a = a / (r.std(unbiased=False) + eps) if r.std(unbiased=False) > eps else torch.zeros_like(a)
    return a.tolist()


def _chunk_logp(lm_head, h, y):
    return torch.log_softmax(lm_head(h).float(), -1).gather(-1, y[:, None])[:, 0]


def token_logprobs(lm_head, h, y, chunk=1024):
    """Per-token log-prob of y given final hidden states h, chunked + checkpointed (no [N,V] graph)."""
    parts = [checkpoint(_chunk_logp, lm_head, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
             for i in range(0, y.numel(), chunk)]
    return torch.cat(parts) if parts else h.new_zeros(0)


def trajectory_backward(model, traj, adv, norm, seg_len=1024, clip_eps=0.2):
    """Accumulate d(loss)/d(params) for one trajectory, one segment at a time.

    The recurrent state is carried across segments but detached, so memory is bounded by
    seg_len regardless of episode length. Returns (loss_sum, n_tokens, n_clipped).
    """
    dev = next(model.parameters()).device
    ids = torch.tensor(traj.ids, device=dev)
    old = torch.tensor(traj.logp, device=dev)
    ism = torch.tensor(traj.is_model, device=dev)
    T = ids.numel()
    cache, loss_sum, n_tok, n_clip = None, 0.0, 0, 0
    for s in range(0, T, seg_len):
        e = min(s + seg_len, T)
        pos = torch.arange(s, e, device=dev)
        valid = pos + 1 < T
        m = torch.zeros_like(valid)
        m[valid] = ism[pos[valid] + 1]                  # hidden at j predicts token j+1
        has = bool(m.any())
        with torch.set_grad_enabled(has):
            out = model.model(input_ids=ids[None, s:e], past_key_values=cache, use_cache=True)
            if has:
                tgt = pos[m] + 1
                logp = token_logprobs(model.lm_head, out.last_hidden_state[0][m], ids[tgt])
                ratio = torch.exp(logp - old[tgt])
                surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
                loss = -surr.sum() / norm
                loss.backward()
                loss_sum += loss.item(); n_tok += tgt.numel()
                n_clip += int(((ratio - 1).abs() > clip_eps).sum())
        cache = detach_cache(out.past_key_values)
    return loss_sum, n_tok, n_clip


@dataclass
class GRPOConfig:
    seg_len: int = 1024
    clip_eps: float = 0.2
    grad_clip: float = 1.0
    normalize_std: bool = True


def grpo_update(model, opt, groups, cfg: GRPOConfig):
    """groups: list of lists of Trajectory (one inner list per prompt). One optimizer step."""
    items = []
    for trajs in groups:
        for t, a in zip(trajs, group_advantages([t.reward for t in trajs], cfg.normalize_std)):
            if a != 0.0 and t.n_model_tokens:         # zero advantage => zero gradient, skip the compute
                items.append((t, a))
    stats = {"used": len(items), "of": sum(len(g) for g in groups)}
    if not items:
        return {**stats, "loss": 0.0, "gn": 0.0, "clipfrac": 0.0}
    norm = sum(t.n_model_tokens for t, _ in items)    # token-level mean over the whole batch
    model.train(); opt.zero_grad(set_to_none=True)
    loss, n_tok, n_clip = 0.0, 0, 0
    for t, a in items:
        l, n, c = trajectory_backward(model, t, a, norm, cfg.seg_len, cfg.clip_eps)
        loss, n_tok, n_clip = loss + l, n_tok + n, n_clip + c
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step(); opt.zero_grad(set_to_none=True); model.eval()
    return {**stats, "loss": loss, "gn": float(gn), "clipfrac": n_clip / max(n_tok, 1)}


class Curriculum:
    """Raise the task level once the success rate at the current level is high enough."""

    def __init__(self, start=1, max_level=8, window=48, up_at=0.8, current_frac=0.5):
        self.level, self.max_level, self.window = start, max_level, window
        self.up_at, self.current_frac = up_at, current_frac
        self.recent = []

    def sample_level(self, rng):
        if self.level == 1 or rng.random() < self.current_frac:
            return self.level
        return rng.randint(1, self.level - 1)          # keep rehearsing easier levels

    def record(self, level, reward):
        if level != self.level:
            return False
        self.recent = (self.recent + [reward])[-self.window:]
        if (len(self.recent) == self.window and sum(self.recent) / self.window >= self.up_at
                and self.level < self.max_level):
            self.level += 1; self.recent = []
            return True
        return False
