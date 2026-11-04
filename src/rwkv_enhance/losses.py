import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import IGNORE


def _chunk_terms(lm_head, t_head, h, y, temperature):
    """Sum of CE (and KL to teacher) over one chunk of selected positions."""
    s = lm_head(h).float()
    ce = F.cross_entropy(s, y, reduction="sum")
    if t_head is None:
        return ce, ce.new_zeros(())
    with torch.no_grad():
        t = t_head[0](t_head[1]).float()
    T = temperature
    kl = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1),
                  log_target=True, reduction="sum") * T * T
    return ce, kl


def distill_loss(student, input_ids, labels, teacher=None, alpha=0.5, temperature=1.0,
                 chunk=1024):
    """CE on teacher-written (assistant) tokens, plus optional KL to teacher logits.

    Runs the network to final hidden states, keeps only positions whose *next* token is
    an assistant token, and applies lm_head to those in checkpointed chunks, so the
    [B, T, vocab] logits tensor is never materialized.
    """
    y = labels[:, 1:]
    mask = y != IGNORE
    h = student.model(input_ids=input_ids)[0][:, :-1][mask]
    y = y[mask]
    th = None
    if teacher is not None:
        with torch.no_grad():
            th = teacher.model(input_ids=input_ids)[0][:, :-1][mask]
    n = max(y.numel(), 1)
    ce_sum = h.new_zeros((), dtype=torch.float32)
    kl_sum = h.new_zeros((), dtype=torch.float32)
    for i in range(0, y.numel(), chunk):
        sl = slice(i, i + chunk)
        t_arg = None if th is None else (teacher.lm_head, th[sl])
        ce, kl = checkpoint(_chunk_terms, student.lm_head, t_arg, h[sl], y[sl], temperature,
                            use_reentrant=False)
        ce_sum, kl_sum = ce_sum + ce, kl_sum + kl
    ce, kl = ce_sum / n, kl_sum / n
    if teacher is None:
        return ce, {"ce": ce.item(), "tokens": y.numel()}
    return (1 - alpha) * ce + alpha * kl, {"ce": ce.item(), "kl": kl.item(), "tokens": y.numel()}
