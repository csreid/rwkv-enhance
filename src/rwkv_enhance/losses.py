import torch
import torch.nn.functional as F
from .data import IGNORE


def distill_loss(student_logits, labels, teacher_logits=None, alpha=0.5, temperature=1.0):
    """CE on teacher-written tokens (+ optional token-level KL to teacher logits)."""
    s = student_logits[:, :-1].float()
    y = labels[:, 1:]
    mask = y != IGNORE
    ce = F.cross_entropy(s.reshape(-1, s.size(-1)), y.reshape(-1), ignore_index=IGNORE)
    if teacher_logits is None:
        return ce, {"ce": ce.item()}
    t = teacher_logits[:, :-1].float()
    T = temperature
    kl = F.kl_div(
        F.log_softmax(s[mask] / T, -1), F.log_softmax(t[mask] / T, -1),
        log_target=True, reduction="batchmean",
    ) * T * T
    return (1 - alpha) * ce + alpha * kl, {"ce": ce.item(), "kl": kl.item()}
