"""Multi-turn agent rollouts with a carried recurrent state.

Transcript format (identical to the SFT demos built from data.format_example):
    System: ...\n\nUser: <task>\n\n
    Assistant: ```bash\n<cmd>\n```\n\n      <- model-generated (trained); "Assistant:" is forced
    Tool: <output>\n\n                       <- environment (masked)
    ...
    Assistant: Final answer: <ans>\n\n
Only model-sampled tokens are is_model=True; they carry the sampling log-prob for the
policy-gradient update.
"""
import re
from dataclasses import dataclass, field

import torch

from .env import BASH_RE, SYSTEM_PROMPT
from .state import Session


@dataclass
class Trajectory:
    ids: list = field(default_factory=list)
    is_model: list = field(default_factory=list)
    logp: list = field(default_factory=list)      # 0.0 for non-model tokens
    reward: float = 0.0
    turns: int = 0
    family: str = ""
    level: int = 0
    ended: bool = False                            # episode ended via the env (not a budget cut-off)

    @property
    def n_model_tokens(self):
        return sum(self.is_model)


def should_stop(text):
    """A complete fenced block, or a blank line outside an open fence, ends the turn."""
    if BASH_RE.search(text):
        return True
    return "\n\n" in text and text.count("```") % 2 == 0


@torch.no_grad()
def rollout(model, tok, env, inst, temperature=1.0, max_new_per_turn=96, max_tokens=8000,
            system_prompt=SYSTEM_PROMPT) -> Trajectory:
    sess, tr = Session(model), Trajectory(family=inst.family, level=inst.level)

    def force(text):
        ids = tok.encode(text, add_special_tokens=False)
        sess.feed(ids)
        tr.ids += ids; tr.is_model += [False] * len(ids); tr.logp += [0.0] * len(ids)

    prompt = env.reset(inst)
    try:
        force(f"System: {system_prompt}\n\n")
        force(f"User: {prompt}\n\n")
        while True:
            force("Assistant:")
            gen = []
            for _ in range(max_new_per_turn):
                t, lp = sess.sample(temperature)
                gen.append(t); tr.ids.append(t); tr.is_model.append(True); tr.logp.append(lp)
                sess.feed([t])
                if should_stop(tok.decode(gen)):
                    break
            tr.turns += 1
            text = tok.decode(gen)
            res = env.step(text)
            if res.done:
                tr.reward, tr.ended = res.reward, True
                break
            if len(tr.ids) >= max_tokens:
                break                                   # token budget exhausted: reward 0
            missing = "\n\n"[: max(0, 2 - (len(text) - len(text.rstrip("\n"))))]
            force(missing + f"Tool: {res.observation}\n\n")
    finally:
        env.close()
    return tr
