"""Recurrent-state helpers: a stateful LM session and cache detaching.

RWKV's state is fixed-size, so a Session can consume an unbounded token stream
(many agent turns) with constant memory, never re-encoding earlier context.
"""
import torch


def detach_cache(cache):
    """Cut the autograd graph through the carried state (truncated BPTT boundary)."""
    if cache is None:
        return None
    for i in range(len(cache)):
        layer = cache.layers[i]
        layer.state = layer._map_state(lambda t: t.detach())
    return cache


class Session:
    """Incremental inference over one growing token stream (batch size 1 by design)."""

    def __init__(self, model):
        self.model = model
        self.cache = None
        self.logits = None          # next-token logits after the last fed token
        self.n_tokens = 0

    @torch.no_grad()
    def feed(self, ids):
        """Consume a list of token ids in one parallel pass; update state and logits."""
        if not len(ids):
            return
        x = torch.tensor([list(ids)], device=next(self.model.parameters()).device)
        out = self.model.model(input_ids=x, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.logits = self.model.lm_head(out.last_hidden_state[:, -1])[0].float()
        self.n_tokens += len(ids)

    @torch.no_grad()
    def sample(self, temperature=1.0):
        """Sample one token from the current logits. Returns (token_id, logprob under sampling dist)."""
        logp = torch.log_softmax(self.logits / temperature, -1)
        tok = torch.multinomial(logp.exp(), 1).item()
        return tok, logp[tok].item()
