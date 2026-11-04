"""Streaming Nemotron post-training data -> tokenized, assistant-masked batches."""
import random
import torch
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset

IGNORE = -100


def format_example(messages, tokenizer, include_reasoning=True, max_len=2048):
    """RWKV 'User:/Assistant:' format. Loss only on assistant turns."""
    ids, labels = [], []
    for m in messages:
        role, content = m["role"], (m.get("content") or "").strip()
        if role == "assistant":
            head = tokenizer.encode("Assistant:", add_special_tokens=False)
            body = tokenizer.encode(" " + content + "\n\n", add_special_tokens=False)
            ids += head + body
            labels += [IGNORE] * len(head) + body
        else:
            name = {"user": "User", "system": "System"}.get(role, role.capitalize())
            t = tokenizer.encode(f"{name}: {content}\n\n", add_special_tokens=False)
            ids += t
            labels += [IGNORE] * len(t)
    ids, labels = ids[:max_len], labels[:max_len]
    if all(l == IGNORE for l in labels):
        return None
    return ids, labels


class NemotronStream(IterableDataset):
    def __init__(self, tokenizer, dataset, splits, include_reasoning, max_len, shuffle_buffer, seed):
        self.tok, self.max_len, self.inc = tokenizer, max_len, include_reasoning
        parts = [load_dataset(dataset, split=s, streaming=True) for s in splits]
        ds = interleave_datasets(parts, seed=seed) if len(parts) > 1 else parts[0]
        self.ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    def __iter__(self):
        for ex in self.ds:
            msgs = ex["messages"]
            # Nemotron v1 marks reasoning on/off per example; thinking text is
            # already inline in the assistant content when present.
            out = format_example(msgs, self.tok, self.inc, self.max_len)
            if out is not None:
                yield out


def collate(batch, pad_id=0):
    """Right-padding: harmless for a causal/recurrent model, labels masked."""
    n = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, : len(i)] = torch.tensor(i)
        lab[k, : len(l)] = torch.tensor(l)
    return {"input_ids": ids, "labels": lab}
