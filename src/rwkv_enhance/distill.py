"""python -m rwkv_enhance.distill configs/distill_nemotron.yaml"""
import math, os, sys, time, yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import fla  # noqa: F401  registers RWKV7 with transformers

from .data import NemotronStream, collate
from .losses import distill_loss


def lr_at(step, c):
    if step < c["warmup_steps"]:
        return c["lr"] * (step + 1) / c["warmup_steps"]
    p = (step - c["warmup_steps"]) / max(1, c["max_steps"] - c["warmup_steps"])
    return c["lr"] * (c["min_lr_ratio"] + (1 - c["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(p, 1))))


def main(path):
    c = yaml.safe_load(open(path))
    torch.manual_seed(c["seed"])
    dev, dt = c.get("device", "cuda"), getattr(torch, c["dtype"])
    kw = {}
    if c.get("fallback_kernels"):  # GPUs Triton can't target (e.g. Pascal)
        from .torch_fallback import patch
        patch()
        kw = dict(fuse_norm=False, fuse_cross_entropy=False, fuse_linear_cross_entropy=False)
    tok = AutoTokenizer.from_pretrained(c["student"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(c["student"], torch_dtype=dt, trust_remote_code=True, **kw).to(dev)
    model.train()
    teacher = None
    if c["teacher"]:
        teacher = AutoModelForCausalLM.from_pretrained(c["teacher"], torch_dtype=dt, trust_remote_code=True, **kw).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    ds = NemotronStream(tok, c["dataset"], c["splits"], c["include_reasoning"],
                        c["max_seq_len"], c["shuffle_buffer"], c["seed"])
    dl = DataLoader(ds, batch_size=c["batch_size"], collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"], betas=(0.9, 0.99))
    os.makedirs(c["output_dir"], exist_ok=True)

    it, step, t0 = iter(dl), 0, time.time()
    while step < c["max_steps"]:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, c)
        stats = {}
        for _ in range(c["grad_accum"]):
            b = {k: v.to(dev) for k, v in next(it).items()}
            logits = model(input_ids=b["input_ids"]).logits
            tl = None
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(input_ids=b["input_ids"]).logits
            loss, stats = distill_loss(logits, b["labels"], tl, c["kd_alpha"], c["kd_temperature"])
            (loss / c["grad_accum"]).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % c["log_every"] == 0:
            print(f"step {step} lr {lr_at(step, c):.2e} gn {gn:.2f} {stats} {time.time()-t0:.0f}s", flush=True)
        if step % c["save_every"] == 0 or step == c["max_steps"]:
            d = f"{c['output_dir']}/step{step}"
            model.save_pretrained(d); tok.save_pretrained(d)


if __name__ == "__main__":
    main(sys.argv[1])
