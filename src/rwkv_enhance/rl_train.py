"""python -m rwkv_enhance.rl_train configs/rl_agent.yaml"""
import json, os, random, sys, time
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import fla  # noqa: F401

from .env import BashEnv
from .grpo import Curriculum, GRPOConfig, grpo_update
from .rollout import rollout
from .sandbox import BwrapSandbox
from .tasks import sample_task


def main(path):
    c = yaml.safe_load(open(path))
    rng = random.Random(c["seed"]); torch.manual_seed(c["seed"])
    dev, dt = c.get("device", "cuda"), getattr(torch, c["dtype"])
    kw = {}
    if c.get("fallback_kernels") or (dev.startswith("cuda") and torch.cuda.get_device_capability(0) < (7, 0)):
        from .torch_fallback import patch
        patch()
        kw = dict(fuse_norm=False, fuse_cross_entropy=False, fuse_linear_cross_entropy=False)
    tok = AutoTokenizer.from_pretrained(c["student"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(c["student"], torch_dtype=dt, trust_remote_code=True, **kw).to(dev).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"], betas=(0.9, 0.99))
    cfg = GRPOConfig(c["seg_len"], c["clip_eps"], c["grad_clip"], c["normalize_std"])
    cur = Curriculum(c["start_level"], c["max_level"], c["curriculum_window"], c["level_up_at"], c["current_level_frac"])
    env = BashEnv(BwrapSandbox(timeout=c["cmd_timeout"]), c["max_obs_chars"], c["max_turns"])
    os.makedirs(c["output_dir"], exist_ok=True)
    logf = open(f"{c['output_dir']}/log.jsonl", "a")

    t0 = time.time()
    for it in range(1, c["iterations"] + 1):
        groups, toks, turns = [], 0, 0
        for _ in range(c["prompts_per_iter"]):
            lvl = cur.sample_level(rng)
            inst = sample_task(rng.randrange(1 << 30), lvl)
            trajs = [rollout(model, tok, env, inst, c["temperature"], c["max_new_per_turn"], c["max_tokens"])
                     for _ in range(c["group_size"])]
            for t in trajs:
                leveled_up = cur.record(lvl, t.reward)
                toks += len(t.ids); turns += t.turns
            groups.append(trajs)
        st = grpo_update(model, opt, groups, cfg)
        allt = [t for g in groups for t in g]
        row = {"it": it, "level": cur.level, "reward": sum(t.reward for t in allt) / len(allt),
               "turns": turns / len(allt), "tokens": toks / len(allt), **st, "t": round(time.time() - t0)}
        print(json.dumps(row), flush=True); logf.write(json.dumps(row) + "\n"); logf.flush()
        if it % c["save_every"] == 0 or it == c["iterations"]:
            d = f"{c['output_dir']}/it{it}"
            model.save_pretrained(d); tok.save_pretrained(d)


if __name__ == "__main__":
    main(sys.argv[1])
