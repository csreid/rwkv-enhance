"""Build warm-start SFT demos by running the scripted expert in the real sandbox.
    uv run python scripts/make_demos.py demos.jsonl --n 3000 --max-level 8
Episodes are written as {"messages": [system, user, assistant, tool, ..., assistant]}.
"""
import argparse, json, random
from rwkv_enhance.env import BashEnv, SYSTEM_PROMPT, run_expert
from rwkv_enhance.sandbox import BwrapSandbox
from rwkv_enhance.tasks import sample_task

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--max-level", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng, env = random.Random(a.seed), BashEnv(BwrapSandbox())
with open(a.out, "w") as f:
    for i in range(a.n):
        inst = sample_task(rng.randrange(1 << 30), rng.randint(1, a.max_level))
        msgs, reward = run_expert(env, inst)
        assert reward == 1.0, inst                       # never train on a failed demonstration
        f.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + msgs}) + "\n")
print(f"wrote {a.n} demos to {a.out}")
