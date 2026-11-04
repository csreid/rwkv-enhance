"""Multi-turn bash environment: parse the model's turn, run it in a sandbox, score at the end."""
import re
import shutil
import tempfile
from dataclasses import dataclass

from .tasks import TaskInstance, score

BASH_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)\n?```", re.S)
FINAL_RE = re.compile(r"final answer:\s*(.*)", re.S | re.I)

SYSTEM_PROMPT = (
    "You are an agent working in a Linux shell. To run a command, reply with a ```bash fenced block; "
    "you will see its output. When you know the answer or the task is done, reply with "
    "`Final answer: <answer>`."
)


@dataclass
class StepResult:
    observation: str | None   # text to feed back (None when the episode ended)
    done: bool
    reward: float = 0.0
    kind: str = ""            # "bash" | "final" | "invalid"


def parse_action(text):
    """-> ("bash", cmd) | ("final", answer) | ("invalid", None). A bash block wins over text."""
    m = BASH_RE.search(text)
    if m:
        return "bash", m.group(1)
    m = FINAL_RE.search(text)
    if m:
        return "final", m.group(1).strip()
    return "invalid", None


def format_observation(res, max_chars):
    out = res.output
    if len(out) > max_chars:
        out = out[:max_chars] + "\n[output truncated]"
    if res.timed_out:
        out += "\n[command timed out]"
    out = out.rstrip("\n") if out.strip() else "(no output)"
    if res.exit_code not in (0, None) and not res.timed_out:
        out += f"\n[exit code {res.exit_code}]"
    return out


class BashEnv:
    def __init__(self, sandbox, max_obs_chars=2000, max_turns=16):
        self.sandbox, self.max_obs_chars, self.max_turns = sandbox, max_obs_chars, max_turns
        self.workdir = None

    def reset(self, inst: TaskInstance):
        self.close()
        self.inst, self.turns = inst, 0
        self.workdir = tempfile.mkdtemp(prefix="rwkv_env_")
        from pathlib import Path
        for rel, content in inst.files.items():
            p = Path(self.workdir) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        return inst.prompt

    def step(self, text) -> StepResult:
        self.turns += 1
        kind, payload = parse_action(text)
        if kind == "final":
            return StepResult(None, True, score(self.inst, self.workdir, payload), "final")
        if self.turns >= self.max_turns:
            return StepResult(None, True, score(self.inst, self.workdir, None), kind)
        if kind == "bash":
            res = self.sandbox.run(self.workdir, payload)
            return StepResult(format_observation(res, self.max_obs_chars), False, 0.0, "bash")
        return StepResult("Error: reply with a ```bash block to run a command, or `Final answer: ...`.",
                          False, 0.0, "invalid")

    def close(self):
        if self.workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None


def run_expert(env: BashEnv, inst: TaskInstance):
    """Execute the scripted solution for real; return (messages, reward) for SFT demonstrations."""
    env.reset(inst)
    msgs = [{"role": "user", "content": inst.prompt}]
    for cmd in inst.solution:
        text = f"```bash\n{cmd}\n```"
        r = env.step(text)
        msgs += [{"role": "assistant", "content": text}, {"role": "tool", "content": r.observation}]
    final = f"Final answer: {inst.solution_answer}"
    r = env.step(final)
    msgs.append({"role": "assistant", "content": final})
    env.close()
    return msgs, r.reward
