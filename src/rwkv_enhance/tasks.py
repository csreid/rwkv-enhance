"""Procedural, verifiable shell tasks. `level` is the difficulty / horizon knob.

Instances are plain data (JSON-serialisable) so rollout workers anywhere can receive them.
A task is solved (reward 1) iff:
  * expected_answer is set and the episode's `Final answer:` matches it (normalised), and
  * every file in expect_files exists in the work dir with exactly that content.
Each instance carries a scripted `solution` (shell commands, one per turn) used to build
warm-start demonstrations and to test the task + checker.
"""
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "grove", "haze", "iris", "jade",
         "kilo", "lumen", "maple", "nova", "onyx", "pine", "quartz", "river", "sage", "tulip",
         "umber", "vivid", "willow", "xenon", "yarrow", "zephyr"]
EXTS = [".txt", ".log", ".md", ".cfg", ".dat"]


@dataclass
class TaskInstance:
    family: str
    level: int
    prompt: str
    files: dict = field(default_factory=dict)         # relative path -> content
    expected_answer: str | None = None
    expect_files: dict = field(default_factory=dict)  # relative path -> exact content
    solution: list = field(default_factory=list)      # shell commands, one per turn
    solution_answer: str = "done"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return TaskInstance(**d)


def _norm(s):
    return " ".join(s.strip().strip("\"'`").rstrip(".").lower().split())


def score(inst: TaskInstance, workdir, final_answer):
    """1.0 if solved else 0.0. final_answer is None if the episode never produced one."""
    if final_answer is None:
        return 0.0
    if inst.expected_answer is not None and _norm(final_answer) != _norm(inst.expected_answer):
        return 0.0
    for rel, content in inst.expect_files.items():
        p = Path(workdir) / rel
        if not p.is_file() or p.read_text().strip() != content.strip():
            return 0.0
    return 1.0


def _fname(rng, used=()):
    while True:
        n = rng.choice(WORDS) + rng.choice(EXTS)
        if n not in used:
            return n


def _code(rng, n=4):
    return "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(n))


# ---- families --------------------------------------------------------------------------

def file_exists(rng, level):
    n = 2 + 2 * level
    names = []
    for _ in range(n):
        names.append(_fname(rng, names))
    present = rng.random() < 0.5
    target = rng.choice(names) if present else _fname(rng, names)
    if not present and level > 1 and rng.random() < 0.5:   # near-miss decoy
        stem, ext = names[0].rsplit(".", 1)
        target = stem + "s." + ext
        if target in names:
            target = _fname(rng, names)
    return TaskInstance("file_exists", level,
                        f"Does a file named `{target}` exist in the current directory? "
                        f"Answer yes or no.",
                        files={nm: _code(rng, 6) + "\n" for nm in names},
                        expected_answer="yes" if target in names else "no",
                        solution=["ls"], solution_answer="yes" if target in names else "no")


def read_file(rng, level):
    names = []
    for _ in range(2 + level):
        names.append(_fname(rng, names))
    target = rng.choice(names)
    secret = " ".join(rng.choice(WORDS) for _ in range(1 + level // 2)) + " " + _code(rng)
    files = {nm: _code(rng, 8) + "\n" for nm in names}
    files[target] = secret + "\n"
    return TaskInstance("read_file", level,
                        f"What is the exact content of the file `{target}`? Reply with only its content.",
                        files=files, expected_answer=secret,
                        solution=[f"cat {target}"], solution_answer=secret)


def create_file(rng, level):
    name, text = _fname(rng), " ".join(rng.choice(WORDS) for _ in range(1 + level // 2))
    return TaskInstance("create_file", level,
                        f"Create a file named `{name}` in the current directory containing exactly "
                        f"the text `{text}`. Then say done.",
                        expect_files={name: text}, expected_answer=None,
                        solution=[f"echo '{text}' > {name}"], solution_answer="done")


def count_matches(rng, level):
    n = 8 * level + 4
    lv = ["INFO", "WARN", "ERROR", "DEBUG"]
    lines = [f"{i:04d} {rng.choice(lv)} {rng.choice(WORDS)} {_code(rng)}" for i in range(n)]
    cnt = sum("ERROR" in l for l in lines)
    return TaskInstance("count_matches", level,
                        "How many lines of `app.log` contain the word ERROR? Reply with just the number.",
                        files={"app.log": "\n".join(lines) + "\n"}, expected_answer=str(cnt),
                        solution=["grep -c ERROR app.log"], solution_answer=str(cnt))


def hop_chain(rng, level):
    k = max(2, level - 1)                      # number of hops = horizon
    names = ["start.txt"]
    for _ in range(k):
        names.append(_fname(rng, names))
    code = _code(rng, 6)
    files = {names[i]: f"next: {names[i + 1]}\n" for i in range(k)}
    files[names[k]] = f"code: {code}\n"
    return TaskInstance("hop_chain", level,
                        "Read `start.txt`. Each file tells you which file to read next. "
                        "What is the code in the final file? Reply with just the code.",
                        files=files, expected_answer=code,
                        solution=[f"cat {n}" for n in names], solution_answer=code)


def collect_codes(rng, level):
    k = max(2, level)
    words = [rng.choice(WORDS) + _code(rng, 2) for _ in range(k)]
    return TaskInstance("collect_codes", level,
                        f"Read `part1.txt` through `part{k}.txt` (one at a time is fine) and reply with "
                        f"their contents joined by hyphens, in order.",
                        files={f"part{i + 1}.txt": w + "\n" for i, w in enumerate(words)},
                        expected_answer="-".join(words),
                        solution=[f"cat part{i + 1}.txt" for i in range(k)],
                        solution_answer="-".join(words))


def needle_log(rng, level):
    n = 40 * 2 ** (level - 1)
    keys = [f"{rng.choice(WORDS)}_{_code(rng, 3)}" for _ in range(n)]
    vals = [_code(rng, 6) for _ in range(n)]
    i = rng.randrange(n)
    return TaskInstance("needle_log", level,
                        f"`settings.cfg` has many `key=value` lines. What is the value of `{keys[i]}`? "
                        f"Reply with just the value.",
                        files={"settings.cfg": "\n".join(f"{k}={v}" for k, v in zip(keys, vals)) + "\n"},
                        expected_answer=vals[i],
                        solution=[f"grep '^{keys[i]}=' settings.cfg"], solution_answer=vals[i])


# (family fn, minimum level)
FAMILIES = [(file_exists, 1), (read_file, 1), (create_file, 3), (count_matches, 3),
            (hop_chain, 4), (collect_codes, 4), (needle_log, 5)]


def sample_task(seed, level):
    rng = random.Random(f"{seed}:{level}")
    avail = [f for f, lo in FAMILIES if level >= lo]
    return rng.choice(avail)(rng, level)
