import json, shutil, time
import pytest
from rwkv_enhance.sandbox import BwrapSandbox, UnsafeLocalSandbox
from rwkv_enhance.tasks import sample_task, TaskInstance
from rwkv_enhance.env import BashEnv, run_expert, parse_action

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")


@pytest.fixture
def sb():
    return BwrapSandbox(timeout=5)


@needs_bwrap
def test_sandbox_isolation(sb, tmp_path):
    (tmp_path / "f.txt").write_text("hi")
    run = lambda c: sb.run(str(tmp_path), c)
    assert run("cat f.txt").output == "hi" and run("cat f.txt").exit_code == 0
    run("echo x > new.txt"); assert (tmp_path / "new.txt").exists()            # work dir is writable
    assert run("touch /usr/evil").exit_code != 0                                # system is read-only
    assert run("ls /home").exit_code != 0                                       # host home not mounted
    assert "csreid" not in run("echo $HOME; ls ~; env").output                 # no host identity
    assert run("exec 3<>/dev/tcp/1.1.1.1/80").exit_code != 0                    # no network
    assert run("ls /work/..; ls /").exit_code == 0 and "work" in run("ls /").output
    run("touch /tmp/x"); assert not (tmp_path / "x").exists()


@needs_bwrap
def test_sandbox_limits(sb, tmp_path):
    t = time.time(); r = sb.run(str(tmp_path), "sleep 30")
    assert r.timed_out and time.time() - t < 8
    r = BwrapSandbox(timeout=5, max_output_bytes=1000).run(str(tmp_path), "yes")
    assert r.truncated and len(r.output) == 1000
    assert sb.run(str(tmp_path), "exit 3").exit_code == 3


def test_unsafe_sandbox_refuses_without_optin():
    with pytest.raises(RuntimeError):
        UnsafeLocalSandbox()


def test_parse_action():
    assert parse_action("```bash\nls -la\n```") == ("bash", "ls -la")
    assert parse_action(" ```bash\nls\n```\n\n") == ("bash", "ls")
    assert parse_action("Final answer: 42") == ("final", "42")
    assert parse_action("hello") == ("invalid", None)
    assert parse_action("```bash\ncat a\n```\nFinal answer: 1")[0] == "bash"


@needs_bwrap
@pytest.mark.parametrize("level", range(1, 9))
def test_expert_solves_every_family_in_sandbox(sb, level):
    env = BashEnv(sb)
    for seed in range(12):
        inst = sample_task(seed, level)
        msgs, reward = run_expert(env, inst)
        assert reward == 1.0, (inst.family, level, seed, msgs)


@needs_bwrap
def test_wrong_or_missing_work_scores_zero(sb):
    env = BashEnv(sb, max_turns=3)
    for seed in range(20):
        inst = sample_task(seed, 6)
        env.reset(inst)
        assert env.step("Final answer: definitely-wrong").reward == 0.0
        if inst.family == "create_file":
            env.reset(inst); assert env.step("Final answer: done").reward == 0.0   # never created it
    env.reset(sample_task(0, 1))                                                   # turn limit ends episode
    assert not env.step("hmm").done and not env.step("hmm").done and env.step("hmm").done
    env.close()


def test_tasks_deterministic_serializable_and_horizon_grows():
    a, b = sample_task(7, 5), sample_task(7, 5)
    assert a == b and TaskInstance.from_dict(json.loads(json.dumps(a.to_dict()))) == a
    lens = lambda lvl: max(len(sample_task(s, lvl).solution) for s in range(40))
    assert lens(8) > lens(2)
    assert {sample_task(s, 8).family for s in range(60)} >= {"hop_chain", "collect_codes", "needle_log"}
