"""Run on the training box: does bubblewrap work and isolate as expected?
    uv run python scripts/sandbox_check.py
"""
import tempfile
from rwkv_enhance.sandbox import BwrapSandbox

sb = BwrapSandbox(timeout=5)
with tempfile.TemporaryDirectory() as d:
    checks = {
        "runs a command": ("echo ok", lambda r: r.exit_code == 0 and r.output.strip() == "ok"),
        "work dir writable": ("echo x > f && cat f", lambda r: r.output.strip() == "x"),
        "system read-only": ("touch /usr/x", lambda r: r.exit_code != 0),
        "no host home": ("ls /home", lambda r: r.exit_code != 0),
        "no network": ("exec 3<>/dev/tcp/1.1.1.1/80", lambda r: r.exit_code != 0),
        "timeout enforced": ("sleep 30", lambda r: r.timed_out),
    }
    ok = True
    for name, (cmd, test) in checks.items():
        r = sb.run(d, cmd)
        good = test(r)
        ok &= good
        print(("PASS " if good else "FAIL ") + name + ("" if good else f"   -> {r}"))
print("\nsandbox OK" if ok else "\nsandbox NOT safe to use - see FAIL lines "
      "(on Ubuntu 24.04 a first-run error about 'uid map: Permission denied' means AppArmor is "
      "restricting user namespaces; see `apparmor_restrict_unprivileged_userns`)")
raise SystemExit(0 if ok else 1)
