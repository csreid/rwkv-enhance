"""Sandboxed shell execution for agent rollouts.

BwrapSandbox: bubblewrap with all namespaces unshared (no network, own pid/ipc/uts),
read-only system dirs, an empty tmpfs /tmp, the episode work dir as the only writable
location, and a cleared environment. The host home directory is not mounted.

UnsafeLocalSandbox runs commands directly on the host. It exists for tests of TRUSTED
commands only and refuses to construct unless explicitly allowed.
"""
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass


@dataclass
class ExecResult:
    output: str
    exit_code: int
    timed_out: bool = False
    truncated: bool = False


def _run(argv, cwd, timeout, max_bytes, env=None):
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         start_new_session=True)
    state = {"timed_out": False}

    def kill():
        try:
            os.killpg(p.pid, 9)
        except ProcessLookupError:
            pass

    def on_timeout():
        state["timed_out"] = True
        kill()

    timer = threading.Timer(timeout, on_timeout)
    timer.start()
    try:
        data = p.stdout.read(max_bytes + 1)      # never buffers more than the cap
        truncated = len(data) > max_bytes
        if truncated:
            kill()                                # runaway output (e.g. `yes`)
        p.wait()
    finally:
        timer.cancel()
        kill()                                    # nothing may outlive the call
        p.stdout.close()
    return ExecResult(data[:max_bytes].decode("utf-8", "replace"), p.returncode,
                      state["timed_out"], truncated)


class BwrapSandbox:
    def __init__(self, timeout=10.0, max_output_bytes=64_000, mem_limit_kb=4_000_000,
                 max_file_kb=10_240):
        if shutil.which("bwrap") is None:
            raise RuntimeError("bubblewrap not found (apt install bubblewrap)")
        self.timeout, self.max_bytes = timeout, max_output_bytes
        self.limits = f"ulimit -v {mem_limit_kb}; ulimit -f {max_file_kb}; "

    def _argv(self, workdir, cmd):
        a = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
             "--ro-bind", "/usr", "/usr",
             "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
             "--symlink", "usr/lib", "/lib"]
        if os.path.isdir("/usr/lib64"):
            a += ["--symlink", "usr/lib64", "/lib64"]
        a += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
              "--bind", os.path.abspath(workdir), "/work", "--chdir", "/work",
              "--setenv", "HOME", "/work", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
              "--setenv", "LANG", "C.UTF-8", "--setenv", "TERM", "dumb",
              "bash", "-c", self.limits + cmd]
        return a

    def run(self, workdir, cmd):
        return _run(self._argv(workdir, cmd), None, self.timeout, self.max_bytes)


class UnsafeLocalSandbox:
    """NOT a sandbox. Trusted commands only (tests, expert solvers)."""

    def __init__(self, timeout=10.0, max_output_bytes=64_000, i_understand_this_is_unsafe=False):
        if not (i_understand_this_is_unsafe or os.environ.get("RWKV_ENHANCE_ALLOW_UNSAFE") == "1"):
            raise RuntimeError("UnsafeLocalSandbox runs commands on the host; refusing. "
                               "Use BwrapSandbox for model-generated commands.")
        self.timeout, self.max_bytes = timeout, max_output_bytes

    def run(self, workdir, cmd):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": workdir}
        return _run(["bash", "-c", cmd], workdir, self.timeout, self.max_bytes, env)
