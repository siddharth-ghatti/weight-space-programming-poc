"""Isolated Python runner, a port of lm_understand-logs/harness/sandbox.py for macOS.

Each run is a fresh `python -I` process from a restricted venv (numpy/scipy/safetensors/tokenizers;
no torch, no transformers), with a cleared environment, cwd = the persistent work dir, a wall-clock
timeout, capped output, and (when available) a `sandbox-exec` profile that denies network access and
restricts writes to the work dir. A JSONL log records every run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from wsp.model.weights import ROOT

SANDBOX_PY = ROOT / ".sandbox-venv" / "bin" / "python"


def _profile(work_dir: Path, tmp_dir: Path) -> str:
    # allow-default + targeted denies is far more robust than deny-default on macOS.
    allowed_writes = [work_dir, tmp_dir, Path("/private/tmp"), Path("/private/var/folders"), Path("/dev/null")]
    allow = "\n".join(f'  (subpath "{p.resolve()}")' if p.is_dir() else f'  (literal "{p}")' for p in allowed_writes)
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write*
{allow}
)
"""


@dataclass
class RunResult:
    output: str
    returncode: int
    duration_s: float
    timed_out: bool


class Sandbox:
    def __init__(self, work_dir: Path, log_path: Path, *, timeout_s: int = 900, stdout_cap: int = 20000,
                 stderr_tail: int = 6000, use_sandbox_exec: bool = True, python: Path = SANDBOX_PY):
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path)
        self.timeout_s = timeout_s
        self.stdout_cap = stdout_cap
        self.stderr_tail = stderr_tail
        self.python = Path(python)
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="wsp-sbx-"))
        self.sandbox_exec = shutil.which("sandbox-exec") if use_sandbox_exec else None
        if not self.python.exists():
            raise FileNotFoundError(f"sandbox python not found at {self.python}; create .sandbox-venv first")
        self.runs = 0

    def run(self, code: str, prelude: str = "") -> RunResult:
        self.runs += 1
        script = self.tmp_dir / f"run_{self.runs:04d}.py"
        script.write_text(prelude + "\n# ---- editor code ----\n" + code + "\n")
        cmd = [str(self.python), "-I", "-u", str(script)]
        env = {
            "HOME": str(self.tmp_dir),
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(self.tmp_dir),
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "WSP_WORK": str(self.work_dir),
        }
        if self.sandbox_exec:
            prof = self.tmp_dir / "profile.sb"
            prof.write_text(_profile(self.work_dir, self.tmp_dir))
            cmd = [self.sandbox_exec, "-f", str(prof)] + cmd
        t0 = time.time()
        timed_out = False
        try:
            p = subprocess.run(cmd, cwd=self.work_dir, env=env, capture_output=True, text=True,
                               timeout=self.timeout_s, start_new_session=True)
            out, err, rc = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            rc, timed_out = -9, True
        dur = time.time() - t0
        text = out
        if len(text) > self.stdout_cap:
            text = text[: self.stdout_cap] + f"\n... [stdout truncated to {self.stdout_cap} chars]"
        if err.strip():
            tail = err[-self.stderr_tail :]
            text += ("\n" if text else "") + "[stderr]\n" + tail
        if timed_out:
            text += f"\n[timed out after {self.timeout_s}s; process killed]"
        if rc not in (0, -9):
            text += f"\n[exit code {rc}]"
        with self.log_path.open("a") as f:
            f.write(json.dumps({"ts": t0, "run": self.runs, "duration_s": dur, "returncode": rc,
                                "timed_out": timed_out, "code": code, "output": text[:6000]}) + "\n")
        return RunResult(text, rc, dur, timed_out)

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
