from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_NAME = "riemann-e2e-test"
ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def verify_isolated_volume_config(env: dict[str, str]) -> None:
    """Assert docker compose -p riemann-e2e-test uses an isolated project-scoped volume."""
    res = _run(
        ["docker", "compose", "-p", PROJECT_NAME, "config"],
        env=env,
        check=True,
    )
    cfg = res.stdout
    expected_prefix = f"name: {PROJECT_NAME}_"
    if expected_prefix not in cfg:
        raise RuntimeError(
            f"Refusing to run E2E: expected isolated volume '{expected_prefix}*' in compose config.\n"
            f"Config snippet:\n{cfg[-600:]}"
        )
    if "127.0.0.1" not in cfg:
        raise RuntimeError("Expected loopback host port binding 127.0.0.1 in compose config")
    print(f"Verified isolated Compose volume prefix '{PROJECT_NAME}_' and loopback binding.", flush=True)


def main() -> int:
    env = os.environ.copy()
    env["RIEMANN_BLOCK_OUTBOUND_ON_INFERENCE"] = "1"
    env["RIEMANN_E2E_BASE_URL"] = "http://127.0.0.1:8000"
    env["RIEMANN_E2E_COMPOSE_PROJECT"] = PROJECT_NAME

    verify_isolated_volume_config(env)

    try:
        up = _run(
            ["docker", "compose", "-p", PROJECT_NAME, "up", "--build", "-d", "--wait"],
            env=env,
            check=False,
        )
        if up.returncode != 0:
            print(up.stdout)
            print(up.stderr, file=sys.stderr)
            return up.returncode

        py_exe = ROOT / ".venv" / "Scripts" / "python.exe"
        if not py_exe.exists():
            py_exe = Path(sys.executable)

        test_proc = subprocess.run(
            [str(py_exe), "-m", "pytest", "tests/test_api_e2e.py", "-v"],
            cwd=str(ROOT),
            env=env,
        )
        return test_proc.returncode
    finally:
        down = _run(
            ["docker", "compose", "-p", PROJECT_NAME, "down", "-v", "--remove-orphans"],
            env=env,
            check=False,
        )
        if down.returncode != 0:
            print(down.stderr, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
