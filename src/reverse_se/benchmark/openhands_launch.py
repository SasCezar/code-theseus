"""Launch one openhands_runner.py run in a network-isolated container and
load back its result.

Sandboxing matters: OpenHands' LocalWorkspace gives the agent's shell full
network access, and an agent will (confirmed in practice) `pip download`
the package being benchmarked or `git clone` its upstream repo and copy the
reference implementation. Allowlisting can't tell a dependency install from
downloading the answer, so the agent's container is attached only to a
Docker network with no route to the internet (`internal: true`, e.g.
compose.yaml's `agent-sandbox`) that the model endpoint is also on.

The runner executes inside a caller-supplied image that already has the
project's environment installed (Commit0 repo image + OpenHands, see
commit0.build_agent_image), so the agent can run the tests offline.
openhands_runner.py uses the OpenHands SDK from that image's own `uv tool`
environment, never this project's venv.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


_RUNNER_SCRIPT = Path(__file__).with_name("openhands_runner.py")


@dataclass
class OpenHandsRunResult:
    terminal_status: str
    wall_time_seconds: float
    agent_steps: int
    prompt_tokens: int
    completion_tokens: int
    accumulated_cost: float
    trajectory_path: str | None
    error: str | None
    launch_error: str | None = None
    subprocess_returncode: int | None = None
    subprocess_stderr: str = ""
    continuations: int = 0
    finished_by_tool: bool | None = None
    stuck_recoveries: int = 0
    commands_killed: int = 0


def _sandbox_container_name(run_dir: Path) -> str:
    # Deterministic and unique per run (run_dir is already unique per
    # model/project/condition/replicate), so a timed-out run can be found
    # and killed by name afterward -- see _launch()'s TimeoutExpired
    # handler. Docker container names only allow [a-zA-Z0-9_.-].
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(run_dir).strip("/"))
    return f"agent-sandbox-{safe}"[:250]


_EMPTY_RESULT = {
    "terminal_status": "launch_error",
    "wall_time_seconds": 0.0,
    "agent_steps": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "accumulated_cost": 0.0,
    "trajectory_path": None,
    "error": None,
}


def _write_config(task_text: str, run_dir: Path, fields: dict) -> tuple[Path, Path]:
    """Write task.md + openhands_config.json into run_dir. `fields` holds
    paths as openhands_runner.py will see them (container paths when it runs
    in a container); task_file is derived from fields["run_dir"]."""
    run_dir.mkdir(parents=True, exist_ok=True)
    task_file = run_dir / "task.md"
    task_file.write_text(task_text, encoding="utf-8")
    config_path = run_dir / "openhands_config.json"
    result_path = run_dir / "openhands_result.json"
    result_path.unlink(missing_ok=True)
    fields = {"task_file": str(Path(fields["run_dir"]) / "task.md"), **fields}
    config_path.write_text(json.dumps(fields), encoding="utf-8")
    return config_path, result_path


def _launch(
    command: list[str],
    container_name: str,
    result_path: Path,
    timeout: int,
) -> OpenHandsRunResult:
    empty = _EMPTY_RESULT
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # subprocess.run's timeout only kills the local `docker run` client
        # process -- the container it started keeps running on the daemon
        # side, orphaned, unless explicitly stopped. Confirmed in practice:
        # a timed-out batch left every container running until killed by hand.
        subprocess.run(
            ["docker", "kill", container_name], capture_output=True, timeout=30
        )
        return OpenHandsRunResult(
            **{
                **empty,
                "terminal_status": "timed_out",
                "wall_time_seconds": float(timeout),
            },
            launch_error=None,
            subprocess_stderr=_as_text(exc.stderr),
        )

    if not result_path.is_file():
        return OpenHandsRunResult(
            **empty,
            launch_error="openhands_runner.py produced no result.json",
            subprocess_returncode=proc.returncode,
            subprocess_stderr=_as_text(proc.stderr)[-8000:],
        )

    data = json.loads(result_path.read_text(encoding="utf-8"))
    return OpenHandsRunResult(
        terminal_status=data.get("terminal_status") or "unknown",
        wall_time_seconds=data.get("wall_time_seconds", 0.0),
        agent_steps=data.get("agent_steps", 0),
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        accumulated_cost=data.get("accumulated_cost", 0.0),
        trajectory_path=data.get("trajectory_path"),
        error=data.get("error"),
        continuations=data.get("continuations") or 0,
        finished_by_tool=data.get("finished_by_tool"),
        stuck_recoveries=data.get("stuck_recoveries") or 0,
        commands_killed=data.get("commands_killed") or 0,
        subprocess_returncode=proc.returncode,
        subprocess_stderr=_as_text(proc.stderr)[-8000:] if proc.returncode != 0 else "",
    )


def run_task(
    task_text: str,
    workspace_dir: Path,
    run_dir: Path,
    image: str,
    network: str,
    host_path,
    model: str,
    api_key: str,
    api_base: str | None = None,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_output_tokens: int | None = None,
    max_iterations: int = 100,
    reasoning_effort: str | None = None,
    llm_timeout: int = 1200,
    command_limit: int | None = None,
    timeout: int = 3600,
    openhands_python: str = "/root/.local/share/uv/tools/openhands/bin/python",
    workspace_mount: str = "/testbed",
    platform: str | None = "linux/amd64",
    agent_shell: str | None = None,
    llm_extra_body: dict | None = None,
) -> OpenHandsRunResult:
    """Run the agent on workspace_dir inside `image`, attached only to
    `network`. `timeout` is the outer wall-clock budget for the whole run;
    `llm_timeout` is the per-LLM-request HTTP timeout inside the runner;
    `command_limit` the longest one terminal command may run (see
    openhands_runner._limit_commands).
    `agent_shell` (a path in the image) confines the agent: terminal
    commands run through it and file edits stay inside the workspace -- see
    openhands_runner._confined_agent.

    Only what the agent needs is mounted -- never the project root, which
    holds reference clones (data/repository) and every generated doc:
    workspace_dir at workspace_mount, with an anonymous volume over
    <workspace_mount>/.venv (Docker seeds it from the image, so the
    preinstalled venv stays visible under the bind mount); run_dir at /run
    (task, config, trajectory); the runner script read-only. host_path maps
    a local path to the path the Docker daemon sees (they differ when this
    runs inside the `benchmark` container).
    """
    workspace_dir = workspace_dir.resolve()
    run_dir = run_dir.resolve()
    config_path, result_path = _write_config(
        task_text,
        run_dir,
        {
            "workspace_dir": workspace_mount,
            "run_dir": "/run",
            "model": model,
            "api_key": api_key,
            "api_base": api_base,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
            "max_iterations": max_iterations,
            "reasoning_effort": reasoning_effort,
            "llm_timeout": llm_timeout,
            "agent_shell": agent_shell,
            "llm_extra_body": llm_extra_body,
            "command_limit": command_limit,
            # Only the finish tool (or the budget) ends a run -- see
            # openhands_runner._CONTINUE_NUDGE.
            "require_finish": True,
        },
    )
    container_name = _sandbox_container_name(run_dir)
    subprocess.run(
        ["docker", "rm", "-fv", container_name], capture_output=True, timeout=30
    )
    command = [
        "docker",
        "run",
        "--rm",
        # An init process as PID 1 reaps orphaned children of the agent's
        # commands, which the runner (otherwise PID 1) leaves as zombies.
        "--init",
        "--name",
        container_name,
        *(["--platform", platform] if platform else []),
        "--network",
        network,
        "-v",
        f"{host_path(workspace_dir)}:{workspace_mount}",
        "-v",
        f"{workspace_mount}/.venv",
        "-v",
        f"{host_path(run_dir)}:/run",
        "-v",
        f"{host_path(_RUNNER_SCRIPT.resolve())}:/opt/openhands_runner.py:ro",
        "-w",
        workspace_mount,
        "--entrypoint",
        openhands_python,
        image,
        "/opt/openhands_runner.py",
        "/run/" + config_path.name,
        "/run/" + result_path.name,
    ]
    result = _launch(command, container_name, result_path, timeout)
    if result.trajectory_path and result.trajectory_path.startswith("/run/"):
        result.trajectory_path = str(
            run_dir / Path(result.trajectory_path).relative_to("/run")
        )
    return result
