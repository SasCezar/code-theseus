#!/usr/bin/env python3
"""Standalone driver that runs one coding-agent task via the OpenHands SDK
directly (`openhands.sdk` -- Agent/LLM/Conversation), not the `openhands`
CLI/TUI binary.

This is intentionally a self-contained script with no reverse_se imports:
the OpenHands SDK is installed in its own isolated environment (`uv tool
install openhands`, see openhands_launch.py) rather than as a dependency of
this project's own venv, to avoid its large, fast-moving dependency tree
(litellm, pydantic pins, etc.) colliding with reverse_se's. openhands_launch
invokes this script with that environment's own interpreter as a
subprocess and exchanges JSON over files, so the two dependency worlds
never have to share a venv.

Using the SDK directly instead of the `openhands` CLI gives:
  - explicit max_iteration_per_run / budget control (the CLI exposes
    neither as a flag);
  - the full typed event stream via a callback, written here as JSONL --
    spec section 11's "native agent trajectory/logs", not a
    terminal-formatted transcript;
  - direct access to LLM.metrics (accumulated_token_usage, accumulated_cost)
    after the run, for spec section 12's token/usage fields;
  - no dependency on `openhands --headless` requiring a prior interactive
    settings bootstrap (see MissingEnvironmentVariablesError in
    openhands_cli's AgentStore) -- constructing the Agent/LLM here directly
    sidesteps that CLI-only gate entirely.

Usage (always invoked as a subprocess by openhands_launch.py):
    <openhands-tool-venv>/bin/python openhands_runner.py <config.json> <result.json>

config.json fields: task_file, workspace_dir, run_dir, model, api_base,
api_key, temperature, top_p, max_output_tokens, max_iterations,
reasoning_effort, command_limit.
result.json fields: see RESULT_FIELDS below; always written, even on a
hard failure (terminal_status="error", error=<repr>), so the launcher
never has to guess whether a subprocess crash produced partial output.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import sys
import time
import traceback
from pathlib import Path

# One call to the SDK's run() loop -- one "iteration" against
# max_iteration_per_run -- always emits exactly one of these as its primary
# event: a tool call (ActionEvent), a context-compression pass
# (Condensation), or a plain reply with no tool call (MessageEvent from the
# agent, which also ends the run since the conversation moves to FINISHED).
# Verified empirically: for a run that hit the cap, ActionEvent count +
# Condensation count landed exactly `max_iterations` below what
# ConversationErrorEvent's own "reached maximum iterations limit (N)"
# message reported, with the gap fully explained by Condensation events the
# earlier ActionEvent-only count silently missed. There's no public,
# directly-readable iteration counter on Conversation/ConversationState --
# `iteration` is a local variable inside the SDK's own run() loop -- so this
# event-kind tally is the best available proxy, cross-checked against that
# message below whenever the cap is actually hit.
_ITERATION_EVENT_KINDS = {"ActionEvent", "Condensation"}
_MAX_ITERATIONS_RE = re.compile(r"maximum iterations limit \((\d+)\)")

# CLI-mode agents end the conversation on any reply without a tool call --
# observed: an agent thinking aloud ("I need to think about how query
# equality works...") ended its run after 24 steps with the library half
# written. With require_finish, only an explicit call to the finish tool
# (or the step/time budget) ends a run; a text-only stop gets this nudge.
_CONTINUE_NUDGE = (
    "You have not finished the task. Continue implementing the library. When "
    "the implementation is complete, run the visible tests, and then call the "
    "finish tool -- only the finish tool ends the task."
)
# Stuck detection stops a run when the agent repeats itself -- observed: a
# command hung the terminal, then every command (incl. four C-c) returned
# nothing, and the run stopped at step 120 of 1000 with no working code.
# Like a text-only stop, it's recoverable: the terminal tool can reset.
_STUCK_NUDGE = (
    "The task was paused because your recent actions repeated without "
    "progress -- your terminal is probably unresponsive. Call the terminal "
    "tool with reset set to true to start a fresh terminal session, then "
    "continue implementing the library. Only the finish tool ends the task."
)
_MAX_CONTINUATIONS = 50

RESULT_FIELDS = (
    "terminal_status",
    "wall_time_seconds",
    "agent_steps",
    "prompt_tokens",
    "completion_tokens",
    "accumulated_cost",
    "trajectory_path",
    "error",
    "continuations",
    "finished_by_tool",
    "stuck_recoveries",
    "commands_killed",
)

# Terminal commands killed at the per-command limit (see _limit_commands).
_commands_killed = 0


def main() -> None:
    config_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    config = json.loads(config_path.read_text(encoding="utf-8"))

    result: dict = dict.fromkeys(RESULT_FIELDS)
    result["wall_time_seconds"] = 0.0
    result["agent_steps"] = 0
    result["prompt_tokens"] = 0
    result["completion_tokens"] = 0
    result["accumulated_cost"] = 0.0
    result["commands_killed"] = 0

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = run_dir / "trajectory.jsonl"
    result["trajectory_path"] = str(trajectory_path)

    start = time.monotonic()
    llm, agent_steps = None, 0
    try:
        from openhands.sdk import LLM, Conversation
        from openhands_cli.utils import get_default_cli_agent
        from pydantic import SecretStr

        task_text = Path(config["task_file"]).read_text(encoding="utf-8")

        llm = LLM(
            model=config["model"],
            api_key=SecretStr(config["api_key"]),
            base_url=config.get("api_base") or None,
            temperature=config.get("temperature", 0.2),
            top_p=config.get("top_p", 1.0),
            max_output_tokens=config.get("max_output_tokens") or None,
            # The SDK's own default ("high") is Anthropic-oriented and not a
            # value every model's chat template accepts -- observed in
            # practice: Qwen3.8-Flash-Next's template raises a hard 400
            # ("Unexpected reasoning effort high. Supported types are xhigh
            # (default), medium, and low.") rather than ignoring it. Passing
            # None here (rather than leaving the SDK default in place) omits
            # the field so each model's own server-side default applies,
            # unless the caller explicitly opts into a value it knows the
            # model supports.
            reasoning_effort=config.get("reasoning_effort"),
            # The SDK's own default (300s) is a per-request HTTP timeout,
            # not a total-run budget -- too short once the agent is doing
            # real generation work under concurrent load sharing GPU
            # compute. Observed in practice: every run in an 8-concurrent
            # batch hit `litellm.Timeout: Connection timed out after 300.0
            # seconds`, retried 5x, and burned its entire --timeout budget
            # on nothing but repeated request timeouts -- zero completions
            # ever succeeded. This never showed up before closing the
            # network-cheat vector, because the agent's completions were
            # mostly short tool-call decisions (pip/git commands) rather
            # than substantial code generation.
            timeout=config.get("llm_timeout", 1200),
            # e.g. {"chat_template_kwargs": {"enable_thinking": False}} to turn
            # off a Qwen-style model's thinking (vLLM chat-template kwargs).
            litellm_extra_body=config.get("llm_extra_body") or {},
            usage_id="agent",
        )
        if config.get("agent_shell"):
            agent = _confined_agent(llm, config["workspace_dir"], config["agent_shell"])
            if config.get("command_limit"):
                _limit_commands(config["command_limit"])
        else:
            agent = get_default_cli_agent(llm)

        from openhands.sdk.tool.builtins.finish import FinishTool

        finished_by_tool = False

        with trajectory_path.open("w", encoding="utf-8") as events_file:

            def on_event(event) -> None:
                nonlocal agent_steps, finished_by_tool
                kind = type(event).__name__
                if (
                    kind == "ActionEvent"
                    and getattr(event, "tool_name", None) == FinishTool.name
                ):
                    finished_by_tool = True
                if kind in _ITERATION_EVENT_KINDS or (
                    kind == "MessageEvent" and getattr(event, "source", None) == "agent"
                ):
                    agent_steps += 1
                elif kind == "ConversationErrorEvent":
                    detail = getattr(event, "detail", "") or ""
                    match = _MAX_ITERATIONS_RE.search(detail)
                    if match:
                        # Authoritative when the cap is actually hit --
                        # overrides the event-kind tally above, which can
                        # undercount if some other event kind also consumes
                        # an iteration without matching the cases above.
                        agent_steps = int(match.group(1))
                try:
                    line = event.model_dump_json()
                except Exception:
                    line = json.dumps(
                        {"type": type(event).__name__, "repr": repr(event)}
                    )
                events_file.write(line + "\n")
                events_file.flush()

            conversation = Conversation(
                agent=agent,
                workspace=config["workspace_dir"],
                persistence_dir=str(run_dir / "persistence"),
                callbacks=[on_event],
                max_iteration_per_run=config.get("max_iterations", 100),
                stuck_detection=True,
                delete_on_close=False,
            )
            conversation.send_message(task_text)
            max_steps = config.get("max_iterations", 100)
            continuations = stuck_recoveries = 0
            while True:
                # The SDK's limit is per run() call: keep one shared budget.
                conversation.max_iteration_per_run = max(1, max_steps - agent_steps)
                conversation.run()
                status = conversation.state.execution_status.value
                if (
                    not config.get("require_finish")
                    or status not in ("finished", "stuck")
                    or finished_by_tool
                    or agent_steps >= max_steps
                    or continuations + stuck_recoveries >= _MAX_CONTINUATIONS
                ):
                    break
                if status == "stuck":
                    stuck_recoveries += 1
                    conversation.send_message(_STUCK_NUDGE)
                else:
                    continuations += 1
                    conversation.send_message(_CONTINUE_NUDGE)

            result["terminal_status"] = status
            result["continuations"] = continuations
            result["finished_by_tool"] = finished_by_tool
            result["stuck_recoveries"] = stuck_recoveries

    except Exception as exc:
        result["terminal_status"] = "error"
        result["error"] = f"{exc!r}\n{traceback.format_exc()}"

    # Also after an exception: a run cut off midway (observed: the vLLM
    # server crashing after 900 steps) did that work, and reported 0.
    result["agent_steps"] = agent_steps
    if llm is not None:
        usage = llm.metrics.accumulated_token_usage
        if usage is not None:
            result["prompt_tokens"] = usage.prompt_tokens
            result["completion_tokens"] = usage.completion_tokens
        result["accumulated_cost"] = llm.metrics.accumulated_cost
    result["commands_killed"] = _commands_killed
    result["wall_time_seconds"] = time.monotonic() - start
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _confined_agent(llm, workspace_dir: str, agent_shell: str):
    """The default CLI agent, confined to the workspace.

    This process runs as root inside the agent image, next to OpenHands' own
    environment -- which contains copies of libraries being benchmarked
    (jinja2, requests, click, ...); an agent was observed `find`ing and
    copying them. So:
      - terminal commands run through `agent_shell`, which drops to an
        unprivileged uid that can't read /root (OpenHands' env lives there);
      - the file editor, which runs in this (root) process, refuses paths
        outside the workspace (resolved, so symlinks can't escape);
      - no sub-agent tool (it would start unconfined tools of its own);
      - umask 0, so files the (root) editor creates stay writable by the
        agent's shell.
    """
    import os

    from openhands.sdk import Agent, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.file_editor.editor import FileEditor
    from openhands.tools.file_editor.exceptions import ToolError
    from openhands.tools.preset.default import (
        get_default_condenser,
        register_default_tools,
    )
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    os.umask(0)
    root = Path(workspace_dir).resolve()
    unguarded = FileEditor.__call__

    def guarded(self, *, command, path, **kwargs):
        resolved = Path(path).resolve()
        if resolved != root and root not in resolved.parents:
            raise ToolError(f"Access denied: {path} is outside the workspace {root}.")
        return unguarded(self, command=command, path=path, **kwargs)

    FileEditor.__call__ = guarded
    register_default_tools(enable_browser=False)
    return Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name, params={"shell_path": agent_shell}),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ],
        system_prompt_kwargs={"cli_mode": True},
        condenser=get_default_condenser(
            llm=llm.model_copy(update={"usage_id": "condenser"})
        ),
    )


_KILLED_NOTE = (
    "[The command was killed after {limit} seconds: that is the longest a "
    "single command may run. Avoid commands that never return; run long "
    "jobs in the background or with the timeout parameter.]\n"
)


def _agent_commands(include_shells: bool) -> list[int]:
    """Processes of the agent's commands: everything running as a non-root
    uid (agent_shell drops the agent's shell to one; this runner is root),
    minus -- unless include_shells -- the shells themselves (an agent
    process whose parent isn't one)."""
    procs = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            uid = os.stat(f"/proc/{entry}").st_uid
            with open(f"/proc/{entry}/stat", encoding="utf-8") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        procs[int(entry)] = (uid, ppid)
    agent = {pid for pid, (uid, _) in procs.items() if uid != 0}
    return [pid for pid in agent if include_shells or procs[pid][1] in agent]


def _kill(include_shells: bool) -> None:
    """SIGKILL the agent's command processes (see _agent_commands) in a few
    rounds -- e.g. a multiprocessing pool respawns workers."""
    for _ in range(3):
        for pid in _agent_commands(include_shells):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)


def _limit_commands(limit: int) -> None:
    """Kill a terminal command's processes once one tool call has run for
    `limit` seconds. The terminal tool only returns when its command ends,
    a hard `timeout` the agent set expires, or output stops changing for
    30 s -- observed: a script whose multiprocessing pool respawned failing
    workers forever kept printing, so the call never returned and the run
    sat frozen for 4 h. After the kill the shell prints its prompt and the
    call returns (with a note prepended); if the shell itself is what hangs,
    it's killed a minute later and the terminal is reset."""
    import threading

    from openhands.sdk.llm import TextContent
    from openhands.tools.terminal.impl import TerminalExecutor

    unlimited = TerminalExecutor.__call__

    def limited(self, action, conversation=None):
        global _commands_killed
        fired, shell_killed = threading.Event(), threading.Event()

        def kill(include_shells):
            # Flag first: the call can return as soon as the first round of
            # kills lands, before _kill's later rounds finish.
            if _agent_commands(include_shells):
                fired.set()
                if include_shells:
                    shell_killed.set()
                _kill(include_shells)

        timers = [
            threading.Timer(limit, kill, (False,)),
            threading.Timer(limit + 60, kill, (True,)),
        ]
        for timer in timers:
            timer.daemon = True
            timer.start()
        try:
            observation = unlimited(self, action, conversation)
        finally:
            for timer in timers:
                timer.cancel()
        if not fired.is_set():
            return observation
        _commands_killed += 1
        text = _KILLED_NOTE.format(limit=limit)
        if shell_killed.is_set():
            # The dead shell isn't noticed by the terminal tool: later
            # commands would silently return nothing.
            self.reset()
            text += "[The terminal was restarted: its session state is gone.]\n"
        note = TextContent(text=text)
        return observation.model_copy(update={"content": [note, *observation.content]})

    TerminalExecutor.__call__ = limited


if __name__ == "__main__":
    main()
