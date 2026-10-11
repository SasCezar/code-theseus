#!/usr/bin/env python3
"""Run a grading pytest command so that no single test can stall the grade.

pytest-timeout (commit0.grade's per-test limit) stops a test by raising from
a signal handler, which can't stop everything -- observed:
  - agent code spinning in a loop that never returns to the interpreter
    (100% CPU, ignored pytest-timeout, SIGINT and SIGTERM alike), stuck
    at 87% of cachetools' suite until the whole-grading timeout;
  - pytest finished and wrote its report, but leftover multiprocessing
    workers kept it from exiting (portalocker), again until the timeout.

So this runs pytest in its own process group and watches its verbose
output (it prints each test's node id before running it):
  - no new output for --idle seconds: SIGKILL the group, record the test it
    was on as failed, and rerun pytest with every test seen so far
    deselected (this file doubles as the pytest plugin that deselects them,
    loaded with `-p grade_watchdog`);
  - report written but pytest still running after --linger seconds: SIGKILL
    the group, keep the report.
The per-attempt JSON reports are merged into <out>/report.json (only node
ids and call outcomes -- what commit0.score reads), plus a "watchdog"
record of attempts and hung tests. Exits 3 if --deadline cut grading short.

Standalone (no reverse_se imports, Python 3.8+): it runs inside the grading
container with the library's own interpreter.

Usage:
    python grade_watchdog.py --out /grade --idle 120 --deadline 1700 \
        -- "<pytest command without --json-report-file>"
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time

DESELECT_ENV = "GRADE_WATCHDOG_DESELECT"
MAX_ATTEMPTS = 50
# Verbose per-test lines: "tests/x.py::T::test_y <- tests/base.py PASSED [ 1%]"
_RESULT_RE = re.compile(
    r"^(\S+?::\S.*?) (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s|$)"
)
_INHERITED = re.compile(r" <- \S+$")


# ── pytest plugin (loaded in the pytest process via -p grade_watchdog) ───────


def pytest_collection_modifyitems(config, items):
    path = os.environ.get(DESELECT_ENV)
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        skip = set(f.read().splitlines())
    keep = [item for item in items if item.nodeid not in skip]
    if len(keep) < len(items):
        config.hook.pytest_deselected(
            items=[item for item in items if item.nodeid in skip]
        )
        items[:] = keep


# ── watchdog ─────────────────────────────────────────────────────────────────


def node_id(text: str) -> str:
    """'tests/t.py::A::test_x <- tests/base.py' -> 'tests/t.py::A::test_x'."""
    return _INHERITED.sub("", text.strip())


def verbose_outcomes(log_path: str) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            match = _RESULT_RE.match(line.strip())
            if match:
                outcomes[node_id(match.group(1))] = match.group(2).lower()
    return outcomes


def running_test(log_path: str) -> str | None:
    """The test pytest was on: its node id is the log's last, unfinished
    line (verbose mode prints the id, then the outcome once it's done)."""
    with open(log_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 65536))
        tail = f.read().decode("utf-8", errors="replace")
    last = tail.rstrip("\n").rsplit("\n", 1)[-1].strip()
    if "::" not in last or _RESULT_RE.match(last):
        return None
    return node_id(last)


def kill_group(proc: subprocess.Popen) -> None:
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def run_attempt(command, env, log_path, report_path, idle, linger, deadline):
    """Returns 'exited', 'lingered', 'hung' or 'deadline'."""
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    last_size, last_change = -1, time.monotonic()
    while True:
        if proc.poll() is not None:
            kill_group(proc)  # stray children
            return "exited"
        now = time.monotonic()
        size = os.path.getsize(log_path)
        if size != last_size:
            last_size, last_change = size, now
        if (
            os.path.isfile(report_path)
            and time.time() - os.path.getmtime(report_path) > linger
        ):
            kill_group(proc)
            return "lingered"
        if now - last_change > idle:
            kill_group(proc)
            return "hung"
        if now > deadline:
            kill_group(proc)
            return "deadline"
        time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--idle", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True, help="seconds")
    parser.add_argument("--linger", type=float, default=30)
    parser.add_argument("command")
    args = parser.parse_args()

    deadline = time.monotonic() + args.deadline
    deselect_path = os.path.join(args.out, "deselect.txt")
    env = {
        **os.environ,
        DESELECT_ENV: deselect_path,
        # The node-id lines must reach the log as they're printed.
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(
            p
            for p in (
                os.path.dirname(os.path.abspath(__file__)),
                os.environ.get("PYTHONPATH"),
            )
            if p
        ),
    }
    outcomes: dict[str, str] = {}
    seen: set[str] = set()
    attempts, hung, status = [], [], "exited"
    for attempt in range(MAX_ATTEMPTS):
        suffix = "" if attempt == 0 else f"_{attempt}"
        log_path = os.path.join(args.out, f"test_output{suffix}.txt")
        report_path = os.path.join(args.out, f"report_attempt{attempt}.json")
        status = run_attempt(
            f"{args.command} --json-report-file={report_path} -p grade_watchdog",
            env,
            log_path,
            report_path,
            args.idle,
            args.linger,
            deadline,
        )
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            for t in report.get("tests", []):
                seen.add(t["nodeid"])
                if t.get("call"):
                    outcomes[t["nodeid"]] = t["call"]["outcome"]
            os.remove(report_path)  # can be huge (captured output); merged below
        else:
            ran = verbose_outcomes(log_path)
            seen.update(ran)
            outcomes.update(ran)
        attempts.append(status)
        if status != "hung":
            break
        stuck = running_test(log_path)
        if stuck is None or stuck in seen:
            break  # hung outside a test (e.g. collection): no way forward
        hung.append(stuck)
        seen.add(stuck)
        outcomes[stuck] = "failed"
        with open(deselect_path, "w", encoding="utf-8") as f:
            f.write("".join(f"{t}\n" for t in sorted(seen)))

    merged = {
        "tests": [
            {"nodeid": t, "call": {"outcome": o}} for t, o in sorted(outcomes.items())
        ],
        "watchdog": {"attempts": attempts, "hung_tests": hung},
    }
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f)
    if status == "deadline":
        print(f"grading deadline ({args.deadline:.0f}s) reached", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
