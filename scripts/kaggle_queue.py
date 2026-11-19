"""Push Kaggle jobs in order, unattended: each job waits for a start time (such as the weekly
quota reset) and for other jobs to complete, then goes out through scripts/kaggle_jobs.py push.

The queue is a local JSON list. Keep it in .kaggle_jobs/, which git ignores, because it names
token files:

    [{"name": "a256-rb", "token_file": "C:/.../token.txt", "plans": ["--arms blind ..."],
      "after": ["r256-smoke"], "not_before": "2026-11-21T00:05:00Z",
      "input_job": [], "resume_of": null}, ...]

- after: jobs that must reach COMPLETE first. If one errors or is cancelled, this job is skipped
  and the reason is recorded.
- resume_of: the job whose run this one continues (normally also its input_job). It is pushed
  only if that job's log shows a run stopped by --max-hours. Otherwise the run already finished,
  and a second session would only rebuild the cache.
- not_before: the earliest push time, in UTC.

Each job's outcome is written back into the file ("state": pushed / skipped / failed, with "why"),
so the queue can be stopped and started again. Progress goes to stdout.

usage:
    python scripts/kaggle_queue.py .kaggle_jobs/queue.json [--every 300]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_jobs as kj  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DONE = ("pushed", "skipped", "failed")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def status(name: str) -> str:
    """complete / error / cancelled / running / queued / unknown, from the kernel's status."""
    try:
        job = kj.load_job(name)
        text = kj.kaggle("kernels", "status", job["kernel"], token_file=kj._token(job), capture=True).lower()
    except SystemExit as e:  # not pushed yet, or the CLI failed: try again next round
        return f"unknown ({e})"
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in text:
            return s
    return "unknown"


def stopped_by_max_hours(name: str) -> bool:
    job = kj.load_job(name)
    text = kj.kaggle("kernels", "logs", job["kernel"], token_file=kj._token(job), capture=True)
    return "stopped by max_hours" in text or "max_hours reached" in text


def push(e: dict) -> tuple[bool, str]:
    cmd = [sys.executable, "scripts/kaggle_jobs.py", "push", e["name"], "--runner", e.get("runner", "gen")]
    if e.get("token_file"):
        cmd += ["--token-file", e["token_file"]]
    for p in e["plans"]:
        cmd += ["--plan", p]
    for j in e.get("input_job", []):
        cmd += ["--input-job", j]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"})
    text = (out.stdout + out.stderr).strip()
    ok = out.returncode == 0 and "successfully pushed" in text
    return ok, text.splitlines()[-1] if text else f"exit {out.returncode}"


def step(queue: list[dict]) -> bool:
    """One pass over the queue; returns True while jobs are still waiting."""
    waiting = False
    for e in queue:
        if e.get("state") in DONE:
            continue
        nb = e.get("not_before")
        if nb and datetime.now(timezone.utc) < datetime.fromisoformat(nb.replace("Z", "+00:00")):
            waiting = True
            continue
        states = {a: status(a) for a in e.get("after", [])}
        bad = {a: s for a, s in states.items() if s in ("error", "cancel")}
        if bad:
            e.update(state="skipped", why=f"{now()}: {bad}")
            print(f"[{now()}] skip {e['name']}: {bad}", flush=True)
            continue
        if any(s != "complete" for s in states.values()):
            waiting = True
            continue
        if e.get("resume_of") and not stopped_by_max_hours(e["resume_of"]):
            e.update(state="skipped", why=f"{now()}: {e['resume_of']} finished its run in one session")
            print(f"[{now()}] skip {e['name']}: {e['resume_of']} needs no second session", flush=True)
            continue
        ok, msg = push(e)
        e.update(state="pushed" if ok else "failed", why=f"{now()}: {msg}")
        print(f"[{now()}] {'pushed' if ok else 'FAILED'} {e['name']}: {msg}", flush=True)
    return waiting


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("queue", type=Path)
    p.add_argument("--every", type=int, default=300, help="seconds between passes")
    args = p.parse_args()
    while True:
        queue = json.loads(args.queue.read_text(encoding="utf-8"))
        waiting = step(queue)
        args.queue.write_text(json.dumps(queue, indent=1), encoding="utf-8")
        if not waiting:
            print(f"[{now()}] queue done: " + ", ".join(f"{e['name']}={e.get('state')}" for e in queue), flush=True)
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
