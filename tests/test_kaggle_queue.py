"""scripts/kaggle_queue.py --watch: failed Kaggle sessions are retried and their dependants follow."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("kaggle_queue", ROOT / "scripts" / "kaggle_queue.py")
    kq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kq)
    return kq


def test_failed_sessions_are_retried_and_dependants_follow(monkeypatch):
    kq = load()
    t0 = datetime.now(timezone.utc).isoformat()
    queue = [
        {"name": "a", "state": "pushed", "pushed_at": t0, "plans": ["--x"], "input_job": [], "resume_of": None,
         "token_file": "tok"},
        {"name": "a-2", "plans": ["--x"], "after": ["a"], "input_job": ["a"], "resume_of": "a", "token_file": "tok"},
        {"name": "old", "state": "pushed", "plans": ["--y"]},  # pushed before --watch: not followed
        {"name": "done", "state": "pushed", "pushed_at": t0, "plans": ["--z"]},
    ]
    states = {"a": "error", "done": "complete", "a-r1": "cancel", "a-r2": "error"}
    seen = []
    monkeypatch.setattr(kq, "status", lambda n: seen.append(n) or states.get(n, "running"))
    assert kq.watch(queue)
    names = [e["name"] for e in queue]
    assert "a-r1" in names and "old" not in seen
    r1 = queue[names.index("a-r1")]
    assert (r1["plans"], r1["token_file"], r1["retry_of"], r1["after"]) == (["--x"], "tok", "a", [])
    assert (queue[1]["after"], queue[1]["input_job"], queue[1]["resume_of"]) == (["a-r1"], ["a-r1"], "a-r1")
    assert "retried as a-r1" in queue[0]["outcome"] and "complete" in queue[3]["outcome"]
    assert not kq.watch(queue)  # a-r1 not pushed yet: nothing in flight is watched again
    r1.update(state="pushed", pushed_at=t0)
    kq.watch(queue)
    r2 = queue[[e["name"] for e in queue].index("a-r2")]
    assert r2["retry_of"] == "a" and queue[1]["after"] == ["a-r2"]
    r2.update(state="pushed", pushed_at=t0)
    kq.watch(queue)
    assert "giving up" in r2["outcome"] and not any(e["name"] == "a-r3" for e in queue)


def test_jobs_past_the_watch_window_are_closed(monkeypatch):
    kq = load()
    old = (datetime.now(timezone.utc) - timedelta(hours=14)).isoformat()
    queue = [{"name": "a", "state": "pushed", "pushed_at": old, "plans": []}]
    monkeypatch.setattr(kq, "status", lambda n: "error")
    assert not kq.watch(queue)
    assert "no final status" in queue[0]["outcome"] and len(queue) == 1
