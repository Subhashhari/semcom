"""Compare two runs' validation curves at the same steps, while they train (plan of 27 Nov: decide
early where validation can, and only there).

Every validation pass scores the same 32 validation images with fixed channel and sampling seeds
(semcom.gen.checks.val_quality), and the validation loss uses fixed timesteps and noise, so two
runs of one data set and resolution pair image for image at equal steps. The scores come from the
VAL lines of the training logs.

Use it to stop a clear loser or confirm a clear winner early, never to rank close runs or to
replace a test-set comparison (month plan, 6.1):
- 32 images: a paired DINOv2 difference below about 0.02 is noise;
- validation loss does not order recipes with different auxiliary losses (REPA's is higher
  although its test results are better), so look at DINOv2 and PSNR first.

usage: val_compare.py REF NEW [--ref-run NAME] [--new-run NAME] [--last 3]
  REF, NEW: a log file, or kaggle:<job> (the job's full Kaggle log, while it runs too)
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAL = re.compile(r"^VAL run=(\S+) (.*)$")
SIGNIFICANT = {"dino": 0.02, "psnr": 0.2}  # below these, a 32-image paired difference is noise


def read_log(src: str) -> str:
    if src.startswith("kaggle:"):
        sys.path.insert(0, str(ROOT / "scripts"))
        import kaggle_jobs as kj  # noqa: E402

        job = kj.load_job(src.split(":", 1)[1])
        return kj.kaggle("kernels", "logs", job["kernel"], token_file=kj._token(job), capture=True)
    return Path(src).read_text(encoding="utf-8", errors="replace")


def curves(text: str, run: str | None) -> tuple[str, dict[int, dict[str, float]]]:
    """{step: {metric: value}} of one run's VAL lines (the run with most of them by default)."""
    by_run: dict[str, dict[int, dict[str, float]]] = {}
    for line in text.replace("\\n", "\n").replace("\r", "\n").splitlines():
        line = line.strip().strip('"').strip()
        m = VAL.search(line[line.find("VAL run="):] if "VAL run=" in line else "")
        if not m:
            continue
        vals = {}
        for kv in m.group(2).split():
            k, _, v = kv.partition("=")
            try:
                vals[k] = float(v)
            except ValueError:
                pass
        if "step" in vals:
            by_run.setdefault(m.group(1), {})[int(vals.pop("step"))] = vals
    if not by_run:
        raise SystemExit("no VAL lines found")
    name = run or max(by_run, key=lambda r: len(by_run[r]))
    if name not in by_run:
        raise SystemExit(f"no VAL lines for run {name}; runs in this log: {sorted(by_run)}")
    return name, by_run[name]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ref")
    p.add_argument("new")
    p.add_argument("--ref-run")
    p.add_argument("--new-run")
    p.add_argument("--last", type=int, default=3, help="validations averaged for the verdict")
    a = p.parse_args()
    rname, ref = curves(read_log(a.ref), a.ref_run)
    nname, new = curves(read_log(a.new), a.new_run)
    steps = sorted(set(ref) & set(new))
    if not steps:
        raise SystemExit(f"no common validation steps ({rname}: up to {max(ref)}, {nname}: up to {max(new)})")
    keys = [k for k in new[steps[-1]] if re.fullmatch(r"(dino|psnr)@-?[\d.]+", k) and k in ref[steps[-1]]]
    keys = sorted(keys, key=lambda k: (k.split("@")[0], float(k.split("@")[1])))
    print(f"ref {rname}\nnew {nname}\nnew - ref at equal steps:")
    print("step".rjust(7) + "".join(k.rjust(11) for k in ["val_loss", *keys]))
    for s in steps:
        row = [new[s].get("val_loss", float("nan")) - ref[s].get("val_loss", float("nan"))]
        row += [new[s][k] - ref[s][k] for k in keys]
        print(f"{s:7d}" + "".join(f"{v:+11.4f}" for v in row))
    last = steps[-a.last:]
    print(f"\nmean over the last {len(last)} common validations (steps {last[0]}-{last[-1]}):")
    verdict = []
    for k in keys:
        d = sum(new[s][k] - ref[s][k] for s in last) / len(last)
        thr = SIGNIFICANT[k.split("@")[0]]
        tag = "new ahead" if d > thr else "new behind" if d < -thr else "within noise"
        verdict.append(tag)
        print(f"  {k:10s} {d:+.4f}  {tag}")
    if all(v == "new behind" for v in verdict):
        print("verdict: new is behind on every score: a candidate to stop")
    elif all(v == "new ahead" for v in verdict):
        print("verdict: new is ahead on every score")
    else:
        print("verdict: mixed or within noise: keep training and decide on the test set")


if __name__ == "__main__":
    main()
