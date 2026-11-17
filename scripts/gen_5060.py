"""Training queue for the RTX 5060: the runs that must reach convergence and are too long for
one 12-hour Kaggle session. Runs the stages in order, each as scripts/run_gen.py; safe to stop and
start again (finished runs are skipped, unfinished ones resume from checkpoint.pt).

Every run here trains to convergence (--patience: stop after 10 validations, i.e. 20k steps,
without a 0.2% val-loss gain; --steps is only a cap) at a constant learning rate, and is scored on
the same test sets as the Kaggle runs. Outputs go to results/gen/p128-conv/.

  1 cnn16       blind 16-QAM CNN, pixel MSE (the reference's first stage)
  2 cnn16-ft    + 0.1 LPIPS from stage 1's best weights: the converged reference
  3 cnn         blind analog CNN, pixel MSE
  4 cnn-ft      + 0.1 LPIPS: the converged analog reference
  ref16         CNN + diffusion refiner on the converged reference (option B); only when named:
                the 18 Nov sweep (g-s3) showed a refiner in the SD-VAE latent at 128 px cannot keep
                the CNN's semantics

usage (from the repo root, on the 5060 machine):
    python scripts/gen_5060.py                 # the default stages in order
    python scripts/gen_5060.py cnn16 cnn16-ft  # only these stages
    python scripts/gen_5060.py --check         # environment check only

First time: a PyTorch build with Blackwell (sm_120) support, e.g.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt diffusers lpips torch-fidelity pyyaml scipy
The Imagenette data (about 0.3 GB) downloads on first use; the cache needs about 5 GB.

Bring back: copy results/gen/p128-conv/ (with best.pt, about 0.3 GB per generative run) into the
same path on the development PC."""
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
OUT = "results/gen/p128-conv"
CONV = ["--patience", "10", "--out-dir", OUT, "--wandb-mode", "offline"]
TESTS = [  # the test sets every compared run is scored on (as the Kaggle jobs)
    ["--eval-snrs", "-5", "0", "5", "--evaluate"],
    ["--eval-subset", "spread", "--eval-suffix", "spread", "--eval-snrs", "-5", "0", "5", "--evaluate"],
    ["--eval-subset", "spread", "--eval-suffix", "spread-curve", "--eval-snrs", "-10", "-7.5", "-2.5", "2.5",
     "7.5", "10", "--evaluate"],
]
GEN_STEPS = ["--sample-steps", "10"]  # sampling steps for the generative runs' test sets


def cnn(qam: bool) -> list[list[str]]:
    base = ["--models", "cnn", "--arms", "blind", "--sanity-steps", "0"] + (["--qam", "16"] if qam else [])
    name = "cnn_cbr48_qam16_blind-5to5_s0" if qam else "cnn_cbr48_blind-5to5_s0"
    pixel = base + ["--steps", "300000"] + CONV
    ft = base + ["--init-from", f"{OUT}/{name}", "--cnn-lpips-weight", "0.1", "--tag", "ft",
                 "--steps", "150000"] + CONV
    return [pixel + TESTS[1]], [ft + t for t in TESTS]


CNN16, CNN16_FT = cnn(True)
CNN, CNN_FT = cnn(False)
REF16 = ["--models", "refiner", "--arms", "blind", "--qam", "16",
         "--refine-from", f"{OUT}/cnn_cbr48_qam16_blind-5to5_lpips0.1_ft_s0", "--steps", "200000"] + CONV
STAGES = {
    "cnn16": CNN16,
    "cnn16-ft": CNN16_FT,
    "cnn": CNN,
    "cnn-ft": CNN_FT,
    "ref16": [REF16 + GEN_STEPS + t for t in TESTS],
}
DEFAULT = ["cnn16", "cnn16-ft", "cnn", "cnn-ft"]


def check() -> None:
    import torch

    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: install a CUDA build of PyTorch (see the docstring)")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    print(f"GPU {name}, compute capability {cap}, memory {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    if f"sm_{cap[0]}{cap[1]}" not in arch:
        raise SystemExit(f"this PyTorch build has no kernels for sm_{cap[0]}{cap[1]} (has {arch}); "
                         "install the cu128 build (see the docstring)")
    x = torch.randn(256, 256, device="cuda")
    assert torch.isfinite(x @ x).all()
    print("environment OK")


def main() -> None:
    args = sys.argv[1:]
    check()
    if "--check" in args:
        return
    names = args or DEFAULT
    bad = [n for n in names if n not in STAGES]
    if bad:
        raise SystemExit(f"unknown stages {bad}; choose from {list(STAGES)}")
    for n in names:
        for plan in STAGES[n]:
            t0 = time.time()
            print(f"\n=== {n}: run_gen.py {' '.join(plan)}", flush=True)
            subprocess.run([sys.executable, "scripts/run_gen.py", *plan], cwd=R, check=True)
            print(f"=== {n} step done in {(time.time() - t0) / 3600:.2f} h", flush=True)
    print(f"\nall done: copy {OUT}/ back to the development PC")


if __name__ == "__main__":
    main()
