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
  (stages 1-4 and cnn16-cos were dropped on 19 Nov: 128 px is closed; month plan, section 6)
  cnn16-cos     stage 1 with a cosine decay over 60k steps: does it reach the converged quality?
  ref16         CNN + diffusion refiner on the converged reference (option B); only when named:
                the 18 Nov sweep (g-s3) showed a refiner in the SD-VAE latent at 128 px cannot keep
                the CNN's semantics

256 px test (configs/gen/p256.yaml, same bandwidth ratio; only when named):
  dit256        blind 16-QAM DiT-S to convergence, the recipe of the 128 px g-v1 (results/gen/p256-conv)
  cnn256        blind 16-QAM CNN, pixel MSE, 40k steps: the budget of the 128 px reference g-cb2
  cnn256-ft     + 0.1 LPIPS for 10k steps from cnn256 (results/gen/p256-perc)
  dit256-s1     dit256 with seed 1;  dit256-repa  dit256 + REPA (0.5, depth 4); both to convergence
  dump256       reconstructions of the three for the analysis suite (results/gen/dump/p256; then
                scripts/gen_analysis/suite.py scripts/gen_analysis/specs/p256.json on the development PC)
The CNN is trained at the reference's budget, not to convergence: a DiT that loses to it loses to
a converged one too; only a DiT that wins needs the converged CNN (pre-registered 18 Nov):
  cnn256-conv     the cnn256 run continued from its checkpoint to convergence (patience 10, as the DiT)
  cnn256-conv-ft  + 0.1 LPIPS from it, to convergence; both in results/gen/p256-conv
  dump256-conv    reconstructions for scripts/gen_analysis/specs/p256-conv.json

usage (from the repo root, on the 5060 machine):
    python scripts/gen_5060.py cnn16 cnn16-ft  # only these stages
    python scripts/gen_5060.py --check         # environment check only

First time: a PyTorch build with Blackwell (sm_120) support, e.g.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt diffusers lpips torch-fidelity torchmetrics pyyaml scipy
(without torchmetrics the test sets run but FID is silently left out)
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
# Recipe calibration (plan of 19 Nov): the cnn16 run with a cosine decay over 60k steps. If its
# spread test lands within the seed spread of the converged constant-LR cnn16 (~200k steps), the
# SNR-placement study's CNN arms and a converged 256 px CNN can use it at a third of the cost.
CNN16_COS = [["--models", "cnn", "--arms", "blind", "--sanity-steps", "0", "--qam", "16",
              "--lr-schedule", "cosine", "--steps", "60000", "--out-dir", OUT, "--wandb-mode", "offline"]
             + TESTS[1]]
STAGES = {
    "cnn16": CNN16,
    "cnn16-cos": CNN16_COS,
    "cnn16-ft": CNN16_FT,
    "cnn": CNN,
    "cnn-ft": CNN_FT,
    "ref16": [REF16 + GEN_STEPS + t for t in TESTS],
}
P256 = ["--config", "configs/gen/p256.yaml", "--wandb-mode", "offline"]
DIT256 = P256 + ["--models", "dit", "--arms", "blind", "--qam", "16", "--steps", "150000", "--patience", "10",
                 "--out-dir", "results/gen/p256-conv"]
# The 256 px CNN needs about 12 GB at batch 64 (24 GB with LPIPS): each batch of 64 goes through
# in chunks of 32 (8 with LPIPS) whose gradients are added up (grad_accum, CNN only; the gradient
# matches the whole batch's to 0.03%). On the 5060: 1.18 batches/s, 0.59 with LPIPS.
CNN256_BASE = P256 + ["--models", "cnn", "--arms", "blind", "--qam", "16", "--sanity-steps", "0"]
CNN256 = CNN256_BASE + ["--out-dir", "results/gen/p256-perc", "--grad-accum", "2"]
CNN256_FT = CNN256 + ["--init-from", "results/gen/p256-perc/cnn_cbr48_qam16_blind-5to5_s0",
                      "--cnn-lpips-weight", "0.1", "--steps", "10000", "--tag", "ft", "--grad-accum", "8"]
# The pre-registered re-check if the DiT passes (18 Nov): the same CNN trained to convergence by
# the DiT's rule (constant LR, patience 10). The pixel stage continues the 40k-step run from its
# checkpoint (optimiser state and validation history kept), then the LPIPS stage to convergence.
CONV256 = "results/gen/p256-conv"
CNN256_SRC = "results/gen/p256-perc/cnn_cbr48_qam16_blind-5to5_s0"
# Capped at 140k steps on 21 Nov, the converged DiT's step count: at 132k the validation loss was
# still falling by about 0.05% per validation, so patience alone would have run on for hours.
CNN256_CONV = CNN256_BASE + ["--out-dir", CONV256, "--grad-accum", "2", "--steps", "140000", "--patience", "10"]
# LPIPS stage capped at 36k on 22 Nov (user): validation loss still fell about 0.15% per validation,
# DINOv2 on validation had flattened (+0.007 over the last 12k), patience would have run 10-20 h more.
CNN256_CONV_FT = CNN256_BASE + ["--out-dir", CONV256, "--init-from", f"{CONV256}/cnn_cbr48_qam16_blind-5to5_s0",
                                "--cnn-lpips-weight", "0.1", "--tag", "ft", "--grad-accum", "8",
                                "--steps", "36000", "--patience", "10"]
RUNS256 = ["results/gen/p256-conv/dit-S-v_cbr48_qam16_blind-5to5_s0",
           "results/gen/p256-perc/cnn_cbr48_qam16_blind-5to5_lpips0.1_ft_s0",
           "results/gen/p256-perc/cnn_cbr48_qam16_blind-5to5_s0"]
STAGES.update({
    "dit256": [DIT256 + GEN_STEPS + t for t in TESTS],
    # Paired with dit256 on this GPU (19 Nov): its seed 1 (the 256 px seed spread, for the seed
    # rule and for judging any 256 px variant) and REPA as at 128 px (g-v3: weight 0.5, depth 4).
    "dit256-s1": [DIT256 + ["--seeds", "1"] + GEN_STEPS + t for t in TESTS],
    "dit256-repa": [DIT256 + ["--repa-weight", "0.5", "--repa-depth", "4"] + GEN_STEPS + t for t in TESTS],
    "cnn256": [CNN256 + ["--steps", "40000"] + TESTS[1]],
    "cnn256-ft": [CNN256_FT + t for t in TESTS],
    # Reconstructions for the analysis suite (scripts/gen_analysis/specs/p256.json): the same 1,000
    # class-spread test images, channel and sampling seeds as the eval-spread scores, on this GPU.
    "dump256": [["scripts/gen_analysis/dump.py", "results/gen/dump/p256", "1000", "-5,0,5", *RUNS256,
                 "--base", "configs/gen/p256.yaml"]],
    "cnn256-conv": [["continue", CNN256_SRC, CONV256], CNN256_CONV + TESTS[1]],
    "cnn256-conv-ft": [CNN256_CONV_FT + t for t in TESTS],
    # A separate folder: the converged CNNs have the same run names as the 40k-step ones.
    "dump256-conv": [["scripts/gen_analysis/dump.py", "results/gen/dump/p256-conv", "1000", "-5,0,5",
                      RUNS256[0], f"{CONV256}/cnn_cbr48_qam16_blind-5to5_lpips0.1_ft_s0",
                      f"{CONV256}/cnn_cbr48_qam16_blind-5to5_s0", "--base", "configs/gen/p256.yaml"]],
})


def continue_run(src: str, out_dir: str) -> None:
    """Copy a finished run into out_dir without its history.json and test results, so run_gen.py
    resumes it from checkpoint.pt (step, optimiser and validation history) under a new step cap."""
    import shutil

    dst = R / out_dir / Path(src).name
    if (dst / "history.json").exists() or (dst / "checkpoint.pt").exists():
        print(f"{dst} already set up; skipped")
        return
    shutil.copytree(R / src, dst, ignore=shutil.ignore_patterns("history.json", "eval*"))
    print(f"copied {src} -> {dst}")


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
    names = args
    if not names:
        raise SystemExit(f"name the stages to run (month plan, section 6, says which); choose from {list(STAGES)}")
    bad = [n for n in names if n not in STAGES]
    if bad:
        raise SystemExit(f"unknown stages {bad}; choose from {list(STAGES)}")
    for n in names:
        for plan in STAGES[n]:
            t0 = time.time()
            if plan[0] == "continue":
                continue_run(*plan[1:])
                continue
            cmd = plan if plan[0].endswith(".py") else ["scripts/run_gen.py", *plan]  # a script, or a run_gen plan
            print(f"\n=== {n}: {' '.join(cmd)}", flush=True)
            subprocess.run([sys.executable, *cmd], cwd=R, check=True)
            print(f"=== {n} step done in {(time.time() - t0) / 3600:.2f} h", flush=True)
    print(f"\nall done: copy {OUT}/ back to the development PC")


if __name__ == "__main__":
    main()
