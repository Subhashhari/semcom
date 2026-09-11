"""End-to-end tests: train -> checkpoint -> evaluate -> analyse gates.

These use a tiny synthetic dataset and a tiny model, so they run in seconds while still
exercising the real `train()`, `evaluate_run()` and `analyse()` code paths. They catch the
integration bugs unit tests cannot: a checkpoint that does not reload, a config that does
not round-trip through the run directory, an evaluation that silently reports nothing.
"""

import json
from pathlib import Path

import pytest
import torch

from semcom.config import Config
from semcom.evaluate import evaluate_run, load_run
from semcom.train import build_model, evaluate, train


@pytest.fixture
def fake_cifar(monkeypatch):
    """Replace the CIFAR-10 loaders with a small in-memory dataset.

    Avoids a 170 MB download in CI and keeps these tests to a few seconds, while leaving
    every other part of the pipeline real.
    """
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    # Low-frequency images: compressible, so the model can actually learn something.
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 32), indexing="ij")
    phases = torch.arange(32).view(32, 1, 1, 1) * 0.3
    x = (torch.sin(torch.stack([xx, yy, xx * yy])[None] * 6.28 + phases) * 0.5 + 0.5).clamp(0, 1)
    ds = TensorDataset(x, torch.zeros(32, dtype=torch.long))

    def loaders(*a, **kw):
        return (
            DataLoader(ds, batch_size=8, shuffle=True, drop_last=True),
            DataLoader(ds, batch_size=8),
            DataLoader(ds, batch_size=8),
        )

    for module in ("semcom.train", "semcom.evaluate", "semcom.analyze_gates"):
        monkeypatch.setattr(f"{module}.cifar10_loaders", loaders, raising=False)
    return ds


def tiny_config(tmp_path: Path, **kw) -> Config:
    defaults = dict(
        hidden=8,
        c_out=8,
        batch_size=8,
        epochs=2,
        eval_every=1,
        eval_repeats=1,
        eval_snrs=[0.0, 10.0, 20.0],
        num_workers=0,
        amp=False,
        wandb=False,
        out_dir=str(tmp_path),
    )
    defaults.update(kw)
    return Config(**defaults)


ARMS = [
    dict(snr_adaptive=False, digital=False),
    dict(snr_adaptive=True, digital=False),
    dict(snr_adaptive=False, digital=True),
    dict(snr_adaptive=True, digital=True),
]


@pytest.mark.parametrize("arm", ARMS, ids=["BDJSCC", "ADJSCC", "DeepJSCC-Q", "ADJSCC-Q"])
def test_train_then_evaluate_roundtrip(arm, tmp_path, fake_cifar):
    cfg = tiny_config(tmp_path, **arm)
    summary = train(cfg)

    assert summary["arm"] == cfg.arm
    assert summary["best_val_psnr"] > 0
    # Reported so the writeup cannot imply protocol parity with a 1280-epoch reference.
    assert summary["epochs_trained"] <= cfg.epochs
    assert summary["epochs_configured"] == cfg.epochs

    run_dir = cfg.run_dir
    for f in ("config.yaml", "best.pt", "checkpoint.pt", "history.json"):
        assert (run_dir / f).exists(), f"missing {f}"

    result = evaluate_run(run_dir, use_wandb=False)
    assert set(result["psnr_by_snr"]) == {0.0, 10.0, 20.0}
    assert all(v > 0 for v in result["psnr_by_snr"].values())
    # Digital arms must certify the constellation invariant; analog arms have none.
    assert result["on_constellation"] is (True if arm["digital"] else None)

    saved = json.loads((run_dir / "evaluation.json").read_text())
    assert saved["arm"] == cfg.arm


def test_reloaded_model_reproduces_its_own_outputs(tmp_path, fake_cifar):
    """A checkpoint that loads but produces different outputs would corrupt every result."""
    cfg = tiny_config(tmp_path, snr_adaptive=True, digital=True)
    train(cfg)

    device = torch.device("cpu")
    model_a, _ = load_run(cfg.run_dir, device)
    model_b, _ = load_run(cfg.run_dir, device)

    x = torch.stack([fake_cifar[i][0] for i in range(4)])
    snr = torch.full((4,), 10.0)
    torch.manual_seed(0)
    out_a = model_a(x, snr)["x_hat"]
    torch.manual_seed(0)
    out_b = model_b(x, snr)["x_hat"]
    torch.testing.assert_close(out_a, out_b)


def test_training_resumes_from_checkpoint(tmp_path, fake_cifar, capsys):
    cfg = tiny_config(tmp_path, epochs=2)
    train(cfg)
    capsys.readouterr()

    train(cfg.replace(epochs=4))
    assert "resumed from epoch" in capsys.readouterr().out


def test_trained_digital_model_still_transmits_legal_symbols(tmp_path, fake_cifar):
    """Training moves the encoder's output distribution a long way.

    The unit tests check an untrained quantiser; this checks the property that makes the
    scheme standards-legal survives optimisation.
    """
    cfg = tiny_config(tmp_path, snr_adaptive=True, digital=True, modulation_order=16)
    train(cfg)
    model, _ = load_run(cfg.run_dir, torch.device("cpu"))

    x = torch.stack([fake_cifar[i][0] for i in range(8)])
    for snr in (0.0, 10.0, 20.0):
        z = model.transmit(x, torch.full((8,), snr))
        assert model.quantiser.is_on_constellation(z)


def test_evaluate_run_refuses_off_constellation_results(tmp_path, fake_cifar, monkeypatch):
    """The guard must actually fire, not just exist.

    Reporting numbers from a model that transmits illegal symbols would invalidate the
    entire comparison, so this path must raise rather than warn.
    """
    cfg = tiny_config(tmp_path, digital=True)
    train(cfg)

    monkeypatch.setattr(
        "semcom.evaluate.verify_transmitted_symbols", lambda *a, **kw: False
    )
    with pytest.raises(RuntimeError, match="off-constellation"):
        evaluate_run(cfg.run_dir, use_wandb=False)


def test_evaluate_run_without_checkpoint_raises(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.run_dir / "config.yaml")
    with pytest.raises(FileNotFoundError):
        evaluate_run(cfg.run_dir, use_wandb=False)


def test_evaluate_averages_over_repeats(tmp_path, fake_cifar):
    """More channel realisations must reduce variance, not change the expected value."""
    from torch.utils.data import DataLoader

    cfg = tiny_config(tmp_path, digital=False)
    model = build_model(cfg)
    loader = DataLoader(fake_cifar, batch_size=8)

    torch.manual_seed(0)
    spread_1 = [evaluate(model, loader, torch.device("cpu"), [5.0], 1)[5.0] for _ in range(5)]
    torch.manual_seed(0)
    spread_8 = [evaluate(model, loader, torch.device("cpu"), [5.0], 8)[5.0] for _ in range(5)]

    def rng(v):
        return max(v) - min(v)

    assert rng(spread_8) < rng(spread_1)


def test_gate_analysis_reports_both_patterns(tmp_path, fake_cifar):
    from semcom.analyze_gates import analyse as analyse_gates

    cfg = tiny_config(tmp_path, snr_adaptive=True, digital=True)
    train(cfg)

    result = analyse_gates(cfg.run_dir, snrs=(1.0, 10.0, 19.0), max_batches=2)
    assert len(result["selectivity_by_module"]) == 4
    assert len(result["snr_spread_by_module"]) == 4
    assert isinstance(result["pattern_2_snr_dependence_concentrates_early"], bool)
    assert (cfg.run_dir / "gate_analysis.json").exists()


def test_gate_analysis_refuses_non_adaptive_arms(tmp_path, fake_cifar):
    from semcom.analyze_gates import analyse as analyse_gates

    cfg = tiny_config(tmp_path, snr_adaptive=False)
    train(cfg)
    with pytest.raises(ValueError, match="no AF modules"):
        analyse_gates(cfg.run_dir, snrs=(1.0, 19.0), max_batches=1)
