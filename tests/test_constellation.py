"""Tests for the M-QAM constellation and soft-to-hard quantiser.

The invariant in `test_forward_output_is_on_constellation` is the one the entire
deployability claim rests on: if the forward pass ever emits an off-constellation value,
the scheme is not standards-legal and every comparison against a 5G baseline is void.
"""

import math

import pytest
import torch

from semcom.constellation import SoftToHardQuantiser, qam_constellation

ORDERS = [4, 16, 64, 256, 1024]


@pytest.mark.parametrize("order", ORDERS)
def test_constellation_shape_and_uniqueness(order):
    points = qam_constellation(order)
    assert points.shape == (order, 2)
    # All points distinct: the smallest pairwise distance must be strictly positive.
    dist = torch.cdist(points, points)
    dist.fill_diagonal_(float("inf"))
    assert dist.min() > 1e-6


@pytest.mark.parametrize("order", ORDERS)
def test_constellation_average_power_is_unit(order):
    points = qam_constellation(order, avg_power=1.0)
    assert points.pow(2).sum(dim=1).mean().item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("order", ORDERS)
def test_constellation_is_square_and_symmetric(order):
    points = qam_constellation(order)
    side = int(round(math.sqrt(order)))
    # A square QAM lattice has exactly `side` distinct levels per dimension...
    assert torch.unique(points[:, 0]).numel() == side
    assert torch.unique(points[:, 1]).numel() == side
    # ...and is symmetric about the origin, so the constellation sums to zero.
    assert points.sum(dim=0).abs().max().item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("order", [4, 16, 64])
def test_constellation_scales_with_avg_power(order):
    p1 = qam_constellation(order, avg_power=1.0)
    p4 = qam_constellation(order, avg_power=4.0)
    assert p4.pow(2).sum(dim=1).mean().item() == pytest.approx(4.0, abs=1e-4)
    torch.testing.assert_close(p4, p1 * 2.0, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("order", [6, 3, 0, 32])
def test_constellation_rejects_invalid_orders(order):
    # 32 is a power of two but not a perfect square, so not a square QAM lattice.
    with pytest.raises(ValueError):
        qam_constellation(order)


@pytest.mark.parametrize("order", ORDERS)
def test_forward_output_is_on_constellation(order):
    """THE critical invariant: transmitted values are exact alphabet members."""
    q = SoftToHardQuantiser(order)
    z = torch.randn(64, 128, 2) * 0.5
    out = q(z)

    assert out.shape == z.shape
    assert q.is_on_constellation(out)

    # Independent check, not routed through the helper under test. Deliberately avoids
    # torch.cdist: its matmul backend reports errors of ~5e-4 on values that are provably
    # exact constellation points, which is what made the first version of this test lie.
    flat = out.reshape(-1, 2).to(torch.float64)
    pts = q.points.to(torch.float64)
    gap = (flat.unsqueeze(1) - pts.unsqueeze(0)).pow(2).sum(-1).min(dim=1).values
    assert gap.max().item() == 0.0, "transmitted values must be bit-exact alphabet members"


def test_forward_selects_nearest_neighbour():
    q = SoftToHardQuantiser(16)
    # Feed the constellation points themselves, nudged slightly: each must map to itself.
    z = q.points.clone().unsqueeze(0) + torch.randn(1, 16, 2) * 1e-3
    out = q(z)
    torch.testing.assert_close(out.squeeze(0), q.points, rtol=1e-5, atol=1e-5)


def test_gradients_are_finite_and_nonzero():
    q = SoftToHardQuantiser(16)
    z = (torch.randn(8, 32, 2) * 0.5).requires_grad_(True)
    q(z).pow(2).sum().backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_soft_converges_to_hard_as_sigma_grows():
    """As sigma_q rises the surrogate approaches the true transmitted signal."""
    z = torch.randn(16, 64, 2) * 0.5
    gaps = []
    for sigma in (1.0, 10.0, 100.0):
        q = SoftToHardQuantiser(16, sigma_q_init=sigma)
        hard = q(z)
        # Recompute the soft value at this temperature to measure the surrogate gap.
        dist = torch.cdist(z.reshape(-1, 2), q.points).pow(2)
        soft = torch.softmax(-q.sigma_q * dist, dim=1) @ q.points
        gaps.append((soft - hard.reshape(-1, 2)).abs().mean().item())

    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[-1] < 0.05


def test_anneal_schedule_climbs_and_saturates():
    q = SoftToHardQuantiser(16, sigma_q_init=5.0, sigma_q_max=100.0, anneal_period=10)
    assert q.sigma_q.item() == pytest.approx(5.0)

    # Flat through the first period, since floor(t / period) is 0.
    for _ in range(9):
        q.step_anneal()
    assert q.sigma_q.item() == pytest.approx(5.0)

    # Then it climbs, and never exceeds the cap.
    for _ in range(500):
        q.step_anneal()
    assert q.sigma_q.item() == pytest.approx(100.0)


def test_kl_is_zero_for_uniform_usage_and_positive_otherwise():
    q = SoftToHardQuantiser(16, sigma_q_init=50.0)

    # Feeding each constellation point once gives near-uniform usage.
    q(q.points.clone().unsqueeze(0))
    assert q.kl_to_uniform().item() == pytest.approx(0.0, abs=1e-3)

    # Collapsing everything onto one point is maximally non-uniform.
    q(q.points[0].view(1, 1, 2).expand(1, 256, 2).clone())
    assert q.kl_to_uniform().item() > math.log(16) * 0.5


def test_kl_is_differentiable():
    q = SoftToHardQuantiser(16)
    z = (torch.randn(4, 32, 2) * 0.5).requires_grad_(True)
    q(z)
    q.kl_to_uniform().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_kl_before_forward_raises():
    with pytest.raises(RuntimeError):
        SoftToHardQuantiser(16).kl_to_uniform()


def test_rejects_wrong_trailing_dimension():
    q = SoftToHardQuantiser(16)
    with pytest.raises(ValueError):
        q(torch.randn(8, 32, 3))


def test_quantisation_error_shrinks_with_order():
    """Higher M is a finer codebook, so the displacement of z shrinks monotonically.

    This is the mechanism behind DeepJSCC-Q approaching analog DeepJSCC at high M, and
    the reason M is a *resolution* knob here rather than the *rate* knob it is in 5G.
    """
    torch.manual_seed(0)
    # Power-normalised latents, matching what the model actually feeds the quantiser.
    z = torch.randn(32, 256, 2)
    z = z / z.pow(2).sum(dim=(1, 2), keepdim=True).sqrt() * math.sqrt(256)

    errors = []
    for order in ORDERS:
        q = SoftToHardQuantiser(order, sigma_q_init=100.0)
        errors.append((q(z) - z).pow(2).mean().item())

    assert errors == sorted(errors, reverse=True), dict(zip(ORDERS, errors))
