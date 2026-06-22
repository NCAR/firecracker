"""
Diagnostic terrain-channeling gain (Phase 6, Option 2 -- the graded "B-with-min" metric).

compute_channel_gain imposes the gap/Venturi speedup the level-lid prognostic core cannot produce:
it scans the cross-stream relief (perpendicular to the synoptic wind), grades each step's openness by
relief (clamp(1 - rise/height_scale, 0, 1)), shadows it outward (a wall hides everything beyond),
sums an open width per side, keeps the *more-open* side, and maps gain = clamp(radius/W_eff, 1,
gain_max). These tests call it directly on synthetic terrain (no env spin-up) and pin the properties
that motivated the graded metric: a taller wall and a closer wall both raise the gain continuously,
while a one-sided ramp or a single wall with the other side open stays ~1 (slope / mountain-base
immunity -- a valley needs confinement on *both* sides, which is what makes a channel).
"""

import numpy as np
import torch

from scenarios import make_config
from simulation import Simulation


GRID = 64
C = GRID // 2


def _sim(height_scale_m=100.0, gain_max=3.0, smooth_sigma=0.0) -> Simulation:
    """A channeling-enabled sim with smoothing off by default (crisp per-cell gain to assert on)."""
    return Simulation(make_config(GRID, channeling={
        "enabled": True, "radius_m": 300.0, "height_scale_m": height_scale_m,
        "gain_max": gain_max, "smooth_sigma": smooth_sigma,
    }))


def _gain(sim: Simulation, terrain: np.ndarray, ambient=(15.0, 0.0)) -> np.ndarray:
    """Gain field for a terrain under an x-wind (so the cross-stream scan runs along y/rows)."""
    t = lambda v: torch.tensor(float(v), dtype=sim.dtype, device=sim.device)
    terr = torch.as_tensor(terrain, dtype=sim.dtype, device=sim.device)
    g = sim.compute_channel_gain(terr, t(ambient[0]), t(ambient[1]))
    return g.cpu().numpy()


def _two_wall(width_cells: int, height_m: float) -> np.ndarray:
    """A flat channel of half-width `width_cells` either side of centre, walls (height_m) beyond.

    Terrain varies in y only (the cross-stream axis for an x-wind), so the channel runs along the
    wind -- the gap geometry. abs(row - C) <= width_cells is open floor; everything else is wall.
    """
    rows = np.abs(np.arange(GRID) - C)
    profile = np.where(rows <= width_cells, 0.0, height_m).astype(np.float64)
    return np.tile(profile[:, None], (1, GRID))


def test_flat_ground_gives_unity_gain():
    """No relief -> open both ways -> gain is exactly 1 everywhere."""
    g = _gain(_sim(), np.zeros((GRID, GRID), dtype=np.float64))
    np.testing.assert_allclose(g, 1.0, rtol=1e-12)


def test_taller_wall_gives_larger_gain():
    """Same channel width, a taller wall closes the channel more -> larger centreline gain."""
    short = _gain(_sim(height_scale_m=100.0), _two_wall(width_cells=6, height_m=50.0))
    tall  = _gain(_sim(height_scale_m=100.0), _two_wall(width_cells=6, height_m=150.0))
    assert tall[C, C] > short[C, C] + 0.1     # height grading: a 150 m wall boosts more than a 50 m one
    assert short[C, C] > 1.0                  # even a partial wall confines somewhat


def test_closer_wall_gives_larger_gain():
    """Same (fully closing) wall height, narrower channel -> walls are closer -> larger gain."""
    wide   = _gain(_sim(), _two_wall(width_cells=12, height_m=300.0))
    narrow = _gain(_sim(), _two_wall(width_cells=4,  height_m=300.0))
    assert narrow[C, C] > wide[C, C] + 0.1    # distance grading: closer walls pinch harder


def test_one_sided_ramp_gives_unity_gain():
    """A pure slope (rises one way, falls the other) is not a channel -> gain ~ 1.

    The downhill side stays fully open (rise <= 0 -> openness 1 -> open width = R), so the more-open
    side keeps W_eff = R and the gain is ~1 -- the worry that a gradient-based rule would wrongly
    accelerate wind on hillsides does not happen here.
    """
    ramp = np.tile(np.linspace(0.0, 600.0, GRID)[:, None], (1, GRID))
    g = _gain(_sim(), ramp)
    np.testing.assert_allclose(g[12:-12, C], 1.0, atol=1e-3)


def test_mountain_base_gives_unity_gain():
    """A single wall on one side with the other side open (a mountain base) -> gain ~ 1.

    Confinement on only one side is a slope/base, not a channel; the open side fixes W_eff = R, so
    the gain stays ~1 -- only a two-sided valley boosts.
    """
    rows = np.arange(GRID)
    profile = np.where(rows > C + 6, 800.0, 0.0).astype(np.float64)   # wall on the +y side only
    g = _gain(_sim(), np.tile(profile[:, None], (1, GRID)))
    np.testing.assert_allclose(g[C, C], 1.0, atol=1e-3)


def test_gain_is_capped_at_gain_max():
    """A tight, tall gap saturates the gain at gain_max (the continuity speedup is bounded)."""
    g = _gain(_sim(gain_max=3.0), _two_wall(width_cells=1, height_m=800.0))
    assert abs(g[C, C] - 3.0) < 1e-6
