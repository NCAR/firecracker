# Phase 6 leg (b) investigation — gap/Venturi winds (session recovery notes)

**Date:** 2026-06-21 · **Branch:** `sim-physics` · **Last commit:** `1410c70` (leg a).
**Status:** investigation only; working tree has uncommitted changes (see below). NOT committed.
This file exists so the session's findings survive a restart. Decision is pending the user.

---

## What we set out to do

Implement Phase 6 leg (b): surface-layer terrain blocking → gap/Venturi winds. Plan (approved)
was in `~/.claude/plans/crispy-juggling-island.md`: add face **porosity** blocking to the
two-layer surface layer so realistic relief blocks it and funnels a gap jet, flipping the strict
xfail `test_narrow_channel_does_not_speed_up_wind` into a passing Venturi test.

User locked the mechanism choice: **Approach A (face porosity + pressure feedback)**, NOT the
doc's literal "explicit perpendicular flux spill" (Approach B). Porosity scales each surface-layer
face flux by openness `θ = 1 − f`, `f = clamp(Δb/h_s, 0, 1)`; funnelling was meant to emerge via
the pressure feedback (blocked air dams → `b_s` rises → `−g'_s∇b_s` diverts it to the gap).

## What was implemented (uncommitted, in the working tree)

All in `src/simulation.py` + `tests/test_equilibrium.py`:

1. `_advect_periodic(phi, dispx, dispy, open_x=None, open_y=None)` — optional per-face openness;
   `None` ⇒ byte-identical to before. Scales each signed face flux by openness (conservation
   preserved: scaled flux still added to one neighbour, subtracted from the other).
2. `_face_openness(self, terrain, h_s) -> (open_x, open_y)` — `θ = clamp(1 − Δb/throat, 0, 1)`,
   `Δb = |z_{i+1}−z_i|`, `throat = min(h_s_i, h_s_{i+1}).clamp(min=1e-9)`. Periodic, rank-agnostic.
3. `_advance_layer(..., open_x=None, open_y=None)` — threads openness into all four advected
   quantities (m, energy, mom_x, mom_y).
4. `step_dynamics` two-layer branch: each substep computes `open_x,open_y =
   _face_openness(s.terrain, eta_s)` and passes them to the **surface-layer** `_advance_layer`
   call only (upper layer unblocked).
5. Test: converted the xfail to `test_narrow_channel_speeds_up_wind` (two-layer, asserts channel
   surface speed > ambient) + a single-layer companion `test_single_layer_channel_does_not_speed_
   up_wind` (asserts ≤ ambient). Dropped the now-unused `pytest` import.

**Suite state: 55 passed, 1 failed** — only the new Venturi test fails (channel 14.29 < 15).
The porosity blocking itself is correct and conservation-safe (closed-core two-layer test still
passes; single-layer path byte-identical).

## What we found (the core results)

Porosity blocking works mechanically but **does not produce a gap jet > ambient**. Root causes,
established by experiment (transverse ridge + gap, ambient 15 m/s, ~800 steps):

1. **`si_channel` geometry can't show a Venturi at all.** Its walls run *parallel* to the wind —
   a straight constant-width channel with uniform inflow doesn't accelerate above free-stream in
   conservative shallow water. Swept knobs to zero drag / high g'_s → channel maxed at **14.93**.
   (The original xfail's premise is geometrically wrong for blocking-driven gap winds.)

2. **Level-interface rest state makes the surface layer too thick to block (root cause #1).**
   At rest `h_s = surface_lid − z ≈ 700 m` over low ground (deepens to ~900 m as air converges).
   Thick → high wave speed `√(g'_s h_s)` → **subcritical (Fr ≈ 0.8–0.94)** → no hydraulic
   acceleration. Lowering `g'_s` pushes it supercritical (Fr up to 5) but the gap **still** never
   exceeds 15 — because nothing concentrates flux through the gap.

3. **Realistic ridges don't block the thick layer; tall ones blow up.** A 300–500 m ridge is
   shorter than the ~700 m layer (`f ≈ 0.25–0.5`, leaks — air rides over → no gap concentration).
   Ridges tall enough to block (≥800 m) make the over-wall surface layer nearly massless
   (`m_s ≈ 0.9`), so `−g'_s∇b_s` across the steep face explodes `u = mom/m` → **dmax ~90 m/s**.

4. **Thin reduced-gravity layer (the proposed fix) helps but is NOT sufficient.** Prototyped a
   throwaway: constant rest thickness `H_s` (≈150–200 m), force by `−g'_s∇(h_s − H_s)` (no
   drainage at rest, terrain enters via blocking not forcing). Reusing `Simulation` helpers
   (`_advance_layer`, `_face_openness`, sponge). Results:
   - Layer **stays thin** (h ≈ 150–240 m vs 700) and **supercritical** — good.
   - With a corrected dry-cell limiter (`wet = clamp(h/h_dry, 0, 1)`, gate velocity only in
     genuinely dry cells), a **windward dam forms** (h piles to ~880–1150 m windward vs ~330 m
     lee) — blocking/damming now works.
   - **But the gap flow is still 7–14 m/s, never > 15**, for every `H_s`/`g'_s`/gap-width/wall
     tried. The gap is a hydraulic exchange `√(2 g'_s Δh) ≈ 10 m/s` — comparable to but below the
     15 m/s free stream (would need head Δh > ~1100 m; dam only reached ~550 m).
   - Tall/steep walls (≥1500 m) **still blow up** (dmax 55) even with the basic dry-cell limiter.

   (First dry-cell attempt used `wet = h/(h+h_dry)` applied every substep — a bug: compounds into
   a huge momentum sink, killed all flow to ~0. The `clamp(h/h_dry,0,1)` form is the right one.)

## Diagnosis

Gap/Venturi winds in this sponge-bounded shallow-water framework are a **research-grade problem**,
not a single leg. They need *all* of: a thin blockable layer, production wetting/drying for
stability over real relief, Froude-regime calibration, and — structurally missing — a **sustained
synoptic pressure gradient across the barrier** to ram air through the gap. The open-boundary
sponge injects ambient wind but bleeds off the dam a gap jet depends on. The level interface is a
real contributor (too-thick layer) but fixing it alone doesn't deliver the jet.

The porosity blocking is still genuinely useful (correct windward damming, slows flow over
obstacles = real surface-wind realism) — it just doesn't, by itself, make a super-ambient jet.

## Strategic reframing (the key takeaway)

Phase 6 conflated **two separable goals** with very different difficulty:

- **Goal 1 — gap/Venturi winds.** Hard (see above). Question whether it's even needed for an RL
  training env vs a plausible terrain-responsive wind.
- **Goal 2 — the fire reads the near-surface wind (not the column mean).** *This is the one that
  matters for a fire sim, and it's cheap* — no second prognostic layer needed.

## Recommendation (pending user decision — user chose "back to the drawing board")

1. **Goal 2, cheaply:** derive a **diagnostic near-surface wind** from the existing single-layer
   wind (log-law / surface-layer reduction `u_sfc = u·f(z₀,…)`, optionally terrain-modulated) and
   point `update_fire`/oxygen/plume at it. A few lines on the well-behaved single-layer core.
2. **Goal 1, pragmatically:** either a **parameterized terrain-channeling factor** (operational
   fire-model style), or accept gap winds as a **documented limitation**. Reserve the heavy routes
   (diagnostic **mass-consistent / WindNinja-style** Poisson wind, or full prognostic two-layer
   wetting-drying SWE) only if gap winds become a research priority of their own.
3. **Shelve the two-layer prognostic core (leg a):** likely the wrong vehicle for both goals.
   Keep it gated-off/dormant, or revert it — user's call.

## Open questions for the user (asked at end of session, unanswered — went to bed)

- (a) Does decoupling the two goals match intent?
- (b) How much are *real* gap winds needed for the RL env vs a plausible terrain-responsive wind?
- (c) Revert the working-tree changes back to the clean leg-(a) baseline (`1410c70`), or keep them?

## To restore context next session

- Read this file, `docs/units.md` (Phase 6 section, lines ~505–580), and `src/simulation.py`
  (`step_dynamics`, `_advance_layer`, `_advect_periodic`, `_face_openness`).
- Working tree: `git status` / `git diff` shows the uncommitted leg-(b) attempt described above.
- Approved (now superseded) plan: `~/.claude/plans/crispy-juggling-island.md`.
- Throwaway prototypes were run inline via `python -c` (not saved); the numerical results above
  are the distilled outcome.
