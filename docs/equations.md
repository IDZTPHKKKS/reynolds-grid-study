# Equations and Conventions — Forced 2D Navier–Stokes Benchmark

**Status**: authoritative reference for every physical and numerical
convention in this repository. Conventions were chosen from first principles
for this project; where a choice coincides with a standard textbook or
literature convention this is noted. Nothing here was copied silently from
another codebase — every sign, normalization and truncation is stated and
verified numerically (see `docs/solver_validation.md`).

**Purpose of this benchmark**: ground-truth data for cross-Reynolds
generalization of neural PDE solvers, following the setting used by the
cross-Reynolds neural-PDE literature (forced 2D Navier–Stokes / Kolmogorov
flow; e.g. Shi 2026, Srinivasan et al. 2024, Pawar et al. 2022, Subel et al.
2023). The setting is deliberately scoped to forced 2D turbulence with a
forward enstrophy cascade (Kraichnan-type), where small-scale universality
arguments are physically defensible — the scoping caveat of Girimaji
(2023/24), i.e. that large-scale quantities are *not* Kolmogorov-universal,
is respected by construction (the benchmark provides the similarity-relevant
small-scale fields; it does not claim large-scale universality).

---

## 1. Governing equations

Incompressible constant-density Navier–Stokes with a body force, on the
doubly-periodic domain Ω = [0, Lx) × [0, Ly), Lx = Ly = 2π:

    ∂u/∂t + (u·∇)u = −∇p + ν∇²u − σu + f,        ∇·u = 0

* `u = (u, v)`: velocity. `p`: kinematic pressure. `ν`: kinematic viscosity.
* `σ ≥ 0`: coefficient of linear (Rayleigh/Ekman) drag. Uniform drag absorbs
  inverse-cascade energy at large scales; without it, forced 2D turbulence on
  a small periodic box drifts toward a box-scale condensate and does not
  reach a clean statistically stationary state (classic result; verified in
  our own pilots). Drag is standard in forced-2D studies. It is diagonal in
  Fourier space and treated exactly by the exponential integrators.
* Working variables: the scalar vorticity ω = ∂v/∂x − ∂u/∂y (2D: the only
  nonzero component of ∇×u is the z-component; **sign convention** ω_z as
  defined). Taking the curl eliminates pressure:

      ∂ω/∂t + u·∇ω = ν∇²ω − σω + F_ω,     F_ω = (∇×f)_z = ∂f_y/∂x − ∂f_x/∂y

* The nonlinear term is evaluated in **conservative (divergence) form**
  `−∇·(uω)` — algebraically identical to `−u·∇ω` for incompressible flow,
  and it makes the k = 0 Fourier mode of the advection term exactly zero
  (mean vorticity conservation).

**Inviscid, unforced double conservation law (2D-specific).** With ν=σ=0 and
f=0, the vorticity equation `∂ω/∂t + u·∇ω = 0` conserves **both** kinetic
energy `E = ½⟨u²+v²⟩` and enstrophy `Ω = ½⟨ω²⟩` exactly — unlike 3D, where
only energy survives. Energy conservation follows from the pointwise
kinematic identity `u·∇ψ ≡ 0` (u is by construction the perpendicular
gradient of ψ); enstrophy conservation follows from the integral identity
`∫ω(u·∇ω)dA = ∫u·∇(ω²/2)dA = 0` (periodicity + ∇·u=0). This is the reason
forced 2D turbulence needs an explicit dissipation mechanism (§ here: ν and
σ) to reach any statistically stationary state at all — the undissipated
dynamics simply redistributes E and Ω among scales (the inverse energy /
forward enstrophy cascade) without ever damping either. Both invariants are
conserved by the *discretized* (dealiased, Galerkin-truncated) system too,
to the same near-machine-precision level as the continuous PDE — verified by
`scripts/verify_solver.py` Group I; see `docs/audit_report.md` for a bug this
check caught (fixed) in the dealiasing truncation.

## 2. Streamfunction conventions

    u = +∂ψ/∂y,   v = −∂ψ/∂x,   hence  ω = −∇²ψ.

In Fourier space (see §3 for conventions): `ω̂ = +k²ψ̂`, so

    ψ̂ = ω̂/k²,      û = i k_y ψ̂ = i k_y ω̂ / k²,      v̂ = −i k_x ψ̂ = −i k_x ω̂ / k²,

with the k = 0 mode set to zero (the periodic Poisson solvability condition
⟨ω⟩ = 0 holds exactly: mean vorticity is conserved and zero for our ICs and
forcing). This reconstruction makes ∇·u = 0 hold to machine precision —
incompressibility is structural, not approximate.

## 3. Fourier conventions

* Fields are real, stored as arrays of shape `(Ny, Nx)`: axis 0 = y, axis 1 = x.
* Transform: `numpy.fft.rfft2` with default ("backward", i.e. unnormalized
  forward / orthonormal-inverse) normalization:

      f̂(k) = Σ_{i,j} f(y_i, x_j) exp(−i(k_y y_i + k_x x_j)),   f = irfft2(f̂).

* Wavenumbers in physical units: `k_x = (2π/Lx)·n_x` with integer mode
  numbers `n_x ∈ {0,…,Nx/2}` (rfft layout), `k_y = (2π/Ly)·n_y` with
  `n_y ∈ {0,…,Ny/2−1, −Ny/2,…,−1}`. On the 2π-domain, `k = n` exactly.
* Derivatives: ∂/∂x ↔ `i k_x`, ∂/∂y ↔ `i k_y`, ∇² ↔ `−k²`.
* Domain average of a grid function equals its mean (equal-area cells);
  Parseval for the rfft layout:

      ⟨f²⟩ = Σ_k w_k |f̂(k)|² / (Nx Ny)²,

  with weight `w_k = 2` for modes with `0 < n_x < Nx/2` (each represents a
  ±pair) and `w_k = 1` on the `n_x = 0` and `n_x = Nx/2` columns. Verified
  against physical-space means to 1e-12.

## 4. Pseudo-spectral discretization and dealiasing (2/3 rule)

The discrete system is a **Galerkin truncation**: only modes with
`|n_x| ≤ n_max` and `|n_y| ≤ n_max` exist, where

    n_max = ⌊(N − 1)/3⌋.

Consequences:

* Guaranteed-resolved wavenumber `k_max = (2π/L)·n_max` per direction
  (e.g. N=64 → k_max = 21; N=128 → 42; N=256 → 85 on the 2π domain — these
  three values are **unchanged** by the formula above; see the note below).
* Quadratic products (u·ω) are computed on the full N-point physical grid
  from truncated inputs, then projected back. This is **exactly alias-free**
  provided `N > 3·n_max` **strictly**: a product of two modes with
  `|n| ≤ n_max` has `|n| ≤ 2·n_max`; if this exceeds the Nyquist limit N/2 it
  aliases to `N − 2·n_max`, which must exceed `n_max` (lie outside the kept
  band) for no contamination. (Verified by an explicit convolution test
  against the exact Galerkin product, by a worst-case Nyquist-crossing test,
  and — the strongest test — by confirming the discrete enstrophy-
  conservation identity `⟨ω, N_adv(ω)⟩ = 0` to machine precision for an
  inviscid, unforced state; see `docs/audit_report.md`.)

  **Off-by-one at N divisible by 3 (found and fixed during audit).** The
  naive `n_max = ⌊N/3⌋` satisfies `N > 3·n_max` strictly whenever N is *not*
  a multiple of 3 — true for all three production grids (64, 128, 256) — but
  gives `n_max = N/3` exactly (non-strict) when N *is* a multiple of 3 (e.g.
  48, 96, 192), letting a worst-case product alias exactly onto the kept
  boundary mode. This silently contaminated the discrete enstrophy budget
  (~1e-3 relative error per step; energy was unaffected — it is protected by
  the stronger, purely kinematic pointwise identity `u·∇ψ ≡ 0`, which does
  not depend on the truncation boundary at all) while running without error
  or warning. `n_max = ⌊(N−1)/3⌋` is the general, always-strict formula; it
  equals `⌊N/3⌋` for every N not divisible by 3, so re_low/re_mid/re_high are
  numerically identical to before this fix.
* The state is re-projected onto the truncation after every time step, and
  its k = 0 mode is set to zero (exactly conserved quantities).
* Linear terms and the forcing need no dealiasing (they act mode-by-mode).

## 5. Forcing (Kolmogorov flow)

    f(x, y) = ( A sin(k_f y),  0 ),    k_f = (2π/Ly)·n_f,  n_f = 4 (integer),

* **Forcing convention**: monochromatic force along x varying in y, exactly
  representable on the grid and inside the truncation (n_f ≤ ⌊Ny/3⌋).
  Fourier coefficients are constructed *exactly* (support = {(n_y,n_x) =
  (±n_f, 0)}; amplitudes from the closed-form DFT of sine/cosine), so the
  forcing injects energy at k_f and only at k_f, at machine precision.
* Vorticity forcing: `F_ω = −A k_f cos(k_f y)`.
* Energy injection rate `P = ⟨u·f⟩`; enstrophy injection `⟨ω F_ω⟩ = k_f² P`
  exactly (single-wavenumber forcing identity; verified numerically).
* Laminar exact steady solution (with drag σ):

      u_lam = (A/(ν k_f² + σ)) sin(k_f y) e_x,  ω_lam = −(A k_f/(ν k_f²+σ)) cos(k_f y).

  It is linearly unstable for laminar Reynolds number `Re_lam ≡ U_lam/(ν k_f)`
  beyond ≈ √2 (Meshalkin & Sinai 1961). All regimes here have
  `Re_lam ≫ √2`, so the flows are unsteady/chaotic. The laminar state is used
  in validation as an end-to-end convention check (solver must relax to it
  from a perturbed state when `Re_lam < √2`).

## 6. Time integration

The semi-discrete system in Fourier space:

    dω̂/dt = L ω̂ + N(ω̂),   L = −(ν k² + σ) (diagonal),   N = −(∇·(uω))̂ + F̂_ω,

with N dealiased and N(0)=0. Three fourth-order schemes are implemented:

1. **RK4** — classical explicit RK4 on the full RHS (linear part explicit).
2. **IFRK4** — integrating-factor RK4: RK4 in `w = e^{−Lt}ω̂`; linear part
  exact. Stage derivation is given in `solver/integrator.py`.
3. **ETDRK4** (default) — Cox & Matthews (2002) exponential
  time-differencing RK4 in the implementation form of Kassam & Trefethen
  (2005). With `v = hL`, `E = e^v`, `E2 = e^{v/2}`:

       a  = E2 u + Q N(u),          Q  = (h/2) φ₁(v/2)
       b  = E2 u + Q N(a)
       c  = E2 a + Q (2N(b) − N(u))
       u⁺ = E u + f₁ N(u) + f₂ (N(a)+N(b)) + f₃ N(c),
       f₁ = h(φ₁ − 3φ₂ + 4φ₃),   f₂ = 2h(φ₂ − 2φ₃),   f₃ = h(4φ₃ − φ₂),

  with φ-functions `φ_j(z) = (e^z − Σ_{m<j} z^m/m!)/z^j` evaluated stably
  (Taylor series for |z| < 0.5, closed forms otherwise). Properties, all
  verified numerically: (i) exact RK4 reduction as v→0; (ii) exactness for
  time-independent N (`f₁ + 2f₂ + f₃ = h φ₁(v)`); (iii) 4th-order
  convergence on manufactured problems and on the forced system. During
  development, alternative stage forms (different c-stage coefficients)
  were tested and rejected — they converge at 2nd order only
  (see `docs/solver_validation.md`).

**Stability/accuracy of stepping**: the advective spectral Courant number is

    C = dt · max_{x,y} ( |u| k_max,x + |v| k_max,y ),

with RK4-class stability boundary ≈ 2.8. Production configs keep
`C ≲ 0.5` (measured max C is recorded in every dataset file). The diffusive
stiffness `ν k²_max dt` is ≪ 1 in all regimes (e.g. ≈ 0.02–0.07), so the
exponential schemes' stiff robustness is a safety margin rather than a
necessity at these dt; all three schemes cross-validate to < 1e-6 relative
difference over multi-unit integrations. `scripts/verify_solver.py` Group J
confirms this stability boundary is physically real (a run with
`ν k²_max dt ≈ 176` for explicit RK4 genuinely diverges) and not just an
unverified docstring claim.

**Divergence safety net.** `NS2D.run` aborts with `RuntimeError` if the
advective Courant number exceeds `time.fail_courant` (default 2.0) at any
diagnostic check, *and* — since a Python/numpy comparison against NaN is
always `False`, which would otherwise let a state that overflowed to NaN
between two checks return silently — if any of E, enstrophy, or the Courant
number is non-finite. The latter matters because `scripts/generate_dataset.py`
checks diagnostics far less often than every `dt` (`min(dt_sample, 0.5)` vs.
a `dt` of a few 1e-3), so a divergence could previously overflow to NaN
strictly between checks and be missed entirely (bug found and fixed during
audit; see `docs/audit_report.md`, and the regression test
`tests/test_stability.py::test_nan_state_is_caught_even_with_coarse_diagnostics_interval`).

## 7. Diagnostics (definitions)

With ⟨·⟩ the domain (grid) average:

* Kinetic energy `E = ½⟨u² + v²⟩`; rms velocity `u_rms = √(2E)`.
* Enstrophy `Ω = ½⟨ω²⟩`; palinstrophy `Pal = ½⟨|∇ω|²⟩`.
* Energy dissipation `ε = 2νΩ + 2σE` (split as `ε_visc = 2νΩ`,
  `ε_drag = 2σE`). Identity used: `⟨S:S⟩ = ½⟨ω²⟩` for 2D incompressible
  periodic flow (S = strain-rate tensor), so `2ν⟨S:S⟩ = 2νΩ`. Pointwise
  `ε(x) = 2ν S:S` differs from `νω²(x)` but has the same mean; both are
  provided.
* Enstrophy dissipation `ε_ω = 2ν Pal + 2σΩ`.
* Budgets (verified to < 0.2% in chaotic runs):

      dE/dt  = P − ε,           P = ⟨u·f⟩
      dΩ/dt  = ⟨ω F_ω⟩ − ε_ω.

* Shell energy spectrum `E(k_n)`: shells collect integer mode radius
  `√(n_x²+n_y²) ∈ [n−½, n+½)`; `Σ_n E(k_n) = E` (Parseval).
* Gradient fields (physical space): S components; invariants
  `Q = ¼ω² − ½S:S` and `R = −det(∇u)` (2D analogues of the velocity-gradient
  invariants used in the Q–R collapse literature, cf. Fukami et al. 2024);
  local dissipation densities `ε(x) = 2νS:S` (K41 form) and
  `ζ(x) = ν|∇ω|²` (2D enstrophy form); local scales

      η_ε(x) = (ν³/ε(x))^{1/4}   (K41 local length scale),
      η_ζ(x) = (ν³/ζ(x))^{1/6}  (2D/Kraichnan local dissipation length).

  These fields are recomputable from stored vorticity snapshots; they exist
  to support the local-similarity methodology and collapse diagnostics
  planned for the ML stage (Agdestein & Sanderse 2026; Fukami et al. 2024;
  Bailey et al. 2009).

## 8. Reynolds numbers (all measured, none free-label)

* **Primary regime parameter**: `Re_f = u_rms/(ν k_f)` — large-scale
  Reynolds number based on the forcing wavenumber and the *measured*
  stationary rms velocity. Because the velocity scale is **calibrated**
  (forcing amplitude A adjusted until stationary `u_rms = target_u_rms`
  within tolerance), and `ν` is then set by construction
  `ν = target_u_rms/(target_re_f · k_f)`, the measured Re_f lands on the
  target by construction, and Re ratios across regimes equal ν ratios.
* `Re_lam = U_lam/(ν k_f)` with `U_lam = A/(νk_f²+σ)` (control-parameter
  definition; instability threshold ≈ √2).
* `Re_L = u_rms L/ν` (domain scale), `Re_λ = u_rms λ/ν` with the
  enstrophy-based 2D Taylor microscale `λ = √(E/Ω)`.
* Normalized dissipation `c_ε = ε/(u_rms³ k_f)`; 2D mean dissipation
  wavenumber `k_d = (ε_ω/ν³)^{1/6}` (total enstrophy dissipation, viscous +
  drag).

**Resolution criterion**: `k_max ≥ 2 k_d` in every regime (checked at
generation time and recorded; spectral tails additionally decay ≥ 5 decades
below the forcing peak — measured and recorded).

## 9. Initial conditions, burn-in, stationarity

* IC: band-limited random vorticity with shell amplitudes following
  `E_ic(n) ∝ n³ exp(−n/n_cut)`, `n_cut = 2k_f` (mode number 8), k = 0
  removed, truncated to the Galerkin set, rescaled so the initial `u_rms`
  equals the target velocity scale exactly. Deterministic from the seed.
* Burn-in: evolve in chunks; after a minimum burn time, test the trailing
  window for: relative split-half drift of E < `drift_e`, of Ω < `drift_z`,
  and |⟨P⟩−⟨ε⟩|/⟨P⟩ < `balance`. Production data start only after all three
  hold. The **production–dissipation balance is the physically authoritative
  criterion**: it is the direct statement that the flow is not systematically
  gaining or losing energy, converges fast (few time units), and is largely
  insensitive to which "phase" of any slower intermittent oscillation the
  window happens to sample. The E/Ω split-half drift tests are a secondary
  sanity check against secular drift (e.g. large-scale condensate growth);
  they require a window long enough to average over the flow's natural
  correlation time, which is **regime-dependent**.
* **Regime-dependent window/thresholds (found during audit; see
  `docs/audit_report.md` Finding 4).** re_low (Re_f=10) equilibrates quickly
  and cleanly with a short window (40 time units) and tight drift thresholds
  (0.02/0.05) — this was verified to hold with driftE≈0.0004 at burn=130.
  re_mid (Re_f=40) and re_high (Re_f=160) show large-amplitude (~17%
  relative std of instantaneous E), long-correlation-time (~60–90 time
  units) intermittency — physically expected 2D-Kolmogorov-flow behavior
  (quasi-coherent-structure switching, e.g. Chandler & Kerswell 2013), *not*
  a numerical artifact (independently confirmed: production–dissipation
  balance <1% throughout; resolution criterion k_max/k_d = 3.12 for re_mid,
  well above the required 2.0). A fixed 40-unit window can never satisfy a
  2%/5% drift threshold for such a flow regardless of how long burn-in runs,
  because the split-half comparison is then dominated by which oscillation
  phase each half samples, not by any real trend — `configs/re_mid.yaml` and
  `configs/re_high.yaml` therefore use a substantially longer window
  (250 / 300 time units) and thresholds set from a directly measured
  300-time-unit diagnostic trace of re_mid (drift_e=0.12, drift_z=0.15 —
  both comfortably satisfied by the measured data at window ≥ 220), while
  keeping the balance threshold tight (0.02, tighter than before).
* **Amplitude-calibration robustness.** For the same reason, a short
  calibration measurement window (`pilot_measure`) at re_mid/re_high yields a
  noisy u_rms estimate; a 2-point secant fit through two noisy points can
  have a spurious slope that, taken at face value, produces a wildly wrong
  amplitude update (observed: a slope-0.105 fit sent one iteration's
  amplitude up 36%, making the estimate worse). `scripts/generate_dataset.py`
  now clamps the fitted slope to a physically sane range ([0.4, 1.2]; the
  a priori default is 0.75) and hard-caps the per-iteration multiplicative
  change in amplitude (±35%) regardless of the fit, and `pilot_measure` is
  lengthened for the higher-Re regimes.
* Every dataset file records burn-in time, the stationarity report, and the
  full diagnostic time series.

## 10. Dataset format

`data/<regime>_traj<k>.npz`:

* `omega`: vorticity snapshots `(T, Ny, Nx)`, `float32` by default (values
  are O(1–30); float32 relative precision ≈ 1e-7, ample for ML training;
  physics validation is done in float64 in-situ),
* `times`: snapshot times (float64), `dt_sample` spacing (0.25 low/mid,
  0.5 high — one snapshot per forcing-scale turnover time at u_rms = 1),
* `diag_*`: full scalar diagnostic series (E, Ω, P, ε, Courant, …),
* sidecar `.json`: complete provenance (all parameters, measured Reynolds
  numbers and scales, calibration history, stationarity report, conventions
  version string, software versions), plus a global `data/manifest.json`.

Velocity/pressure/streamfunction and all gradient fields are recoverable
exactly (spectrally) from `omega` — helpers in `solver/spectral.py` and
`solver/diagnostics.py`.

## 11. Regimes (parameter table)

| regime | N | ν (at u_rms=1) | target Re_f | k_f | drag σ | dt | measured quantities |
|---|---|---|---|---|---|---|---|
| re_low  | 64  | 2.5e-2   | 10  | 4 | 0.05 | 0.004  | recorded in data/*.json |
| re_mid  | 128 | 6.25e-3  | 40  | 4 | 0.05 | 0.002  | (u_rms, Re_f, Re_lam, |
| re_high | 256 | 1.5625e-3| 160 | 4 | 0.05 | 0.0018 | k_d, k_max/k_d, c_ε, …)|

The measured span is ≥ 10× (16× low→high), supporting Shi (2026)-style
cross-Reynolds extrapolation protocols (10× Re shift, relative-L2 rollout
error).

## 12. Sign/Convention quick reference

| quantity | convention |
|---|---|
| vorticity | ω = ∂v/∂x − ∂u/∂y |
| streamfunction | u = ∂ψ/∂y, v = −∂ψ/∂x ⇒ ω = −∇²ψ, ψ̂ = ω̂/k² |
| velocity from ω̂ | û = i k_y ω̂/k², v̂ = −i k_x ω̂/k² |
| advection term | −∇·(uω), k=0 mode zero |
| forcing | f = (A sin(k_f y), 0), F_ω = −A k_f cos(k_f y) |
| drag | momentum −σu ⇔ vorticity −σω; L = −(νk²+σ) |
| FFT | numpy rfft2 backward; fields (Ny, Nx); k = 2πn/L |
| dealias | 2/3 Galerkin: |n| ≤ ⌊N/3⌋; products alias-free |
| Re | Re_f = u_rms/(νk_f) primary; see §8 |
| time | ETDRK4 default; RK4/IFRK4 cross-validated |
