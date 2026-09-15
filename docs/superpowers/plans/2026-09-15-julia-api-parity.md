# Julia API Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Rust crate and optional Python extension to `Float64` API and behavior parity with `FewSpecialFunctions.jl` commit `b909e7b`.

**Architecture:** Port the Julia algorithms directly into focused Rust modules. Share only complex gamma/Kummer, zeta, and adaptive-quadrature primitives in a private `numerics` module; keep family-specific algorithms and coefficient tables with their public modules.

**Tech Stack:** Rust 2024, `libm`, `num-complex`, optional PyO3 0.28 with its `num-complex` feature, Cargo integration tests, and Julia 1.10+ as the reference-value oracle.

**Spec:** `docs/superpowers/specs/2026-09-15-julia-api-parity-design.md`

## Global Constraints

- Target Julia commit: `b909e7b`, package version 0.1.15.
- Public numeric scope: `f64` and `num_complex::Complex<f64>` only.
- Keep the crate portable and pure Rust; add no broad special-functions or C/C++ dependency.
- Preserve all current Rust APIs while adding snake-case Julia API mappings.
- Expose all Rust APIs through the optional Python extension, including complex variants.
- Invalid Rust inputs panic with clear messages; Python domain failures become `ValueError`.
- Do not create git commits.
- Every behavior change follows a witnessed red-green test cycle.

## File Structure

- Create `src/numerics.rs`: private shared complex gamma/digamma, Kummer M/U, real zeta, and adaptive Gauss-Kronrod integration.
- Create `src/bose_einstein.rs`: Bose--Einstein public APIs and Fukushima coefficient tables.
- Create `src/coulomb.rs`: Coulomb public APIs and real/complex adapters.
- Create `src/parabolic_cylinder.rs`: U/V/W, D, derivatives, and scaled variants.
- Create `src/whittaker.rs`: Whittaker M/W and derivatives for real/complex inputs.
- Modify `src/clausen.rs`: expose Julia-public helper functions.
- Modify `src/debye.rs`: port Julia's corrected quadrature algorithm.
- Modify `src/fresnel.rs`: port current real/complex stability branches.
- Modify `src/marcum_q.rs`: port centered Poisson--gamma large-order branch.
- Modify `src/voigt.rs`: align current Julia edge handling.
- Modify `src/lib.rs`: register modules and document the full API.
- Modify `src/python.rs`: bind the complete public API.
- Create `tests/bose_einstein.rs`, `tests/coulomb.rs`, `tests/parabolic_cylinder.rs`, and `tests/whittaker.rs`.
- Create `tests/python_smoke.py`: executable smoke coverage for every Python family.
- Modify existing family tests for synchronized behavior.
- Copy reference tables into `tests/test_data/` as plain TSV/TXT fixtures.
- Modify `README.md` and `.github/workflows/ci.yml`: inventory, naming guide, and Python smoke coverage.

---

### Task 1: Baseline and Julia-Public Clausen Helpers

**Files:**
- Modify: `src/clausen.rs`
- Test: `tests/clausen.rs`

**Interfaces:**
- Produces: `pub fn ci_complex(Complex<f64>) -> Complex<f64>`
- Produces: `pub fn f_n(usize, usize, f64) -> f64`
- Produces: `pub fn f_clausen(usize, Complex<f64>, f64) -> Complex<f64>`
- Produces: `pub fn clausen_with_options(usize, f64, usize, usize) -> f64`
- Preserves: `clausen` and `clausen_n20`

- [ ] **Step 1: Establish the clean baseline**

Run:

```bash
cargo test
```

Expected: all pre-existing tests pass before production changes.

- [ ] **Step 2: Add failing public-helper tests**

Add imports and assertions equivalent to:

```rust
use few_special_functions::clausen::{ci_complex, f_clausen, f_n};
use num_complex::Complex;

#[test]
fn julia_public_helpers_are_available() {
    let ci = ci_complex(Complex::new(1.0, 0.0));
    assert!((ci.re - 0.3374039229009681).abs() < 1e-10);
    assert!((f_n(2, 2, 0.7) - (1.4_f64).sin() / 4.0).abs() < 1e-15);
    let value = f_clausen(1, Complex::new(2.0, 0.5), 0.7);
    assert!(value.re.is_finite() && value.im.is_finite());
}
```

- [ ] **Step 3: Verify RED**

Run `cargo test --test clausen julia_public_helpers_are_available`.
Expected: compilation fails because the three helpers are private or absent.

- [ ] **Step 4: Expose the exact helper behavior**

Rename `sum_term` to `f_n`, make `ci_complex` public, and change
`f_clausen` to return its complex primitive rather than discarding `.re`.
Keep `.re` only at the Euler--Maclaurin call site:

```rust
pub fn f_n(n: usize, k: usize, theta: f64) -> f64 { /* parity formula */ }
pub fn ci_complex(z: Complex<f64>) -> Complex<f64> { /* existing body */ }
pub fn f_clausen(n: usize, z: Complex<f64>, theta: f64) -> Complex<f64> {
    match n { 1 => ci_complex(z * theta), 2..=6 => /* existing primitives */, _ => panic!("n must be 1..=6") }
}
```

- [ ] **Step 5: Verify GREEN**

Run `cargo test --test clausen` and `cargo test --lib clausen`.
Expected: both commands pass.

### Task 2: Shared Numerical Kernels

**Files:**
- Create: `src/numerics.rs`
- Modify: `src/lib.rs`
- Test: unit tests in `src/numerics.rs`

**Interfaces:**
- Produces: `pub(crate) fn log_gamma_complex(Complex64) -> Complex64`
- Produces: `pub(crate) fn gamma_complex(Complex64) -> Complex64`
- Produces: `pub(crate) fn digamma_complex(Complex64) -> Complex64`
- Produces: `pub(crate) fn kummer_m(Complex64, Complex64, Complex64) -> Complex64`
- Produces: `pub(crate) fn tricomi_u(Complex64, Complex64, Complex64) -> Complex64`
- Produces: `pub(crate) fn zeta_real(f64) -> f64`
- Produces: `pub(crate) fn integrate<F: Fn(f64) -> f64>(F, f64, f64, f64, usize) -> Result<f64, &'static str>`

- [ ] **Step 1: Add failing kernel tests**

Create `src/numerics.rs` with only tests for authoritative values:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn complex_gamma_and_kummer_references() {
        let z = Complex64::new(0.7, 0.4);
        let lg = log_gamma_complex(z);
        assert!((lg - Complex64::new(0.057044748102826226, -0.42944066621229504)).norm() < 2e-14);
        let m = kummer_m(Complex64::new(0.8, -0.3), Complex64::new(1.7, 0.2), Complex64::new(1.0, 2.0));
        assert!((m - Complex64::new(1.3922407178086509, 1.6474537115932955)).norm() < 2e-13);
    }

    #[test]
    fn zeta_and_quadrature_references() {
        assert!((zeta_real(2.0) - std::f64::consts::PI.powi(2) / 6.0).abs() < 2e-15);
        let integral = integrate(|x| x * x, 0.0, 1.0, 1e-14, 1000).unwrap();
        assert!((integral - 1.0 / 3.0).abs() < 2e-14);
    }
}
```

The literals above come from Julia `SpecialFunctions.loggamma` and
`HypergeometricFunctions._1F1` at the pinned package commit.

- [ ] **Step 2: Verify RED**

Run `cargo test --lib numerics`.
Expected: compilation fails because all kernel functions are missing.

- [ ] **Step 3: Implement complex gamma and digamma**

Implement a Lanczos log-gamma with reflection for `Re(z) < 0.5`, principal
complex logarithms, and a recurrence-plus-asymptotic digamma. Derive gamma as
`log_gamma_complex(z).exp()`. Use only `Complex64` and `f64` constants.

- [ ] **Step 4: Implement Kummer M and Tricomi U**

Use compensated direct series for M, Kummer transformation when it reduces
the cancellation risk, and asymptotic expansion when decreasing terms reach
`f64` precision. Define U by its connection formula with a symmetric limit
when `b` is an integer. Reject a partial sum that reaches the work limit.

- [ ] **Step 5: Implement real zeta and adaptive integration**

Implement zeta through reflection/functional equation, Euler--Maclaurin in
the regular region, and exact trivial-zero handling. Implement an adaptive
15-point Gauss--Kronrod integrator using an explicit interval stack and both
relative and absolute error estimates.

- [ ] **Step 6: Verify GREEN**

Run `cargo test --lib numerics`.
Expected: all kernel tests pass with no warnings.

### Task 3: Synchronize Existing Families

**Files:**
- Modify: `src/debye.rs`
- Modify: `src/fresnel.rs`
- Modify: `src/marcum_q.rs`
- Modify: `src/voigt.rs`
- Modify: `tests/debye.rs`
- Modify: `tests/fresnel.rs`
- Modify: `tests/marcum_q/main.rs`
- Modify: `tests/voigt.rs`

**Interfaces:**
- Preserves all current signatures.
- Adds: `fresnel_e`, `fresnel_complex`, `fresnel_c_complex`, `fresnel_s_complex`, and `fresnel_e_complex`.
- Adds: `marcum_q_order_one(a, b)` and `dq_db_order_one(a, b)` for Julia's two-argument overloads.
- Consumes: `numerics::integrate`.

- [ ] **Step 1: Add failing Julia-regression tests**

Add the corrected Debye origin rules and generalized domain:

```rust
assert_eq!(debye_function(2.0, 0.5, 0.0), 0.0);
assert_eq!(debye_function(2.0, 1.0, 0.0), 1.0);
assert!(debye_function(2.0, 1.5, 0.0).is_infinite());
assert_eq!(debye_function(2.0, 1.0, f64::INFINITY), 0.0);
```

Add complex Fresnel references from Julia's `test_Fresnel.jl`, and add a
large-order Marcum Q case that fails the old asymptotic branch but matches the
current centered Poisson--gamma mixture. Add Julia's current Voigt NaN and
infinity cases.

- [ ] **Step 2: Verify RED family by family**

Run:

```bash
cargo test --test debye
cargo test --test fresnel
cargo test --test marcum_q
cargo test --test voigt
```

Expected: each newly introduced behavior fails for the documented reason.

- [ ] **Step 3: Port the corrected Debye quadrature**

Translate Julia `_debye_function` exactly: validate finite `n`, enforce
`0 < beta < n + 1`, transform the singular integral to a bounded interval,
call `integrate`, and combine scale factors in logarithmic form when direct
multiplication is not representable. Keep `debye_function_tol` as the explicit
tolerance/work-limit API.

- [ ] **Step 4: Port complex Fresnel evaluation**

Translate `_fresnel_series`, Miller recurrence, asymptotic auxiliary terms,
quadrant handling, and the independent `E` path. Keep the current real names;
add:

```rust
pub fn fresnel_complex(z: Complex64) -> (Complex64, Complex64, Complex64);
pub fn fresnel_c_complex(z: Complex64) -> Complex64;
pub fn fresnel_s_complex(z: Complex64) -> Complex64;
pub fn fresnel_e_complex(z: Complex64) -> Complex64;
```

- [ ] **Step 5: Port Marcum Q and Voigt corrections**

Replace only `marcum_q_large_m` with the centered mode recurrence from Julia
commit `1336b43`; preserve Rust's earlier relative-precision and quadrature
table fixes. Apply the current Julia Voigt finite/domain branches without
rewriting the Fourier kernel.

- [ ] **Step 6: Verify GREEN**

Run the four integration-test commands from Step 2 and `cargo test --lib`.
Expected: all pass.

### Task 4: Bose--Einstein Integrals

**Files:**
- Create: `src/bose_einstein.rs`
- Modify: `src/lib.rs`
- Create: `tests/bose_einstein.rs`
- Copy: `tests/test_data/bose_einstein_fukushima.tsv`
- Copy: `tests/test_data/bose_einstein_reference.tsv`

**Interfaces:**
- Produces: `pub fn bose_einstein_integral(k: f64, eta: f64) -> f64`
- Produces: `pub fn bose_einstein_integral_norm(k: f64, eta: f64) -> f64`
- Consumes: `numerics::zeta_real`

- [ ] **Step 1: Copy reference fixtures and add failing tests**

Copy the two Julia TSV files byte-for-byte. Add tests for normalized and
unnormalized references, endpoints, valid integer/half-integer orders, invalid
orders, monotonicity, large orders, and overflow/underflow. Include:

```rust
assert!((bose_einstein_integral_norm(0.5, -1.0) - 0.4284407345998379).abs() < 2e-15);
assert_eq!(bose_einstein_integral_norm(1.0, 0.0), std::f64::consts::PI.powi(2) / 6.0);
assert_eq!(bose_einstein_integral_norm(-0.5, 0.0), f64::INFINITY);
assert_eq!(bose_einstein_integral_norm(1.0, f64::NEG_INFINITY), 0.0);
```

- [ ] **Step 2: Verify RED**

Run `cargo test --test bose_einstein`.
Expected: compilation fails because `bose_einstein` is not exported.

- [ ] **Step 3: Port implementation and coefficients**

Translate `BoseEinstein.jl` and `BoseEinsteinCoefficients.jl` at `b909e7b`:
exact endpoint formulas, compensated fugacity and near-zero series, the
piecewise Fukushima rational evaluation, and logarithmic rescaling of the
unnormalized result. Keep coefficient tables in the same source file because
they change only with this family.

- [ ] **Step 4: Verify GREEN**

Run `cargo test --test bose_einstein`.
Expected: all table rows and boundary tests pass.

### Task 5: Parabolic-Cylinder Family

**Files:**
- Create: `src/parabolic_cylinder.rs`
- Modify: `src/lib.rs`
- Create: `tests/parabolic_cylinder.rs`
- Copy: `tests/test_data/U_data.txt`
- Copy: `tests/test_data/V_data.txt`
- Copy: `tests/test_data/cylinder_scaled.tsv`

**Interfaces:**
- Produces: `u`, `v`, `w`, `du`, `dv`, `dw`: `(f64, f64) -> f64`
- Produces: `u_scaled`, `v_scaled`: `(f64, f64) -> f64`
- Produces: `parabolic_cylinder_d`, `d_parabolic_cylinder_d`, and `parabolic_cylinder_d_scaled`: `(f64, f64) -> f64`
- Consumes: `numerics::log_gamma_complex`

- [ ] **Step 1: Add failing value, identity, and domain tests**

Port the Julia Float64 rows from `test_parabolic_cylinder.jl` and
`test_cylinder_scaled.jl`. Include Gaussian/Hermite identities, Wronskians,
derivative values, large-argument underflow/overflow, all scaled references,
and finite/nonnegative scaled-domain checks.

- [ ] **Step 2: Verify RED**

Run `cargo test --test parabolic_cylinder`.
Expected: compilation fails because the module does not exist.

- [ ] **Step 3: Implement series and initial values**

Translate `_cylinder_initial` and the actual-term Taylor recurrence for values
and derivatives. For f64 cancellation cases, use compensated sums and detect
non-finite or non-converged results rather than returning a partial sum.

- [ ] **Step 4: Implement asymptotic and scaled paths**

Translate U, V-scaled, and W asymptotic recurrences, accept them only when
decreasing terms reach target precision, and combine exponential scales in
log space. Implement negative-half-integer U parity exactly.

- [ ] **Step 5: Add public wrappers and verify GREEN**

Implement all eleven public signatures listed above. Run
`cargo test --test parabolic_cylinder`; expected: all tests pass.

### Task 6: Whittaker Family

**Files:**
- Create: `src/whittaker.rs`
- Modify: `src/lib.rs`
- Create: `tests/whittaker.rs`
- Copy: `tests/test_data/whittaker.tsv`

**Interfaces:**
- Produces: `whittaker_m`, `whittaker_w`, `d_whittaker_m`, `d_whittaker_w`: `(f64, f64, f64) -> f64`
- Produces matching `*_complex(Complex64, Complex64, Complex64) -> Complex64` variants.
- Consumes: `numerics::{gamma_complex, kummer_m, log_gamma_complex}`.

- [ ] **Step 1: Add failing reference and identity tests**

Translate every Float64/ComplexF64 row from Julia `test/data/whittaker.jl` and
the elementary identities from `test_Whittaker.jl`. Test nonzero argument,
negative-real branch rejection for real APIs, parameter poles, conjugate sides
of the negative-axis branch, large-z signed underflow, and analytic derivatives.

- [ ] **Step 2: Verify RED**

Run `cargo test --test whittaker`.
Expected: compilation fails because the module does not exist.

- [ ] **Step 3: Implement M, W, and derivatives**

Translate the Julia prefactors, W connection formula, symmetric removable-pole
limit, accepted Poincare expansion, and parameter-shift derivative identities.
Use log-space products where a separate prefactor would overflow or underflow.

- [ ] **Step 4: Verify GREEN**

Run `cargo test --test whittaker`.
Expected: all real and complex reference rows pass within the Julia Float64
tolerances; no work-limit path returns an incomplete expansion.

### Task 7: Coulomb Family

**Files:**
- Create: `src/coulomb.rs`
- Modify: `src/lib.rs`
- Create: `tests/coulomb.rs`
- Copy: `tests/test_data/CoulombF.txt`
- Copy: `tests/test_data/CoulombG.txt`

**Interfaces:**
- Produces real APIs: `eta`, `eta_from_energy`, `normalization`, `phase`, `f`, `g`, `f_imag`, `m_regularized`, `w`, `aux_g`.
- Produces complex-valued real-parameter APIs: `d_plus`, `d_minus`, `h_plus`, `h_minus`, `phi`, `w_plus`, `w_minus`, `aux_h_plus`, `aux_h_minus`, `phi_dot`, `f_dot`, `psi`, `i`.
- Produces a `*_complex` counterpart for each Julia `Number` API whose complex inputs alter the result.
- Consumes: `numerics::{digamma_complex, gamma_complex, kummer_m, log_gamma_complex, tricomi_u}`.

- [ ] **Step 1: Add failing table, identity, and API tests**

Translate Julia `test_Coulomb.jl`, including F/G fixtures, zero-charge
normalization, complex holomorphic checks, incoming/outgoing conjugation,
regularized Kummer identities, Gamow integer/half-integer behavior, auxiliary
gamma/digamma symmetries, finite-difference functions, and invalid eta/order
cases.

- [ ] **Step 2: Verify RED**

Run `cargo test --test coulomb`.
Expected: compilation fails because the module does not exist.

- [ ] **Step 3: Implement normalization and wave functions**

Port normalization in log-gamma form, phase, regular F, D factors, Tricomi U
incoming/outgoing waves, F/G combinations, regularized Kummer M, and Phi.
Real adapters return the real component exactly where Julia has explicit real
overloads.

- [ ] **Step 4: Implement auxiliary and derivative APIs**

Port integer/half-integer Gamow products, gamma/digamma helpers, and central
differences with default step `f64::EPSILON.cbrt()`. Preserve Julia's order
validation and expose explicit-step variants such as `phi_dot_with_step` and
`f_dot_with_step` because Rust has no keyword arguments.

- [ ] **Step 5: Verify GREEN**

Run `cargo test --test coulomb`.
Expected: fixture, complex, identity, and domain tests all pass.

### Task 8: Python Bindings and Documentation

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/python.rs`
- Modify: `src/lib.rs`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes every public family API from Tasks 1--7.
- Produces Python functions with family prefixes and Python complex conversion.

- [ ] **Step 1: Add failing Python smoke checks**

Extend the CI smoke script to import and call every family. Include complex
round trips:

```python
assert abs(fsf.bose_einstein_integral_norm(0.5, -1.0) - 0.4284407345998379) < 3e-15
assert abs(fsf.parabolic_cylinder_d(0.0, 2.0) - math.exp(-1.0)) < 2e-13
assert abs(fsf.whittaker_w(0.0, 0.5, 2.0) - math.exp(-1.0)) < 2e-13
z = fsf.whittaker_w_complex(0.2 + 0.0j, 0.3 + 0.0j, 1.0 + 2.0j)
assert isinstance(z, complex)
assert abs(fsf.coulomb_f(0.0, 0.0, 1.0) - math.sin(1.0)) < 1e-10
```

- [ ] **Step 2: Verify RED**

Build/install the current extension and run the smoke script. Expected: new
attributes are missing.

- [ ] **Step 3: Enable and implement complex bindings**

Set:

```toml
pyo3 = { version = "0.28", optional = true, features = ["num-complex"] }
```

Add thin `#[pyfunction]` wrappers for every public API, using family prefixes
for collisions. Validate documented domains before invoking panic-based Rust
entry points and return `PyValueError` on invalid input.

- [ ] **Step 4: Update documentation**

Add all families to the crate-level table and README inventory. Add a compact
Julia-to-Rust/Python naming table and one example each for Bose--Einstein,
parabolic cylinder, Whittaker, and Coulomb.

- [ ] **Step 5: Verify GREEN**

Build/install the extension and rerun the complete smoke script. Expected: all
real and complex checks pass.

### Task 9: Full Verification and No-Commit Audit

**Files:**
- Modify only files required to fix failures discovered by the commands below.

**Interfaces:**
- Validates the complete crate, extension, documentation examples, and git state.

- [ ] **Step 1: Run the full Rust suite**

Run `cargo test`. Expected: unit, integration, and doctests all pass.

- [ ] **Step 2: Verify every target builds**

Run `cargo build --all-targets`. Expected: exit 0 with no warnings.

- [ ] **Step 3: Verify formatting**

Run `cargo fmt --all -- --check`. If it fails, run `cargo fmt --all`, inspect
the formatting-only diff, then rerun the check. Expected: exit 0.

- [ ] **Step 4: Verify Clippy**

Run `cargo clippy --all-targets -- -D warnings`. Expected: exit 0.

- [ ] **Step 5: Verify the Python extension**

Run:

```bash
.venv/bin/maturin develop --features extension-module
.venv/bin/python tests/python_smoke.py
```

Expected: all smoke checks pass.

- [ ] **Step 6: Compare exported inventories**

Run Julia `names(FewSpecialFunctions; all=false, imported=false)` and compare
it with the Rust module inventory and Python `dir(few_special_functions)`.
Expected: every Julia public function has its documented Rust/Python mapping.

- [ ] **Step 7: Inspect the diff and commit state**

Run `git status --short`, `git diff --check`, and `git log -1 --oneline`.
Expected: only intended source, test, fixture, CI, and documentation changes;
no whitespace errors; `HEAD` remains `a480595`.
