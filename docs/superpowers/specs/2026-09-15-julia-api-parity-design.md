# Julia API Parity Design

## Goal

Bring the Rust crate and its optional Python extension to functional parity with
the public API of the sibling `FewSpecialFunctions.jl` repository at Julia
commit `b909e7b` (version 0.1.15).

Parity means matching the Julia `Float64` behavior and documented domains. The
Rust implementation remains `f64`-first and uses `num_complex::Complex<f64>`
for APIs that accept complex values. Julia-only `Float16`, `Float32`,
`BigFloat`, broadcasting, and ForwardDiff behavior are not in scope.

No git commits will be created by Codex.

## Approach

Port the Julia algorithms and coefficient tables directly. Reuse the existing
`libm` and `num-complex` dependencies and enable PyO3's complex-number support.
Do not add a broad special-functions dependency or a C/C++ numerical backend.

A private numerical-support module will contain only shared primitives needed
by two or more public families:

- complex log-gamma, gamma, and digamma;
- confluent hypergeometric M and U kernels;
- real zeta support required by Bose--Einstein evaluation;
- adaptive quadrature required by the updated Debye implementation.

Existing helpers, such as incomplete gamma evaluation, will be reused rather
than duplicated.

## Public Rust API

Keep every family in its own public module. Preserve existing function names
and add the following modules:

- `bose_einstein`
- `coulomb`
- `parabolic_cylinder`
- `whittaker`

Use snake-case transliterations. Where Julia overloads real and complex inputs
with different return types, provide an `f64` primary function and an explicit
`*_complex` variant. Resolve names distinguished only by Julia capitalization
with descriptive Rust names:

- the two Coulomb-parameter overloads become `eta(a, k)` and
  `eta_from_energy(epsilon)`;
- Julia `C` and `theta` become `normalization` and `phase`;
- uppercase Coulomb wave `F`, `G`, `H+`, and `H-` become `f`, `g`, `h_plus`,
  and `h_minus`;
- lowercase Coulomb helpers `g`, `h_plus`, and `h_minus` become `aux_g`,
  `aux_h_plus`, and `aux_h_minus`;
- `Phi`, `Psi`, and `I` become `phi`, `psi`, and `i`;
- other names use direct snake-case transliteration.

The Clausen module will also expose Julia's public `ci_complex`, `f_n`, and
`f_clausen` helpers. Existing Rust-only convenience APIs remain available.

## Function Families

### New ports

1. Bose--Einstein integrals
   - normalized and unnormalized forms;
   - exact endpoint and nonpositive-integer formulas;
   - Fukushima minimax tables and convergent-series fallback.
2. Coulomb wave functions
   - parameter, normalization, phase, regular/irregular and incoming/outgoing
     waves;
   - regularized Kummer, modified-wave, Gamow, finite-difference, and auxiliary
     functions;
   - real and complex entry points where Julia supports complex arguments.
3. Parabolic-cylinder functions
   - `U`, `V`, `W` and argument derivatives;
   - `D_nu`, derivative, and scaled variants;
   - convergent series and accepted asymptotic expansions.
4. Whittaker functions
   - `M`, `W`, and their argument derivatives;
   - real and complex entry points;
   - Kummer series, connection formula, removable-pole limit, and asymptotic
     path.

### Existing ports to synchronize

- replace the generalized Debye series with the Julia quadrature formulation,
  including its corrected domain and endpoint behavior;
- port the latest Fresnel stability path applicable to `f64`/`Complex<f64>`;
- replace Marcum Q's large-order approximation with the centered
  Poisson--gamma mixture and preserve earlier Rust precision fixes;
- carry over the current Julia Voigt boundary behavior;
- retain existing Rust APIs and tests unless the Julia correction explicitly
  changes their documented result.

## Python API

Expose every Rust public function through the optional PyO3 extension. Use
family-prefixed names where short Julia names would collide or be unclear, for
example:

- `coulomb_f`, `coulomb_f_complex`, `coulomb_phi`;
- `parabolic_cylinder_u`, `parabolic_cylinder_d`;
- `whittaker_m`, `whittaker_m_complex`;
- `bose_einstein_integral` and `bose_einstein_integral_norm`.

Python complex values map to `Complex<f64>`. Domain validation in Python
wrappers returns `ValueError`; valid-input numerical results match the Rust
entry points.

## Error and Convergence Behavior

Follow the crate's existing convention: invalid Rust inputs panic with a clear
domain message. Iterative algorithms have explicit work limits and panic when
they cannot produce a converged value. Python wrappers validate documented
domains and translate those failures to Python exceptions where they can be
detected before evaluation.

No silent fallback may return a known unconverged partial sum. Overflow and
underflow may return IEEE infinity or signed zero where the Julia `Float64`
implementation does so.

## Testing

Implement each behavior test-first and observe the expected failure before
adding production code. Tests will use real code and deterministic data.

Coverage includes:

- Julia's checked-in reference tables for Coulomb, parabolic-cylinder,
  Bose--Einstein, scaled-cylinder, and Whittaker functions;
- elementary identities and conjugation/connection symmetries;
- derivative identities and finite-difference comparisons;
- endpoint, invalid-domain, overflow, underflow, and convergence-limit cases;
- a representative conformance grid generated from Julia commit `b909e7b`;
- Python smoke tests for every family, including complex round trips.

The final verification gate is:

1. `cargo test`
2. `cargo build --all-targets`
3. `cargo fmt --all -- --check`
4. `cargo clippy --all-targets -- -D warnings`
5. build the PyO3 extension and run its smoke tests
6. inspect the final git diff and confirm that no commit was created

## Documentation

Update crate-level API documentation and the README's function inventory and
examples. Document the `f64`/complex scope and the ASCII naming adaptations so
users can translate Julia calls directly.
