# few-special-functions

[![CI](https://github.com/MartinMikkelsen/few_special_functions/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinMikkelsen/few_special_functions/actions/workflows/ci.yml)
[![docs.rs](https://img.shields.io/docsrs/few_special_functions)](https://docs.rs/few_special_functions)
[![Crates.io](https://img.shields.io/crates/v/few_special_functions)](https://crates.io/crates/few_special_functions)

A pure Rust crate of special functions, with no native build-time
dependencies. It includes:

- [Clausen functions](https://en.wikipedia.org/wiki/Clausen_function)
- [Marcum Q-function](https://en.wikipedia.org/wiki/Marcum_Q-function)
- [Debye functions](https://en.wikipedia.org/wiki/Debye_function)
- [Fermi-Dirac integrals](https://en.wikipedia.org/wiki/Incomplete_Fermi%E2%80%93Dirac_integral)
- Bose-Einstein integrals
- [Fresnel integrals](https://en.wikipedia.org/wiki/Fresnel_integral)
- [Parabolic-cylinder functions](https://en.wikipedia.org/wiki/Parabolic_cylinder_function)
- [Whittaker functions](https://en.wikipedia.org/wiki/Whittaker_function)
- [Coulomb wave functions](https://en.wikipedia.org/wiki/Coulomb_wave_function)
- [Dawson function](https://en.wikipedia.org/wiki/Dawson_function)
- [Voigt profile](https://en.wikipedia.org/wiki/Voigt_profile)

The API uses `f64` for real-valued functions and `Complex<f64>` where the
[FewSpecialFunctions.jl](https://github.com/MartinMikkelsen/FewSpecialFunctions.jl)
API supports complex arguments; complex variants end in `_complex`.

## Install

```toml
[dependencies]
few_special_functions = "0.1"
```

The minimum supported Rust version is 1.96.

## Examples

```rust
use few_special_functions::clausen::clausen;
use few_special_functions::fermi_dirac::fermi_dirac_integral;
use few_special_functions::fresnel::fresnel_c;
use few_special_functions::marcum_q::marcum_q;

fn main() {
    // Fermi-Dirac integral F_{3/2}(1.0)
    println!("{}", fermi_dirac_integral(1.5, 1.0)); // 2.661682624732004

    // Clausen function Cl_2(π/2) = Catalan's constant
    println!("{}", clausen(2, std::f64::consts::FRAC_PI_2)); // 0.9159655941772190

    // Fresnel cosine integral C(1.0)
    println!("{}", fresnel_c(1.0)); // 0.7798934003...

    // Marcum Q-function Q_1(1.0, 2.0)
    println!("{}", marcum_q(1.0, 1.0, 2.0));
}
```

The `examples/` directory has runnable examples; the plotting ones save an SVG:

```
cargo run --example fresnel      # → fresnel.svg
cargo run --example fermi_dirac  # → fermi_dirac.svg
cargo run --example clausen      # → clausen.svg
cargo run --example debye        # → debye.svg
cargo run --example evaluate -- clausen 2 1.5707963
```

## SciPy counterparts

| This crate | SciPy |
|---|---|
| `fresnel::fresnel_c`, `fresnel::fresnel_s` | `scipy.special.fresnel(x)`, which returns `(S, C)` |
| `dawson::dawson` | `scipy.special.dawsn` |
| `voigt::voigt(x, y)` | `scipy.special.wofz(x + 1j*y).real`, or `sqrt(pi) * scipy.special.voigt_profile(x, 1/sqrt(2), y)` |
| `marcum_q::marcum_q(mu, a, b)` | `scipy.stats.ncx2.sf(b**2, 2*mu, a**2)` (`a > 0`) |
| `parabolic_cylinder::parabolic_cylinder_d(nu, x)`, `d_parabolic_cylinder_d` | `scipy.special.pbdv(nu, x)` |
| `parabolic_cylinder::u(a, x)` | `scipy.special.pbdv(-a - 0.5, x)[0]` |
| `parabolic_cylinder::v(a, x)` | `scipy.special.pbvv(-a - 0.5, x)[0]` |
| `parabolic_cylinder::w(a, x)` | `scipy.special.pbwa(a, x)[0]` |

The Clausen, Fermi–Dirac, Bose–Einstein, Debye, Whittaker and Coulomb
families have no SciPy counterpart.

## Accuracy

Every family is compared with reference values computed by
[mpmath](https://mpmath.org) at 50 significant digits. The tables live in
`tests/reference/`; each one records its definition, the mpmath, SciPy and
Python versions, and the generation date. Where the generator can, it
computes each table two independent ways, for example by quadrature of the
defining integral or from a Wronskian identity, and stops if the two
disagree. To regenerate the tables:

```
uv run scripts/generate_reference.py            # all tables
uv run scripts/generate_reference.py coulomb    # one family (about 15 minutes)
```

`tests/mpmath_reference.rs` asserts a stated tolerance on the relative error
of every function. To print the measured errors:

```
cargo test --release --test mpmath_reference -- --nocapture --test-threads=1
```

Measured maximum relative error on the reference tables. Where SciPy has the
function, its error on the same points is shown for comparison.

| Function | Region | Max. relative error | SciPy |
|---|---|---|---|
| `clausen` | n = 1..6, θ ∈ [−2.7, 100] | 1.1e-14 | — |
| `fermi_dirac_integral`, `_norm` | j ∈ [−0.5, 10], x ∈ [−50, 10⁴] | 8.1e-15 | — |
| `bose_einstein_integral`, `_norm` | k ∈ [−4.5, 4.5], η ∈ [−50, 0] | 1.1e-15 | — |
| `debye_function` | n ∈ [0.5, 4], β ∈ [0.5, 3], x ∈ [1e-3, 30] | 4.8e-16 | — |
| `dawson` | \|x\| ∈ [1e-10, 1e10] | 9.5e-16 | 8.3e-16 |
| `voigt` | \|x\| < 5 or y > 0.01 | 1.7e-13 | 7.2e-15 |
| `fresnel_c`, `fresnel_s` | \|x\| ∈ [1e-6, 1e3] | 2.2e-12 | 2.1e-16 |
| `fresnel_c_complex`, `fresnel_s_complex` | \|z\| ≤ 4.5 | 5.5e-12 | — |
| `marcum_q` | μ ∈ [0.5, 25], a, b ∈ [0, 30] | 2.9e-14 | 5.5e-14 |
| `dq_db` | M ∈ {1, 2, 3, 5}, a ∈ [0.1, 10], b ∈ [0.1, 20] | 3.3e-14 | — |
| `parabolic_cylinder::u`, `du` | a ∈ [−5, 10], x ∈ [−5, 10] | 1.6e-12 | — |
| `parabolic_cylinder::v`, `dv` | x ≥ 0 or a < 0 | 2.3e-12 | — |
| `parabolic_cylinder::w`, `dw` | a ∈ [−3, 3], x ∈ [−4, 4] | 4.6e-11 | — |
| `parabolic_cylinder_d`, `d_parabolic_cylinder_d` | ν ∈ [−3.5, 5], x ∈ [−5, 10] | 1.2e-13 | 5.8e-12 |
| `whittaker_m`, `d_whittaker_m` | κ ∈ [−2, 3], μ ∈ [0.25, 2.3], z ∈ [0.01, 30] | 3.8e-15 | — |
| `whittaker_w`, `d_whittaker_w` | same, except the regions below | 1.9e-11 | — |
| `whittaker_m_complex`, `whittaker_w_complex` | 6 points | 4.7e-14 | — |
| `coulomb::f`, `coulomb::g`, `coulomb::h_plus`, `coulomb::h_minus` | ℓ ∈ {0, 0.5, 1, 2, 5, 10, 20}, η ∈ [−50, 50], ρ ∈ [1e-3, 1e3] | 4.7e-13 | — |
| `coulomb::normalization` | ℓ ∈ [0, 5], η ∈ [−5, 10] | 6.2e-14 | — |
| `coulomb::phase`, `coulomb::aux_g` | 4 points | 2.2e-14 | — |
| `coulomb::psi`, `coulomb::i` | 4 points | 6.6e-9 (central difference in ℓ) | — |

### Known accuracy gaps

These regions are tested by `#[ignore]`d tests in `tests/mpmath_reference.rs`
that assert a relative error of 1e-12; run them with
`cargo test --release --test mpmath_reference -- --ignored`.

| Function | Region | Max. relative error |
|---|---|---|
| `voigt` | \|x\| ≥ 5 and y ≤ 0.01 | 1.5e-6 (SciPy: 7.0e-15) |
| `parabolic_cylinder::v`, `dv` | x < 0 and a ≥ 0 | 7.5e-1 (a = 10, x = −5) |
| `whittaker_w`, `d_whittaker_w` | 1/2 + μ − κ a non-positive integer, or κ < 0 with z ≥ 30 | 3.4e-7 |
