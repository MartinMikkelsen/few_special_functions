# few-special-functions

[![CI](https://github.com/MartinMikkelsen/few_special_functions/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinMikkelsen/few_special_functions/actions/workflows/ci.yml)
[![docs.rs](https://img.shields.io/docsrs/few_special_functions)](https://docs.rs/few_special_functions)
[![Crates.io](https://img.shields.io/crates/v/few_special_functions)](https://crates.io/crates/few_special_functions)

A Rust crate collecting a few special functions, with optional Python bindings via [PyO3](https://pyo3.rs). Includes the following functions:

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

The Rust API uses `f64` for real-valued functions and `Complex<f64>` where the
Julia API supports complex arguments. The optional Python module exposes the
same families; complex overloads end in `_complex`, and Coulomb functions use a
`coulomb_` prefix to avoid name collisions.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
few_special_functions = "0.1"
```

Or with `cargo add`:

```
cargo add few_special_functions
```

## Examples

```rust
use few_special_functions::fermi_dirac::fermi_dirac_integral;
use few_special_functions::clausen::clausen;
use few_special_functions::fresnel::fresnel_c;
use few_special_functions::marcum_q::marcum_q;

fn main() {
    // Fermi-Dirac integral F_{3/2}(1.0)
    println!("{}", fermi_dirac_integral(1.5, 1.0)); // 2.6616826247307124

    // Clausen function Cl_2(π/2) = Catalan's constant
    println!("{}", clausen(2, std::f64::consts::FRAC_PI_2)); // 0.9159655941772190

    // Fresnel cosine integral C(1.0)
    println!("{}", fresnel_c(1.0)); // 0.7798934003...

    // Marcum Q-function Q_1(1.0, 2.0)
    println!("{}", marcum_q(1.0, 1.0, 2.0));
}
```

## Plotting examples

The `examples/` directory contains runnable examples that print a table of values and save an SVG plot:

```
cargo run --example fresnel      # → fresnel.svg
cargo run --example fermi_dirac  # → fermi_dirac.svg
cargo run --example clausen      # → clausen.svg
cargo run --example debye        # → debye.svg
```

## Python bindings

Build a Python extension module with [maturin](https://www.maturin.rs):

```
maturin develop --features extension-module
```

```python
import few_special_functions as fsf

fsf.fermi_dirac_integral(1.5, 1.0)   # 2.6616826247307124
fsf.clausen(2, 3.14159 / 2)          # 0.9159655941772190
fsf.fresnel_c(1.0)                   # 0.7798934003...
fsf.marcum_q(1.0, 1.0, 2.0)
fsf.bose_einstein_integral_norm(0.5, -1.0)
fsf.parabolic_cylinder_d(1.0, 2.0)
fsf.whittaker_w(0.0, 0.5, 2.0)
fsf.coulomb_f(0.0, 0.0, 1.0)
```
