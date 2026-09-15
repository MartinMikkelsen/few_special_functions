//! # few-special-functions
//!
//! High-performance implementations of mathematical special functions,
//! with optional Python bindings via [PyO3](https://pyo3.rs).
//!
//! ## Available functions
//!
//! | Function family | Module | Public API |
//! |---|---|---|
//! | Bose-Einstein integrals | [`bose_einstein`] | [`bose_einstein::bose_einstein_integral`], [`bose_einstein::bose_einstein_integral_norm`] |
//! | Fermi-Dirac integrals | [`fermi_dirac`] | [`fermi_dirac::fermi_dirac_integral`], [`fermi_dirac::fermi_dirac_integral_norm`] |
//! | Clausen functions Cl_n(θ) | [`clausen`] | [`clausen::clausen`], [`clausen::clausen_with_options`], [`clausen::ci_complex`] |
//! | Fresnel integrals C, S, E | [`fresnel`] | [`fresnel::fresnel`], [`fresnel::fresnel_complex`], [`fresnel::fresnel_e`] |
//! | Debye functions D_n(β, x) | [`debye`] | [`debye::debye_function`], [`debye::debye_function_tol`] |
//! | Marcum Q-function | [`marcum_q`] | [`marcum_q::marcum_q`], [`marcum_q::dq_db`] |
//! | Parabolic-cylinder functions | [`parabolic_cylinder`] | [`parabolic_cylinder::u`], [`parabolic_cylinder::v`], [`parabolic_cylinder::w`], [`parabolic_cylinder::parabolic_cylinder_d`] |
//! | Whittaker functions | [`whittaker`] | [`whittaker::whittaker_m`], [`whittaker::whittaker_w`], [`whittaker::d_whittaker_m`], [`whittaker::d_whittaker_w`] |
//! | Coulomb wave functions | [`coulomb`] | [`coulomb::f`], [`coulomb::g`], [`coulomb::h_plus`], [`coulomb::h_minus`] |
//! | Dawson integral D(x) | [`dawson`] | [`dawson::dawson`] |
//! | Voigt function K(x, y) | [`voigt`] | [`voigt::voigt`] |
//!
//! ## Python bindings
//!
//! Build with `maturin develop --features extension-module` to produce a Python
//! extension module exposing every public family. Coulomb names use a
//! `coulomb_` prefix to avoid collisions with the parabolic-cylinder API, and
//! complex overloads use an explicit `_complex` suffix.

// Coefficients are taken verbatim from published papers and intentionally
// exceed f64 precision so the source value is self-documenting.
#![allow(clippy::excessive_precision)]

pub mod bose_einstein;
pub mod clausen;
pub mod coulomb;
pub mod dawson;
pub mod debye;
pub mod fermi_dirac;
pub mod fresnel;
pub mod marcum_q;
pub mod parabolic_cylinder;
pub mod voigt;
pub mod whittaker;

mod numerics;

// Python bindings — compiled only when the extension-module feature is active.
#[cfg(feature = "extension-module")]
mod python;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

/// Python module entry point — discovered by maturin at build time.
#[cfg(feature = "extension-module")]
#[pymodule]
fn few_special_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
