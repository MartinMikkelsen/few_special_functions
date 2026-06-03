//! # few-special-functions
//!
//! High-performance implementations of mathematical special functions,
//! with optional Python bindings via [PyO3](https://pyo3.rs).
//!
//! ## Available functions
//!
//! | Function family | Module | Public API |
//! |---|---|---|
//! | Fermi-Dirac integrals | [`fermi_dirac`] | [`fermi_dirac::fermi_dirac_integral`], [`fermi_dirac::fermi_dirac_integral_norm`] |
//! | Clausen functions Cl_n(θ) | [`clausen`] | [`clausen::clausen`], [`clausen::clausen_n20`] |
//! | Fresnel integrals C, S | [`fresnel`] | [`fresnel::fresnel`], [`fresnel::fresnel_c`], [`fresnel::fresnel_s`] |
//! | Debye functions D_n(β, x) | [`debye`] | [`debye::debye_function`], [`debye::debye_function_tol`] |
//! | Marcum Q-function | [`marcum_q`] | [`marcum_q::marcum_q`], [`marcum_q::dq_db`] |
//!
//! ## Python bindings
//!
//! Build with `maturin develop --features extension-module` to produce a Python
//! extension module exposing all functions above.

// Coefficients are taken verbatim from published papers and intentionally
// exceed f64 precision so the source value is self-documenting.
#![allow(clippy::excessive_precision)]

pub mod clausen;
pub mod debye;
pub mod fermi_dirac;
pub mod fresnel;
pub mod marcum_q;

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
