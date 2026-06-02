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
