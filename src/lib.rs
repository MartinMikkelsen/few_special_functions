// Coefficients are taken verbatim from published papers and intentionally
// exceed f64 precision so the source value is self-documenting.
#![allow(clippy::excessive_precision)]

pub mod clausen;
pub mod debye;
pub mod fermi_dirac;
pub mod fresnel;
