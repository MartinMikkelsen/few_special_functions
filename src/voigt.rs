//! The real Voigt function K(x, y).
//!
//! Adapted from the Fourier-expansion method in:
//! S. M. Abrarov, B. M. Quine, and R. K. Jagpal, "Rapidly convergent series
//! for high-accuracy calculation of the Voigt function," *Journal of
//! Quantitative Spectroscopy & Radiative Transfer* **111**, 372-375 (2010).
//! <https://doi.org/10.1016/j.jqsrt.2009.09.005>
//!
//! This is a faithful `f64` port of the Julia implementation in
//! `FewSpecialFunctions.jl` (`src/Voigt.jl`).

use num_complex::Complex;
use std::f64::consts::PI;

/// Upper limit t_M of the Fourier expansion interval.
const VOIGT_T_M: f64 = 12.0;

/// Fourier coefficients a₀ … a₂₃ of the expansion (Abrarov et al., 2010).
const VOIGT_FOURIER_COEFFICIENTS: [f64; 24] = [
    0.2954089751509193,
    0.27584023329217705,
    0.22457395522461585,
    0.1594149382739117,
    0.0986657664154542,
    0.053244140787639394,
    0.02505215000539365,
    0.01027746567053954,
    0.00367616433284485,
    0.001146493641242233,
    0.0003117570150461973,
    7.391433429603023e-5,
    1.527949342800837e-5,
    2.753956608221068e-6,
    4.327858781901254e-7,
    5.9300304087459056e-8,
    7.0844903077482305e-9,
    7.379520635816759e-10,
    6.70217160600201e-11,
    5.307265163470824e-12,
    3.664324113467629e-13,
    2.205894944941035e-14,
    1.1578268626285645e-15,
    5.2987114294673457e-17,
];

/// Numerically stable complex expm1(z) = e^z − 1.
///
/// Rust's `num-complex` has no `expm1`, so we build one from the real
/// `f64::exp_m1`. Writing the real part as
/// `expm1(a)·cos(b) − 2·sin²(b/2)` avoids the catastrophic cancellation of the
/// naive `exp(a)·cos(b) − 1` when both `a` and `b` are near zero, matching the
/// accuracy of Julia's `Base.expm1(::Complex)`.
fn cexpm1(z: Complex<f64>) -> Complex<f64> {
    let a = z.re;
    let b = z.im;
    let expm1a = a.exp_m1();
    let sin_half = (0.5 * b).sin();
    let re = expm1a * b.cos() - 2.0 * sin_half * sin_half;
    let im = (expm1a + 1.0) * b.sin();
    Complex::new(re, im)
}

/// Evaluates the elementary integral appearing in each Fourier term.
///
/// `cos_qtM` and `sin_qtM` are `cos(q·t_M)` and `sin(q·t_M)` supplied by the
/// caller. The `expm1` branch is stable throughout `|q|·t_M ≤ √eps`,
/// including `q = 0`, where the direct form would subtract nearly equal terms.
fn voigt_integral(y: f64, q: f64, cos_qtm: f64, sin_qtm: f64) -> f64 {
    let tm = VOIGT_T_M;
    if q.abs() <= f64::EPSILON.sqrt() / tm {
        let z = Complex::new(-y * tm, q * tm);
        return (-cexpm1(z) / Complex::new(y, -q)).re;
    }

    let scale = y.abs().max(q.abs());
    let y_scaled = y / scale;
    let q_scaled = q / scale;
    let numerator = y_scaled - (-y * tm).exp() * (y_scaled * cos_qtm - q_scaled * sin_qtm);
    numerator / (scale * (y_scaled * y_scaled + q_scaled * q_scaled))
}

/// Fourier-series evaluation of K(x, y), following Abrarov, Quine & Jagpal.
fn voigt_fourier(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() || y.is_infinite() {
        return 0.0;
    }

    let tm = VOIGT_T_M;
    let (sin_xtm, cos_xtm) = (x * tm).sin_cos();
    let mut k = VOIGT_FOURIER_COEFFICIENTS[0] * voigt_integral(y, x, cos_xtm, sin_xtm) / 2.0;

    for (n, &coeff) in VOIGT_FOURIER_COEFFICIENTS.iter().enumerate().skip(1) {
        let npi = n as f64 * PI / tm;
        // sinpi(n) = 0 and cospi(n) = (-1)ⁿ exactly for integer n.
        let sin_npi = 0.0;
        let cos_npi = if n.is_multiple_of(2) { 1.0 } else { -1.0 };
        let sin_plus = sin_xtm * cos_npi + cos_xtm * sin_npi;
        let cos_plus = cos_xtm * cos_npi - sin_xtm * sin_npi;
        let sin_minus = sin_xtm * cos_npi - cos_xtm * sin_npi;
        let cos_minus = cos_xtm * cos_npi + sin_xtm * sin_npi;
        k += coeff
            * (voigt_integral(y, x + npi, cos_plus, sin_plus)
                + voigt_integral(y, x - npi, cos_minus, sin_minus))
            / 2.0;
    }

    k / PI.sqrt()
}

/// The real Voigt function
///
/// ```text
///           y   ∞      e^{-t²}
/// K(x, y) = ─  ∫    ───────────── dt ,   y ≥ 0.
///           π  -∞   (x - t)² + y²
/// ```
///
/// For `y == 0`, `K(x, 0) = exp(-x²)`. The implementation follows the
/// Fourier-expansion method of Abrarov, Quine, and Jagpal.
///
/// # Panics
///
/// Panics with a domain error when `y < 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::voigt::voigt;
///
/// assert_eq!(voigt(0.0, 0.0), 1.0);
/// assert_eq!(voigt(2.0, 0.0), (-4.0_f64).exp());
/// // K is even in x.
/// assert_eq!(voigt(2.0, 0.5), voigt(-2.0, 0.5));
/// ```
pub fn voigt(x: f64, y: f64) -> f64 {
    if y < 0.0 {
        panic!("Voigt function requires y ≥ 0, got y = {y}");
    }
    if y == 0.0 {
        return (-x * x).exp();
    }
    voigt_fourier(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_expm1_matches_naive_away_from_zero() {
        let z = Complex::new(-1.5, 0.3);
        let naive = z.exp() - Complex::new(1.0, 0.0);
        let stable = cexpm1(z);
        assert!((stable - naive).norm() < 1e-13);
    }
}
