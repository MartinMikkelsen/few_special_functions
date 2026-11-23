//! Coulomb wave functions and the auxiliary API used by their representations.
//!
//! Ported from `src/CoulombWave.jl` of FewSpecialFunctions.jl
//! (<https://github.com/MartinMikkelsen/FewSpecialFunctions.jl>, MIT license),
//! commit b909e7b22696e942762f800ddad34a70d5b9d51d.
//! The Julia code follows <https://arxiv.org/abs/1804.10976> and DLMF chapter 33.
//! The real-argument `F`, `G` and `H±` are written for this crate: Steed's
//! method (`src/coulomb/steed.rs`) and, outside its domain, the Poincaré
//! asymptotic series integrated backward along the Coulomb equation.
//!
//! # Iteration caps and tolerances
//!
//! | Method | Stops when | Cap |
//! |---|---|---|
//! | Steed CF1 (`F'/F`) | two truncation depths agree to `4ε` | depth 2²⁶ |
//! | Steed CF2 (`H⁺'/H⁺`), modified Lentz | `\|δ − 1\| ≤ 2ε` | 1 000 000 terms |
//! | Taylor steps of `G` inside the turning point | two successive terms below `ε/4` | 2 000 terms per step |
//! | Backward Runge–Kutta integration of real `G` outside Steed's domain | fixed steps | 2 000 000 steps |
//! | Kummer series `M` of the complex representations | term below `4 ε max(1, \|sum\|)` | 100 000 terms |
//! | Regularized Kummer series `M/Γ(b)` and its parameter derivatives | term below `4 ε max(\|sum\|, ε max \|term\|)` | 100 000 terms |
//!
//! Reaching a cap gives [`SpecialError::NoConvergence`] from the `*_checked`
//! functions and NaN from the others. The asymptotic series that starts the
//! backward integration is truncated at its smallest term, within 1 000 terms.
//! The order derivative [`phi_dot`], and [`psi`] and [`i`] built on it,
//! differentiate the Kummer series term by term and share its tolerance and
//! cap. Their `step` argument is ignored. [`f_dot`] is a central difference and has no tolerance of its own.

use crate::SpecialError;
use crate::error::iteration_cap;
use crate::numerics::{
    digamma_complex, gamma_complex, kummer_m, kummer_m_regularized,
    kummer_m_regularized_with_parameter_derivatives, log_gamma_complex, reciprocal_gamma,
    tricomi_u,
};
use num_complex::Complex64;
use std::f64::consts::PI;

mod steed;

const FAMILY: &str = "coulomb";
/// Cap on the Runge–Kutta steps of the backward integration of `G`.
const MAX_ODE_STEPS: usize = 2_000_000;
const NAN: Complex64 = Complex64::new(f64::NAN, f64::NAN);

/// Coulomb parameter `1 / (a k)`; NaN if `a` or `k` is zero.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::eta;
///
/// assert_eq!(eta(2.0, 0.25), 2.0);
/// ```
pub fn eta(a: f64, k: f64) -> f64 {
    if a == 0.0 || k == 0.0 {
        return f64::NAN;
    }
    1.0 / (a * k)
}

/// Coulomb parameter `1 / (a k)` for complex inputs; NaN if `a` or `k` is
/// zero.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::eta_complex;
/// use num_complex::Complex64;
///
/// let v = eta_complex(Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.25));
/// assert!((v - Complex64::new(0.0, -2.0)).norm() < 1e-15);
/// ```
pub fn eta_complex(a: Complex64, k: Complex64) -> Complex64 {
    let zero = Complex64::new(0.0, 0.0);
    if a == zero || k == zero {
        return NAN;
    }
    1.0 / (a * k)
}

/// Coulomb parameter `1 / sqrt(epsilon)`; NaN unless `epsilon > 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::eta_from_energy;
///
/// assert_eq!(eta_from_energy(4.0), 0.5);
/// ```
pub fn eta_from_energy(epsilon: f64) -> f64 {
    if epsilon == 0.0 {
        return f64::NAN;
    }
    1.0 / epsilon.sqrt()
}

/// Coulomb parameter `1 / sqrt(epsilon)` for a complex energy; NaN if
/// `epsilon` is zero.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::eta_from_energy_complex;
/// use num_complex::Complex64;
///
/// // 1 / sqrt(−4) = 1 / (2i) on the principal branch
/// let v = eta_from_energy_complex(Complex64::new(-4.0, 0.0));
/// assert!((v - Complex64::new(0.0, -0.5)).norm() < 1e-15);
/// ```
pub fn eta_from_energy_complex(epsilon: Complex64) -> Complex64 {
    if epsilon == Complex64::new(0.0, 0.0) {
        return NAN;
    }
    1.0 / epsilon.sqrt()
}

/// Analytic Coulomb normalization for complex parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{normalization, normalization_complex};
/// use num_complex::Complex64;
///
/// let v = normalization_complex(Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0));
/// assert_eq!(v.re, normalization(1.0, 0.5));
/// ```
pub fn normalization_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    let imaginary = Complex64::i();
    let log_gamma_product = 0.5
        * (log_gamma_complex(ell + 1.0 + imaginary * eta)
            + log_gamma_complex(ell + 1.0 - imaginary * eta));
    (ell * (2.0_f64).ln() - PI * eta / 2.0 + log_gamma_product - log_gamma_complex(2.0 * ell + 2.0))
        .exp()
}

/// Positive Coulomb normalization for real parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::normalization;
/// use std::f64::consts::PI;
///
/// // C_0(η)² = 2πη / (e^{2πη} − 1)
/// let eta = 1.0_f64;
/// let want = (2.0 * PI * eta / (2.0 * PI * eta).exp_m1()).sqrt();
/// assert!((normalization(0.0, eta) - want).abs() < 1e-13 * want);
/// ```
pub fn normalization(ell: f64, eta: f64) -> f64 {
    normalization_complex(ell.into(), eta.into()).re
}

/// Coulomb phase for real parameters and positive `rho`; NaN unless
/// `rho > 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::phase;
/// use std::f64::consts::FRAC_PI_2;
///
/// // Without the Coulomb field, θ_ℓ = ρ − ℓπ/2.
/// assert!((phase(1.0, 0.0, 2.0) - (2.0 - FRAC_PI_2)).abs() < 1e-15);
///
/// // mpmath: θ_1(0.5, 2) = −0.044354197259463575
/// assert!((phase(1.0, 0.5, 2.0) + 0.044_354_197_259_463_575).abs() < 1e-14);
/// ```
pub fn phase(ell: f64, eta: f64, rho: f64) -> f64 {
    if rho.is_nan() || rho <= 0.0 {
        return f64::NAN;
    }
    rho - ell * PI / 2.0 - eta * (2.0 * rho).ln()
        + log_gamma_complex(Complex64::new(ell + 1.0, eta)).im
}

/// Coulomb phase generalized to complex parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{phase, phase_complex};
/// use num_complex::Complex64;
///
/// let v = phase_complex(
///     Complex64::new(1.0, 0.0),
///     Complex64::new(0.5, 0.0),
///     Complex64::new(2.0, 0.0),
/// );
/// assert!((v.re - phase(1.0, 0.5, 2.0)).abs() < 1e-15);
/// ```
pub fn phase_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    let log_gamma_imaginary = log_gamma_complex(ell + 1.0 + Complex64::i() * eta).im;
    rho - ell * PI / 2.0 - eta * (2.0 * rho).ln() + log_gamma_imaginary
}

/// Regular Coulomb wave function for complex inputs.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::f_complex;
/// use num_complex::Complex64;
///
/// // Without the Coulomb field, F_0(0, ρ) = sin ρ.
/// let rho = 1.3_f64;
/// let v = f_complex(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), rho.into());
/// assert!((v - Complex64::new(rho.sin(), 0.0)).norm() < 1e-14);
/// ```
pub fn f_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    f_complex_checked(ell, eta, rho).unwrap_or(NAN)
}

/// [`f_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{f_complex, f_complex_checked};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(f_complex_checked(ell, eta, rho), Ok(f_complex(ell, eta, rho)));
/// ```
pub fn f_complex_checked(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
) -> Result<Complex64, SpecialError> {
    let imaginary = Complex64::i();
    Ok(normalization_complex(ell, eta)
        * rho.powc(ell + 1.0)
        * (-imaginary * rho).exp()
        * kummer_m(
            ell + 1.0 - imaginary * eta,
            2.0 * ell + 2.0,
            2.0 * imaginary * rho,
            FAMILY,
        )?)
}

/// Regular Coulomb wave function for real inputs.
///
/// For `rho > 0` and `ell > -1` this uses Steed's method (see [`g`] for the
/// accuracy). Other arguments use the Kummer-function representation of
/// [`f_complex`], which the reference tables do not cover.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::f;
///
/// // Without the Coulomb field, F_1(0, ρ) is the Riccati–Bessel function sin ρ / ρ − cos ρ.
/// let rho = 2.0_f64;
/// assert!((f(1.0, 0.0, rho) - (rho.sin() / rho - rho.cos())).abs() < 1e-14);
///
/// // mpmath: F_1(0.5, 2) = 0.61851232128919756
/// assert!((f(1.0, 0.5, 2.0) - 0.618_512_321_289_197_56).abs() < 1e-14);
/// ```
pub fn f(ell: f64, eta: f64, rho: f64) -> f64 {
    f_checked(ell, eta, rho).unwrap_or(f64::NAN)
}

/// [`f`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{f, f_checked};
///
/// assert_eq!(f_checked(1.0, 0.5, 2.0), Ok(f(1.0, 0.5, 2.0)));
/// ```
pub fn f_checked(ell: f64, eta: f64, rho: f64) -> Result<f64, SpecialError> {
    if steed::in_domain(ell, eta, rho) {
        return Ok(steed::waves(ell, eta, rho)?.f);
    }
    if ell == 0.0 && eta == 0.0 {
        return Ok(rho.sin());
    }
    Ok(f_complex_checked(ell.into(), eta.into(), rho.into())?.re)
}

/// Outgoing Coulomb normalization for complex parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{d_plus, d_plus_complex};
/// use num_complex::Complex64;
///
/// let v = d_plus_complex(Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0));
/// assert!((v - d_plus(1.0, 0.5)).norm() < 1e-15 * v.norm());
/// ```
pub fn d_plus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    Complex64::new(0.0, -2.0).powc(2.0 * ell + 1.0)
        * gamma_complex(ell + 1.0 + Complex64::i() * eta)
        * reciprocal_gamma(2.0 * ell + 2.0)
        / normalization_complex(ell, eta)
}

/// Outgoing Coulomb normalization for real parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{d_minus, d_plus};
/// use num_complex::Complex64;
///
/// // D⁺_0(0) = −2i, and D⁻ is the complex conjugate of D⁺ for real parameters.
/// assert!((d_plus(0.0, 0.0) - Complex64::new(0.0, -2.0)).norm() < 1e-14);
/// let v = d_plus(1.0, 0.5);
/// assert!((d_minus(1.0, 0.5) - v.conj()).norm() < 1e-14 * v.norm());
/// ```
pub fn d_plus(ell: f64, eta: f64) -> Complex64 {
    d_plus_complex(ell.into(), eta.into())
}

/// Incoming Coulomb normalization for complex parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{d_minus, d_minus_complex};
/// use num_complex::Complex64;
///
/// let v = d_minus_complex(Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0));
/// assert!((v - d_minus(1.0, 0.5)).norm() < 1e-15 * v.norm());
/// ```
pub fn d_minus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    Complex64::new(0.0, 2.0).powc(2.0 * ell + 1.0)
        * gamma_complex(ell + 1.0 - Complex64::i() * eta)
        * reciprocal_gamma(2.0 * ell + 2.0)
        / normalization_complex(ell, eta)
}

/// Incoming Coulomb normalization for real parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::d_minus;
/// use num_complex::Complex64;
///
/// // D⁻_0(0) = 2i
/// assert!((d_minus(0.0, 0.0) - Complex64::new(0.0, 2.0)).norm() < 1e-14);
/// ```
pub fn d_minus(ell: f64, eta: f64) -> Complex64 {
    d_minus_complex(ell.into(), eta.into())
}

/// Outgoing Coulomb wave for complex inputs.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{h_plus, h_plus_complex};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (0.5, 0.3, 2.0);
/// let v = h_plus_complex(ell.into(), eta.into(), rho.into());
/// assert!((v - h_plus(ell, eta, rho)).norm() < 1e-10 * v.norm());
/// ```
pub fn h_plus_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    h_plus_complex_checked(ell, eta, rho).unwrap_or(NAN)
}

/// [`h_plus_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{h_plus_complex, h_plus_complex_checked};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(h_plus_complex_checked(ell, eta, rho), Ok(h_plus_complex(ell, eta, rho)));
/// ```
pub fn h_plus_complex_checked(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
) -> Result<Complex64, SpecialError> {
    let imaginary = Complex64::i();
    Ok(d_plus_complex(ell, eta)
        * rho.powc(ell + 1.0)
        * (imaginary * rho).exp()
        * tricomi_u(
            ell + 1.0 + imaginary * eta,
            2.0 * ell + 2.0,
            -2.0 * imaginary * rho,
            FAMILY,
        )?)
}

fn h_plus_asymptotic_real(ell: f64, eta: f64, rho: f64) -> (Complex64, Complex64) {
    let a = Complex64::new(ell + 1.0, eta);
    let b = 2.0 * ell + 2.0;
    let z = Complex64::new(0.0, -2.0 * rho);
    let mut term = Complex64::new(1.0, 0.0);
    let mut sum = term;
    let mut derivative_sum = Complex64::new(0.0, 0.0);
    let mut smallest_term = term.norm();
    let mut best = (sum, derivative_sum);

    for n in 1..=1000 {
        let previous = (n - 1) as f64;
        term *= -(a + previous) * (a - b + n as f64) / (n as f64 * z);
        sum += term;
        derivative_sum -= n as f64 * term / rho;
        if term.norm() < smallest_term {
            smallest_term = term.norm();
            best = (sum, derivative_sum);
        } else {
            (sum, derivative_sum) = best;
            break;
        }
        if term.norm() <= f64::EPSILON * sum.norm().max(1.0) / 16.0 {
            break;
        }
    }

    let prefactor =
        d_plus(ell, eta) * rho.powf(ell + 1.0) * Complex64::from_polar(1.0, rho) * z.powc(-a);
    (
        prefactor * sum,
        prefactor * ((Complex64::i() - Complex64::i() * eta / rho) * sum + derivative_sum),
    )
}

fn advance_coulomb(
    ell: f64,
    eta: f64,
    rho: f64,
    value: Complex64,
    derivative: Complex64,
    step: f64,
) -> (Complex64, Complex64) {
    let acceleration = |position: f64, amplitude: Complex64| {
        (ell * (ell + 1.0) / position.powi(2) + 2.0 * eta / position - 1.0) * amplitude
    };
    let k1_value = derivative;
    let k1_derivative = acceleration(rho, value);
    let k2_value = derivative + step * k1_derivative / 2.0;
    let k2_derivative = acceleration(rho + step / 2.0, value + step * k1_value / 2.0);
    let k3_value = derivative + step * k2_derivative / 2.0;
    let k3_derivative = acceleration(rho + step / 2.0, value + step * k2_value / 2.0);
    let k4_value = derivative + step * k3_derivative;
    let k4_derivative = acceleration(rho + step, value + step * k3_value);
    (
        value + step * (k1_value + 2.0 * k2_value + 2.0 * k3_value + k4_value) / 6.0,
        derivative
            + step * (k1_derivative + 2.0 * k2_derivative + 2.0 * k3_derivative + k4_derivative)
                / 6.0,
    )
}

fn irregular_real(ell: f64, eta: f64, rho: f64) -> Result<f64, SpecialError> {
    if !(ell.is_finite() && eta.is_finite() && rho.is_finite()) {
        return Ok(f64::NAN);
    }
    if rho <= 0.0 {
        return Ok(g_complex_checked(ell.into(), eta.into(), rho.into())?.re);
    }

    // At the starting radius the Poincaré series is far into its stable
    // regime for ordinary f64 parameters. Integrating the real ODE backward
    // avoids the removable singularity of the Tricomi connection formula at
    // every integer b = 2ℓ+2.
    let mut position = rho.max(20.0 + 3.0 * (ell.abs() + eta.abs()));
    let (mut value, mut derivative) = h_plus_asymptotic_real(ell, eta, position);
    let mut steps = 0_usize;
    let max_steps = iteration_cap(MAX_ODE_STEPS);
    while position > rho {
        if steps == max_steps {
            return Err(SpecialError::no_convergence(FAMILY, max_steps));
        }
        let step_size = 0.003_f64.min(position / 300.0).min(position - rho);
        let step = -step_size;
        (value, derivative) = advance_coulomb(ell, eta, position, value, derivative, step);
        position += step;
        steps += 1;
    }
    Ok(value.re)
}

/// Outgoing Coulomb wave for real inputs.
///
/// Its real and imaginary parts are `G` and `F` and have the accuracy of
/// [`g`] and [`f`].
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::h_plus;
///
/// // H⁺ = G + iF; without the Coulomb field, H⁺_0(0, ρ) = e^{iρ}.
/// let rho = 1.3_f64;
/// let v = h_plus(0.0, 0.0, rho);
/// assert!((v.re - rho.cos()).abs() < 1e-14 && (v.im - rho.sin()).abs() < 1e-14);
/// ```
pub fn h_plus(ell: f64, eta: f64, rho: f64) -> Complex64 {
    h_plus_checked(ell, eta, rho).unwrap_or(NAN)
}

/// [`h_plus`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{h_plus, h_plus_checked};
///
/// assert_eq!(h_plus_checked(1.0, 0.5, 2.0), Ok(h_plus(1.0, 0.5, 2.0)));
/// ```
pub fn h_plus_checked(ell: f64, eta: f64, rho: f64) -> Result<Complex64, SpecialError> {
    if steed::in_domain(ell, eta, rho) {
        let waves = steed::waves(ell, eta, rho)?;
        return Ok(Complex64::new(waves.g, waves.f));
    }
    Ok(Complex64::new(
        irregular_real(ell, eta, rho)?,
        f_checked(ell, eta, rho)?,
    ))
}

/// Incoming Coulomb wave for complex inputs.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{h_minus, h_minus_complex};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (0.5, 0.3, 2.0);
/// let v = h_minus_complex(ell.into(), eta.into(), rho.into());
/// assert!((v - h_minus(ell, eta, rho)).norm() < 1e-10 * v.norm());
/// ```
pub fn h_minus_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    h_minus_complex_checked(ell, eta, rho).unwrap_or(NAN)
}

/// [`h_minus_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{h_minus_complex, h_minus_complex_checked};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(h_minus_complex_checked(ell, eta, rho), Ok(h_minus_complex(ell, eta, rho)));
/// ```
pub fn h_minus_complex_checked(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
) -> Result<Complex64, SpecialError> {
    let imaginary = Complex64::i();
    Ok(d_minus_complex(ell, eta)
        * rho.powc(ell + 1.0)
        * (-imaginary * rho).exp()
        * tricomi_u(
            ell + 1.0 - imaginary * eta,
            2.0 * ell + 2.0,
            2.0 * imaginary * rho,
            FAMILY,
        )?)
}

/// Incoming Coulomb wave for real inputs.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{h_minus, h_plus};
///
/// // H⁻ = G − iF is the complex conjugate of H⁺ for real arguments.
/// let v = h_minus(1.0, 0.5, 2.0);
/// assert!((v - h_plus(1.0, 0.5, 2.0).conj()).norm() < 1e-14 * v.norm());
/// ```
pub fn h_minus(ell: f64, eta: f64, rho: f64) -> Complex64 {
    h_minus_checked(ell, eta, rho).unwrap_or(NAN)
}

/// [`h_minus`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{h_minus, h_minus_checked};
///
/// assert_eq!(h_minus_checked(1.0, 0.5, 2.0), Ok(h_minus(1.0, 0.5, 2.0)));
/// ```
pub fn h_minus_checked(ell: f64, eta: f64, rho: f64) -> Result<Complex64, SpecialError> {
    Ok(h_plus_checked(ell, eta, rho)?.conj())
}

/// Regular Coulomb wave reconstructed from outgoing and incoming waves.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{f, f_imag};
///
/// let v = f_imag(1.0, 0.5, 2.0);
/// assert!((v - f(1.0, 0.5, 2.0)).abs() < 1e-14);
/// ```
pub fn f_imag(ell: f64, eta: f64, rho: f64) -> f64 {
    f_imag_checked(ell, eta, rho).unwrap_or(f64::NAN)
}

/// [`f_imag`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{f_imag, f_imag_checked};
///
/// assert_eq!(f_imag_checked(1.0, 0.5, 2.0), Ok(f_imag(1.0, 0.5, 2.0)));
/// ```
pub fn f_imag_checked(ell: f64, eta: f64, rho: f64) -> Result<f64, SpecialError> {
    f_checked(ell, eta, rho)
}

/// Regular Coulomb wave reconstructed from complex outgoing and incoming waves.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{f, f_imag_complex};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (0.5, 0.3, 2.0);
/// let v = f_imag_complex(ell.into(), eta.into(), rho.into());
/// assert!((v - Complex64::new(f(ell, eta, rho), 0.0)).norm() < 1e-10);
/// ```
pub fn f_imag_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    f_imag_complex_checked(ell, eta, rho).unwrap_or(NAN)
}

/// [`f_imag_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{f_imag_complex, f_imag_complex_checked};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(f_imag_complex_checked(ell, eta, rho), Ok(f_imag_complex(ell, eta, rho)));
/// ```
pub fn f_imag_complex_checked(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
) -> Result<Complex64, SpecialError> {
    Ok(
        (h_plus_complex_checked(ell, eta, rho)? - h_minus_complex_checked(ell, eta, rho)?)
            / (2.0 * Complex64::i()),
    )
}

/// Irregular Coulomb wave for real inputs.
///
/// For `rho > 0` and `ell > -1` this uses Steed's method, with an inward
/// Taylor-series integration of `G` inside the turning point; `F` then follows
/// from the Wronskian. Other arguments use the Tricomi-function representation
/// integrated from large `rho`, which the reference tables do not cover.
///
/// # Accuracy
///
/// Measured against mpmath for `ell` in {0, 0.5, 1, 2, 5, 10, 20}, `eta` in
/// [-50, 50] and `rho` in [1e-3, 1e3] (1092 points; see the crate README), the
/// relative error is at most 4.7e-13 for `G` and 2.0e-13 for [`f`], with
/// median 1.5e-15. The work grows linearly with `rho` for `rho` well beyond
/// `ell` and `eta`, because CF1 needs about `rho` terms.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::g;
///
/// // Without the Coulomb field, G_1(0, ρ) = cos ρ / ρ + sin ρ.
/// let rho = 2.0_f64;
/// assert!((g(1.0, 0.0, rho) - (rho.cos() / rho + rho.sin())).abs() < 1e-14);
///
/// // mpmath: G_1(0.5, 2) = 1.1982980556731445
/// assert!((g(1.0, 0.5, 2.0) - 1.198_298_055_673_144_5).abs() < 1e-14);
/// ```
pub fn g(ell: f64, eta: f64, rho: f64) -> f64 {
    g_checked(ell, eta, rho).unwrap_or(f64::NAN)
}

/// [`g`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{g, g_checked};
///
/// assert_eq!(g_checked(1.0, 0.5, 2.0), Ok(g(1.0, 0.5, 2.0)));
/// ```
pub fn g_checked(ell: f64, eta: f64, rho: f64) -> Result<f64, SpecialError> {
    if steed::in_domain(ell, eta, rho) {
        return Ok(steed::waves(ell, eta, rho)?.g);
    }
    irregular_real(ell, eta, rho)
}

/// Irregular Coulomb wave for complex inputs.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{g, g_complex};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (0.5, 0.3, 2.0);
/// let v = g_complex(ell.into(), eta.into(), rho.into());
/// assert!((v - Complex64::new(g(ell, eta, rho), 0.0)).norm() < 1e-10);
/// ```
pub fn g_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    g_complex_checked(ell, eta, rho).unwrap_or(NAN)
}

/// [`g_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{g_complex, g_complex_checked};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(g_complex_checked(ell, eta, rho), Ok(g_complex(ell, eta, rho)));
/// ```
pub fn g_complex_checked(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
) -> Result<Complex64, SpecialError> {
    Ok(0.5 * (h_plus_complex_checked(ell, eta, rho)? + h_minus_complex_checked(ell, eta, rho)?))
}

/// Regularized confluent hypergeometric function `M(a,b,z) / Gamma(b)`.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::m_regularized;
/// use num_complex::Complex64;
///
/// let z = Complex64::new(0.4, 0.3);
/// // M(1, 1, z) = e^z and Γ(1) = 1.
/// let one = Complex64::new(1.0, 0.0);
/// assert!((m_regularized(one, one, z) - z.exp()).norm() < 1e-15);
/// // The regularized function stays finite at b = 0: M(1, 0, z)/Γ(0) = z e^z.
/// let zero = Complex64::new(0.0, 0.0);
/// assert!((m_regularized(one, zero, z) - z * z.exp()).norm() < 1e-15);
/// ```
pub fn m_regularized(alpha: Complex64, beta: Complex64, argument: Complex64) -> Complex64 {
    m_regularized_checked(alpha, beta, argument).unwrap_or(NAN)
}

/// [`m_regularized`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{m_regularized, m_regularized_checked};
/// use num_complex::Complex64;
///
/// let (alpha, beta, z) = (Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(m_regularized_checked(alpha, beta, z), Ok(m_regularized(alpha, beta, z)));
/// ```
pub fn m_regularized_checked(
    alpha: Complex64,
    beta: Complex64,
    argument: Complex64,
) -> Result<Complex64, SpecialError> {
    kummer_m_regularized(alpha, beta, argument, FAMILY)
}

/// Modified Coulomb function for complex inputs.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{phi, phi_complex};
/// use num_complex::Complex64;
///
/// let v = phi_complex(
///     Complex64::new(1.0, 0.0),
///     Complex64::new(0.5, 0.0),
///     Complex64::new(2.0, 0.0),
/// );
/// assert_eq!(v, phi(1.0, 0.5, 2.0));
/// ```
pub fn phi_complex(ell: Complex64, eta: Complex64, rho: Complex64) -> Complex64 {
    phi_complex_checked(ell, eta, rho).unwrap_or(NAN)
}

/// [`phi_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{phi_complex, phi_complex_checked};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(phi_complex_checked(ell, eta, rho), Ok(phi_complex(ell, eta, rho)));
/// ```
pub fn phi_complex_checked(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
) -> Result<Complex64, SpecialError> {
    Ok((2.0 * eta * rho).powc(ell + 1.0)
        * (Complex64::i() * rho).exp()
        * m_regularized_checked(
            ell + 1.0 + Complex64::i() * eta,
            2.0 * ell + 2.0,
            -2.0 * Complex64::i() * rho,
        )?)
}

/// Modified Coulomb function for real inputs.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{f, normalization, phi};
///
/// // F_ℓ(η, ρ) = C_ℓ(η) Γ(2ℓ+2) / (2η)^{ℓ+1} Φ_ℓ(η, ρ); Γ(4) = 6.
/// let (ell, eta, rho) = (1.0, 0.5, 2.0);
/// let factor = normalization(ell, eta) * 6.0 / (2.0 * eta).powf(ell + 1.0);
/// let v = phi(ell, eta, rho);
/// assert!((factor * v.re - f(ell, eta, rho)).abs() < 1e-13);
/// assert!(v.im.abs() < 1e-13);
/// ```
pub fn phi(ell: f64, eta: f64, rho: f64) -> Complex64 {
    phi_checked(ell, eta, rho).unwrap_or(NAN)
}

/// [`phi`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{phi, phi_checked};
///
/// assert_eq!(phi_checked(1.0, 0.5, 2.0), Ok(phi(1.0, 0.5, 2.0)));
/// ```
pub fn phi_checked(ell: f64, eta: f64, rho: f64) -> Result<Complex64, SpecialError> {
    phi_complex_checked(ell.into(), eta.into(), rho.into())
}

/// Gamow auxiliary product for nonnegative integer or half-integer `ell`;
/// NaN for other `ell` and for `eta = 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::gamow_w;
///
/// // w_ℓ(η) = ∏ (1 + j²/η²) over j = 0, 1, …, ℓ (or j = 1/2, 3/2, …, ℓ).
/// assert!((gamow_w(2.0, 1.0) - 10.0).abs() < 1e-14);
/// assert!((gamow_w(0.5, 2.0) - 1.0625).abs() < 1e-14);
/// ```
pub fn gamow_w(ell: f64, eta: f64) -> f64 {
    gamow_w_complex(ell, eta.into()).re
}

/// Gamow auxiliary product with a complex Coulomb parameter; NaN as for
/// [`gamow_w`].
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::gamow_w_complex;
/// use num_complex::Complex64;
///
/// // With η = i the factor 1 + j²/η² becomes 1 − j².
/// let v = gamow_w_complex(0.5, Complex64::i());
/// assert!((v - Complex64::new(0.75, 0.0)).norm() < 1e-15);
/// ```
pub fn gamow_w_complex(ell: f64, eta: Complex64) -> Complex64 {
    let half_integer = ell >= 0.0 && ell.is_finite() && 2.0 * ell == (2.0 * ell).round();
    if eta == Complex64::new(0.0, 0.0) || !half_integer {
        return NAN;
    }
    let start = if ell == ell.round() { 0.0 } else { 0.5 };
    let factors = (ell - start) as usize + 1;
    (0..factors)
        .map(|index| start + index as f64)
        .fold(Complex64::new(1.0, 0.0), |product, j| {
            product * (1.0 + j * j / (eta * eta))
        })
}

/// Complex `w+` Coulomb auxiliary function.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{gamow_w, w_plus};
///
/// // For integer ℓ and real η, w⁺ equals the Gamow product.
/// let v = w_plus(1.0, 0.5);
/// assert!((v.re - gamow_w(1.0, 0.5)).abs() < 1e-13 && v.im.abs() < 1e-13);
/// ```
pub fn w_plus(ell: f64, eta: f64) -> Complex64 {
    w_plus_complex(ell.into(), eta.into())
}

/// Complex `w+` Coulomb auxiliary function for complex parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{w_plus, w_plus_complex};
/// use num_complex::Complex64;
///
/// let v = w_plus_complex(Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0));
/// assert_eq!(v, w_plus(1.0, 0.5));
/// ```
pub fn w_plus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    let i_eta = Complex64::i() * eta;
    gamma_complex(ell + 1.0 + Complex64::i() * eta)
        / (i_eta.powc(2.0 * ell + 1.0) * gamma_complex(-ell + Complex64::i() * eta))
}

/// Complex `w-` Coulomb auxiliary function.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{gamow_w, w_minus};
///
/// // For integer ℓ and real η, w⁻ equals the Gamow product.
/// let v = w_minus(1.0, 0.5);
/// assert!((v.re - gamow_w(1.0, 0.5)).abs() < 1e-13 && v.im.abs() < 1e-13);
/// ```
pub fn w_minus(ell: f64, eta: f64) -> Complex64 {
    w_minus_complex(ell.into(), eta.into())
}

/// Complex `w-` Coulomb auxiliary function for complex parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{w_minus, w_minus_complex};
/// use num_complex::Complex64;
///
/// let v = w_minus_complex(Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0));
/// assert_eq!(v, w_minus(1.0, 0.5));
/// ```
pub fn w_minus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    let minus_i_eta = -Complex64::i() * eta;
    gamma_complex(ell + 1.0 - Complex64::i() * eta)
        / (minus_i_eta.powc(2.0 * ell + 1.0) * gamma_complex(-ell - Complex64::i() * eta))
}

/// Complex `h+` auxiliary (distinct from the outgoing wave [`h_plus`]).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{aux_g, aux_h_plus};
/// use std::f64::consts::PI;
///
/// // For integer ℓ and η > 0, h⁺ = g + iπ / (e^{2πη} − 1).
/// let (ell, eta) = (1.0, 0.5);
/// let v = aux_h_plus(ell, eta);
/// assert!((v.re - aux_g(ell, eta)).abs() < 1e-14);
/// assert!((v.im - PI / (2.0 * PI * eta).exp_m1()).abs() < 1e-14);
/// ```
pub fn aux_h_plus(ell: f64, eta: f64) -> Complex64 {
    aux_h_plus_complex(ell.into(), eta.into())
}

/// Complex `h+` auxiliary for complex parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{aux_h_plus, aux_h_plus_complex};
/// use num_complex::Complex64;
///
/// let v = aux_h_plus_complex(Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0));
/// assert_eq!(v, aux_h_plus(1.0, 0.5));
/// ```
pub fn aux_h_plus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    let i_eta = Complex64::i() * eta;
    0.5 * (digamma_complex(ell + 1.0 + i_eta) + digamma_complex(-ell + i_eta)) - i_eta.ln()
}

/// Complex `h-` auxiliary (distinct from the incoming wave [`h_minus`]).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{aux_h_minus, aux_h_plus};
///
/// // h⁻ is the complex conjugate of h⁺ for real parameters.
/// let v = aux_h_minus(1.0, 0.5);
/// assert!((v - aux_h_plus(1.0, 0.5).conj()).norm() < 1e-14);
/// ```
pub fn aux_h_minus(ell: f64, eta: f64) -> Complex64 {
    aux_h_minus_complex(ell.into(), eta.into())
}

/// Complex `h-` auxiliary for complex parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{aux_h_minus, aux_h_minus_complex};
/// use num_complex::Complex64;
///
/// let v = aux_h_minus_complex(Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0));
/// assert_eq!(v, aux_h_minus(1.0, 0.5));
/// ```
pub fn aux_h_minus_complex(ell: Complex64, eta: Complex64) -> Complex64 {
    let minus_i_eta = -Complex64::i() * eta;
    0.5 * (digamma_complex(ell + 1.0 + minus_i_eta) + digamma_complex(-ell + minus_i_eta))
        - minus_i_eta.ln()
}

/// Real digamma-based Coulomb auxiliary function; NaN for `eta = 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::aux_g;
///
/// // g_ℓ(η) = Re ψ(ℓ + 1 + iη) − ln η; mpmath: g_1(0.5) = 1.1642608233304860
/// assert!((aux_g(1.0, 0.5) - 1.164_260_823_330_486).abs() < 1e-13);
/// ```
pub fn aux_g(ell: f64, eta: f64) -> f64 {
    aux_g_complex(ell.into(), eta.into())
}

/// Real-valued `g` auxiliary evaluated with complex parameters.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{aux_g, aux_g_complex};
/// use num_complex::Complex64;
///
/// let v = aux_g_complex(Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0));
/// assert_eq!(v, aux_g(1.0, 0.5));
/// ```
pub fn aux_g_complex(ell: Complex64, eta: Complex64) -> f64 {
    if eta == Complex64::new(0.0, 0.0) {
        return f64::NAN;
    }
    (0.5 * (digamma_complex(ell + 1.0 + Complex64::i() * eta)
        + digamma_complex(ell + 1.0 - Complex64::i() * eta)))
    .re - eta.norm().ln()
}

/// The central-difference step: `step`, or `ε^(1/3)` when it is `None`;
/// `None` if the given step is not finite and positive.
fn finite_difference_step(step: Option<f64>) -> Option<f64> {
    let step = step.unwrap_or(f64::EPSILON.cbrt());
    (step.is_finite() && step > 0.0).then_some(step)
}

/// Order derivative `∂Φ_ℓ(η, ρ)/∂ℓ` of [`phi`].
///
/// `_step` is ignored: the derivative is computed exactly.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::phi_dot;
///
/// // mpmath: ∂Φ_ℓ/∂ℓ at (ℓ, η, ρ) = (1, 0.5, 2) is −1.3983721906603922
/// let v = phi_dot(1.0, 0.5, 2.0, None);
/// assert!((v.re + 1.398_372_190_660_392_2).abs() < 1e-13 && v.im.abs() < 1e-13);
/// ```
pub fn phi_dot(ell: f64, eta: f64, rho: f64, _step: Option<f64>) -> Complex64 {
    phi_dot_checked(ell, eta, rho, None).unwrap_or(NAN)
}

/// [`phi_dot`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{phi_dot, phi_dot_checked};
///
/// assert_eq!(phi_dot_checked(1.0, 0.5, 2.0, None), Ok(phi_dot(1.0, 0.5, 2.0, None)));
/// ```
pub fn phi_dot_checked(
    ell: f64,
    eta: f64,
    rho: f64,
    _step: Option<f64>,
) -> Result<Complex64, SpecialError> {
    phi_dot_complex_checked(ell.into(), eta.into(), rho.into(), None)
}

/// Order derivative `∂Φ_ℓ(η, ρ)/∂ℓ` of [`phi_complex`].
///
/// With `Φ_ℓ = (2ηρ)^{ℓ+1} e^{iρ} M̃(ℓ+1+iη, 2ℓ+2, −2iρ)` and `M̃ = M/Γ(b)`,
/// the derivative is `(2ηρ)^{ℓ+1} e^{iρ} [ln(2ηρ) M̃ + ∂M̃/∂a + 2 ∂M̃/∂b]`,
/// with the parameter derivatives of `M̃` summed from its power series.
///
/// `_step` is ignored: the derivative is computed exactly.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{phi_dot, phi_dot_complex};
/// use num_complex::Complex64;
///
/// let v = phi_dot_complex(
///     Complex64::new(1.0, 0.0),
///     Complex64::new(0.5, 0.0),
///     Complex64::new(2.0, 0.0),
///     None,
/// );
/// assert_eq!(v, phi_dot(1.0, 0.5, 2.0, None));
/// ```
pub fn phi_dot_complex(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
    _step: Option<f64>,
) -> Complex64 {
    phi_dot_complex_checked(ell, eta, rho, None).unwrap_or(NAN)
}

/// [`phi_dot_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{phi_dot_complex, phi_dot_complex_checked};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(phi_dot_complex_checked(ell, eta, rho, None), Ok(phi_dot_complex(ell, eta, rho, None)));
/// ```
pub fn phi_dot_complex_checked(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
    _step: Option<f64>,
) -> Result<Complex64, SpecialError> {
    let imaginary = Complex64::i();
    let base = 2.0 * eta * rho;
    let [m, dm_da, dm_db] = kummer_m_regularized_with_parameter_derivatives(
        ell + 1.0 + imaginary * eta,
        2.0 * ell + 2.0,
        -2.0 * imaginary * rho,
        FAMILY,
    )?;
    Ok(base.powc(ell + 1.0) * (imaginary * rho).exp() * (base.ln() * m + dm_da + 2.0 * dm_db))
}

/// Central order derivative of [`f`].
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::f_dot;
///
/// // mpmath: ∂F_ℓ/∂ℓ at (ℓ, η, ρ) = (1, 0.5, 2) is −0.45773255019706574
/// let v = f_dot(1.0, 0.5, 2.0, None);
/// assert!((v + 0.457_732_550_197_065_74).abs() < 1e-8);
/// ```
pub fn f_dot(ell: f64, eta: f64, rho: f64, step: Option<f64>) -> f64 {
    f_dot_checked(ell, eta, rho, step).unwrap_or(f64::NAN)
}

/// [`f_dot`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{f_dot, f_dot_checked};
///
/// assert_eq!(f_dot_checked(1.0, 0.5, 2.0, None), Ok(f_dot(1.0, 0.5, 2.0, None)));
/// ```
pub fn f_dot_checked(ell: f64, eta: f64, rho: f64, step: Option<f64>) -> Result<f64, SpecialError> {
    let Some(step) = finite_difference_step(step) else {
        return Ok(f64::NAN);
    };
    Ok((f_checked(ell + step, eta, rho)? - f_checked(ell - step, eta, rho)?) / (2.0 * step))
}

/// Central order derivative of [`f_complex`].
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::f_dot_complex;
/// use num_complex::Complex64;
///
/// // mpmath: ∂F_ℓ/∂ℓ at (ℓ, η, ρ) = (1, 0.5, 2) is −0.45773255019706574
/// let v = f_dot_complex(
///     Complex64::new(1.0, 0.0),
///     Complex64::new(0.5, 0.0),
///     Complex64::new(2.0, 0.0),
///     None,
/// );
/// assert!((v.re + 0.457_732_550_197_065_74).abs() < 1e-8 && v.im.abs() < 1e-8);
/// ```
pub fn f_dot_complex(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
    step: Option<f64>,
) -> Complex64 {
    f_dot_complex_checked(ell, eta, rho, step).unwrap_or(NAN)
}

/// [`f_dot_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{f_dot_complex, f_dot_complex_checked};
/// use num_complex::Complex64;
///
/// let (ell, eta, rho) = (Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(f_dot_complex_checked(ell, eta, rho, None), Ok(f_dot_complex(ell, eta, rho, None)));
/// ```
pub fn f_dot_complex_checked(
    ell: Complex64,
    eta: Complex64,
    rho: Complex64,
    step: Option<f64>,
) -> Result<Complex64, SpecialError> {
    let Some(step) = finite_difference_step(step) else {
        return Ok(NAN);
    };
    Ok(
        (f_complex_checked(ell + step, eta, rho)? - f_complex_checked(ell - step, eta, rho)?)
            / (2.0 * step),
    )
}

/// Coulomb auxiliary `Psi`,
/// `Ψ_ℓ(η, ρ) = w_ℓ(η) ∂Φ_ℓ/∂ℓ / 2 + ∂Φ_{−ℓ−1}/∂ℓ / 2`, with [`gamow_w`] for
/// `w_ℓ` and [`phi_dot`] for the order derivatives; NaN where [`gamow_w`] is.
///
/// `_step` is ignored: the derivative is computed exactly.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::psi;
///
/// // mpmath: Ψ_1(0.5, 2) = −3.2653661390660610
/// let v = psi(1.0, 0.5, 2.0, None);
/// assert!((v.re + 3.265_366_139_066_061).abs() < 1e-13 && v.im.abs() < 1e-13);
/// ```
pub fn psi(ell: f64, eta: f64, rho: f64, _step: Option<f64>) -> Complex64 {
    psi_checked(ell, eta, rho, None).unwrap_or(NAN)
}

/// [`psi`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{psi, psi_checked};
///
/// assert_eq!(psi_checked(1.0, 0.5, 2.0, None), Ok(psi(1.0, 0.5, 2.0, None)));
/// ```
pub fn psi_checked(
    ell: f64,
    eta: f64,
    rho: f64,
    _step: Option<f64>,
) -> Result<Complex64, SpecialError> {
    Ok(
        gamow_w(ell, eta) * phi_dot_checked(ell, eta, rho, None)? / 2.0
            + phi_dot_checked(-ell - 1.0, eta, rho, None)? / 2.0,
    )
}

/// Coulomb auxiliary `Psi` with complex `eta` and `rho`.
///
/// `_step` is ignored: the derivative is computed exactly.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{psi, psi_complex};
/// use num_complex::Complex64;
///
/// let v = psi_complex(1.0, Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0), None);
/// assert!((v - psi(1.0, 0.5, 2.0, None)).norm() < 1e-12);
/// ```
pub fn psi_complex(ell: f64, eta: Complex64, rho: Complex64, _step: Option<f64>) -> Complex64 {
    psi_complex_checked(ell, eta, rho, None).unwrap_or(NAN)
}

/// [`psi_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{psi_complex, psi_complex_checked};
/// use num_complex::Complex64;
///
/// let (eta, rho) = (Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(psi_complex_checked(1.0, eta, rho, None), Ok(psi_complex(1.0, eta, rho, None)));
/// ```
pub fn psi_complex_checked(
    ell: f64,
    eta: Complex64,
    rho: Complex64,
    _step: Option<f64>,
) -> Result<Complex64, SpecialError> {
    Ok(
        gamow_w_complex(ell, eta) * phi_dot_complex_checked(ell.into(), eta, rho, None)? / 2.0
            + phi_dot_complex_checked((-ell - 1.0).into(), eta, rho, None)? / 2.0,
    )
}

/// Coulomb auxiliary `I`, `I_ℓ(η, ρ) = C_ℓ(η) Γ(2ℓ+2) Ψ_ℓ(η, ρ) / (2η)^{ℓ+1}`,
/// with [`normalization`] for `C_ℓ`, [`psi`] for `Ψ_ℓ` and the principal
/// branch of the power for `η < 0`.
///
/// `_step` is ignored: the derivative is computed exactly.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::i;
///
/// // mpmath: I_1(0.5, 2) = −2.7504022966664935
/// let v = i(1.0, 0.5, 2.0, None);
/// assert!((v.re + 2.750_402_296_666_493_5).abs() < 1e-13 && v.im.abs() < 1e-13);
/// ```
pub fn i(ell: f64, eta: f64, rho: f64, _step: Option<f64>) -> Complex64 {
    i_checked(ell, eta, rho, None).unwrap_or(NAN)
}

/// [`i`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{i, i_checked};
///
/// assert_eq!(i_checked(1.0, 0.5, 2.0, None), Ok(i(1.0, 0.5, 2.0, None)));
/// ```
pub fn i_checked(
    ell: f64,
    eta: f64,
    rho: f64,
    _step: Option<f64>,
) -> Result<Complex64, SpecialError> {
    Ok(normalization(ell, eta) * libm::tgamma(2.0 * ell + 2.0)
        / Complex64::from(2.0 * eta).powf(ell + 1.0)
        * psi_checked(ell, eta, rho, None)?)
}

/// Coulomb auxiliary `I` with complex `eta` and `rho`.
///
/// `_step` is ignored: the derivative is computed exactly.
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{i, i_complex};
/// use num_complex::Complex64;
///
/// let v = i_complex(1.0, Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0), None);
/// assert!((v - i(1.0, 0.5, 2.0, None)).norm() < 1e-12);
/// ```
pub fn i_complex(ell: f64, eta: Complex64, rho: Complex64, _step: Option<f64>) -> Complex64 {
    i_complex_checked(ell, eta, rho, None).unwrap_or(NAN)
}

/// [`i_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::coulomb::{i_complex, i_complex_checked};
/// use num_complex::Complex64;
///
/// let (eta, rho) = (Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0));
/// assert_eq!(i_complex_checked(1.0, eta, rho, None), Ok(i_complex(1.0, eta, rho, None)));
/// ```
pub fn i_complex_checked(
    ell: f64,
    eta: Complex64,
    rho: Complex64,
    _step: Option<f64>,
) -> Result<Complex64, SpecialError> {
    Ok(
        normalization_complex(ell.into(), eta) * gamma_complex((2.0 * ell + 2.0).into())
            / (2.0 * eta).powf(ell + 1.0)
            * psi_complex_checked(ell, eta, rho, None)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::with_iteration_cap;

    #[test]
    fn iteration_caps_report_no_convergence() {
        let expected = Some(SpecialError::no_convergence(FAMILY, 1));
        let (ell, eta, rho) = (
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(2.0, 0.0),
        );
        with_iteration_cap(1, || {
            // Steed's method.
            assert_eq!(f_checked(1.0, 0.5, 2.0).err(), expected);
            assert_eq!(g_checked(1.0, 5.0, 0.5).err(), expected);
            assert!(h_plus(1.0, 0.5, 2.0).re.is_nan());
            // The backward integration outside Steed's domain.
            assert_eq!(g_checked(-1.5, 0.5, 2.0).err(), expected);
            // The Kummer and Tricomi functions.
            assert_eq!(f_complex_checked(ell, eta, rho).err(), expected);
            assert_eq!(h_plus_complex_checked(ell, eta, rho).err(), expected);
            assert_eq!(phi_checked(1.0, 0.5, 2.0).err(), expected);
            assert_eq!(phi_dot_checked(1.0, 0.5, 2.0, None).err(), expected);
            assert_eq!(i_checked(1.0, 0.5, 2.0, None).err(), expected);
            assert!(i(1.0, 0.5, 2.0, None).re.is_nan());
        });
    }

    #[test]
    fn arguments_outside_the_domain_give_nan() {
        assert!(eta(0.0, 1.0).is_nan());
        assert!(eta_from_energy(0.0).is_nan());
        assert!(phase(1.0, 0.5, -1.0).is_nan());
        assert!(gamow_w(0.3, 1.0).is_nan());
        assert!(gamow_w(1.0, 0.0).is_nan());
        assert!(aux_g(1.0, 0.0).is_nan());
        assert!(f_dot(1.0, 0.5, 2.0, Some(-1.0)).is_nan());
        assert_eq!(g_checked(f64::NAN, 0.5, 2.0).map(f64::is_nan), Ok(true));
        assert_eq!(
            f_checked(1.0, 0.5, f64::INFINITY).map(f64::is_nan),
            Ok(true)
        );
    }
}
