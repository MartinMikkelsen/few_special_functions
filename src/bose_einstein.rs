//! Bose–Einstein integrals for integer and half-integer orders.
//!
//! Ported from `src/BoseEinstein.jl` of FewSpecialFunctions.jl
//! (<https://github.com/MartinMikkelsen/FewSpecialFunctions.jl>, MIT license),
//! commit b909e7b22696e942762f800ddad34a70d5b9d51d: its closed forms and its
//! two convergent series, the series in the fugacity `e^η` and the expansion
//! about `η = 0` (DLMF 25.12.12). The Julia code evaluates `Float64` orders up
//! to 39/2 with T. Fukushima's minimax rational approximations instead; those
//! are not part of this crate.
//!
//! # Iteration caps and tolerances
//!
//! Each series stops when its remainder (fugacity series) or three successive
//! terms (expansion about zero) fall below `ε/4` of the sum, with a cap of
//! 100 000 terms. Reaching a cap gives [`SpecialError::NoConvergence`] from the
//! `*_checked` functions and NaN from the others.

use crate::SpecialError;
use crate::error::iteration_cap;
use crate::numerics::zeta_real;

const FAMILY: &str = "bose_einstein";
/// Cap on the terms of either series.
const MAX_TERMS: usize = 100_000;

/// Whether `k` is an integer or half-integer `>= -9/2` and `eta <= 0`.
fn in_domain(k: f64, eta: f64) -> bool {
    k.is_finite() && k >= -4.5 && 2.0 * k == (2.0 * k).round() && eta <= 0.0
}

fn fugacity_scaled(k: f64, eta: f64) -> Result<f64, SpecialError> {
    let z = eta.exp();
    let q = -eta.exp_m1();
    let mut power = 1.0;
    let mut total = 1.0;
    let mut correction = 0.0;
    let max_terms = iteration_cap(MAX_TERMS);
    for n in 2..=max_terms {
        power *= z;
        let term = power * (n as f64).powf(-k - 1.0);
        let adjusted = term - correction;
        let updated = total + adjusted;
        correction = (updated - total) - adjusted;
        total = updated;

        let tail = if k >= -1.0 {
            let geometric = term * z / q;
            if k > 0.0 {
                geometric.min(term * n as f64 / k)
            } else {
                geometric
            }
        } else {
            let ratio = z * (1.0 + 1.0 / n as f64).powf(-k - 1.0);
            if ratio < 1.0 {
                term * ratio / (1.0 - ratio)
            } else {
                f64::INFINITY
            }
        };
        if tail <= f64::EPSILON * total / 4.0 {
            return Ok(total);
        }
    }
    Err(SpecialError::no_convergence(FAMILY, max_terms))
}

fn near_zero_series(k: f64, eta: f64) -> Result<f64, SpecialError> {
    let integer_order = k >= 0.0 && k == k.round();
    let mut total = if integer_order {
        let order = k as usize;
        let harmonic = (1..=order).map(|j| 1.0 / j as f64).sum::<f64>();
        eta.powi(order as i32) / libm::tgamma(k + 1.0) * (harmonic - (-eta).ln())
    } else {
        libm::tgamma(-k) * (-eta).powf(k)
    };
    if total.is_infinite() {
        return Ok(total);
    }

    let mut coefficient = 1.0;
    let mut correction = 0.0;
    let mut small_terms = 0;
    let max_terms = iteration_cap(MAX_TERMS);
    for m in 0..max_terms {
        if !integer_order || m as f64 != k {
            let term = zeta_real(k + 1.0 - m as f64) * coefficient;
            let adjusted = term - correction;
            let updated = total + adjusted;
            correction = (updated - total) - adjusted;
            total = updated;
            if term.abs() <= f64::EPSILON * total.abs() / 4.0 {
                small_terms += 1;
            } else {
                small_terms = 0;
            }
            if m as f64 > k + 1.0 && small_terms >= 3 {
                return Ok(total);
            }
        }
        coefficient *= eta / (m + 1) as f64;
    }
    Err(SpecialError::no_convergence(FAMILY, max_terms))
}

fn normalized(k: f64, eta: f64) -> Result<f64, SpecialError> {
    if eta == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if eta == 0.0 {
        return Ok(if k > 0.0 {
            zeta_real(k + 1.0)
        } else {
            f64::INFINITY
        });
    }
    if k == 0.0 {
        return Ok(if eta < -std::f64::consts::LN_2 {
            -(-eta.exp()).ln_1p()
        } else {
            -(-eta.exp_m1()).ln()
        });
    }
    let z = eta.exp();
    let q = -eta.exp_m1();
    // Li_{1-n}(z) for n = 1, 2, 3, 4 is rational in z.
    match k {
        -1.0 => return Ok(z / q),
        -2.0 => return Ok(z / q.powi(2)),
        -3.0 => return Ok(z * (1.0 + z) / q.powi(3)),
        -4.0 => return Ok(z * (1.0 + 4.0 * z + z * z) / q.powi(4)),
        _ => {}
    }
    if eta <= -1.0 || k >= 20.0 {
        return Ok(z * fugacity_scaled(k, eta)?);
    }
    near_zero_series(k, eta)
}

/// Normalized Bose–Einstein integral `Li_(k+1)(exp(eta))`.
///
/// Defined for integer and half-integer `k >= -4.5` and `eta <= 0`; returns
/// NaN elsewhere. Negative orders use the polylogarithm continuation.
///
/// # Examples
///
/// ```
/// use few_special_functions::bose_einstein::bose_einstein_integral_norm;
/// use std::f64::consts::PI;
///
/// // Li_2(1) = ζ(2) = π²/6
/// let v = bose_einstein_integral_norm(1.0, 0.0);
/// assert!((v - PI * PI / 6.0).abs() < 1e-14);
///
/// // Li_1(e^η) = −ln(1 − e^η)
/// let eta = -1.0_f64;
/// let v = bose_einstein_integral_norm(0.0, eta);
/// assert!((v + (-eta.exp()).ln_1p()).abs() < 1e-14);
/// ```
pub fn bose_einstein_integral_norm(k: f64, eta: f64) -> f64 {
    bose_einstein_integral_norm_checked(k, eta).unwrap_or(f64::NAN)
}

/// [`bose_einstein_integral_norm`], reporting a numerical failure instead of
/// returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a series reaches its cap (see the
/// module documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::bose_einstein::{
///     bose_einstein_integral_norm, bose_einstein_integral_norm_checked,
/// };
///
/// assert_eq!(
///     bose_einstein_integral_norm_checked(1.5, -0.5),
///     Ok(bose_einstein_integral_norm(1.5, -0.5))
/// );
/// ```
pub fn bose_einstein_integral_norm_checked(k: f64, eta: f64) -> Result<f64, SpecialError> {
    if !in_domain(k, eta) {
        return Ok(f64::NAN);
    }
    normalized(k, eta)
}

/// Unnormalized Bose–Einstein integral.
///
/// This is `Gamma(k+1) * bose_einstein_integral_norm(k, eta)`. It is defined
/// for integer and half-integer `k > -1`, where the defining integral exists,
/// and `eta <= 0`; it returns NaN elsewhere.
///
/// # Examples
///
/// ```
/// use few_special_functions::bose_einstein::bose_einstein_integral;
/// use std::f64::consts::PI;
///
/// // ∫₀^∞ t³ / (eᵗ − 1) dt = π⁴/15
/// let v = bose_einstein_integral(3.0, 0.0);
/// assert!((v - PI.powi(4) / 15.0).abs() < 1e-13);
/// ```
pub fn bose_einstein_integral(k: f64, eta: f64) -> f64 {
    bose_einstein_integral_checked(k, eta).unwrap_or(f64::NAN)
}

/// [`bose_einstein_integral`], reporting a numerical failure instead of
/// returning NaN.
///
/// # Errors
///
/// As for [`bose_einstein_integral_norm_checked`].
///
/// # Examples
///
/// ```
/// use few_special_functions::bose_einstein::{bose_einstein_integral, bose_einstein_integral_checked};
///
/// assert_eq!(bose_einstein_integral_checked(1.5, -0.5), Ok(bose_einstein_integral(1.5, -0.5)));
/// ```
pub fn bose_einstein_integral_checked(k: f64, eta: f64) -> Result<f64, SpecialError> {
    if !in_domain(k, eta) || k <= -1.0 {
        return Ok(f64::NAN);
    }
    if eta == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    let value = normalized(k, eta)?;
    if k <= 170.5 && value >= f64::MIN_POSITIVE {
        let scale = libm::tgamma(k + 1.0);
        if scale.is_finite() {
            return Ok(scale * value);
        }
    }
    let log_value = if eta < -1.0 {
        eta + fugacity_scaled(k, eta)?.ln()
    } else {
        value.ln()
    };
    Ok((libm::lgamma(k + 1.0) + log_value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::with_iteration_cap;

    #[test]
    fn iteration_caps_report_no_convergence() {
        let expected = Some(SpecialError::no_convergence(FAMILY, 1));
        with_iteration_cap(1, || {
            // eta = -2 takes the fugacity series, eta = -0.5 the expansion about 0.
            for eta in [-2.0, -0.5] {
                assert_eq!(
                    bose_einstein_integral_norm_checked(1.5, eta).err(),
                    expected
                );
                assert_eq!(bose_einstein_integral_checked(1.5, eta).err(), expected);
                assert!(bose_einstein_integral_norm(1.5, eta).is_nan());
                assert!(bose_einstein_integral(1.5, eta).is_nan());
            }
        });
    }
}
