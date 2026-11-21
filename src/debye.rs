//! Generalized Debye functions D_n(β, x).
//!
//! Ported from `src/Debye.jl` of FewSpecialFunctions.jl
//! (<https://github.com/MartinMikkelsen/FewSpecialFunctions.jl>, MIT license),
//! commit b909e7b22696e942762f800ddad34a70d5b9d51d.
//!
//! # Iteration cap and tolerance
//!
//! The integral is evaluated by adaptive Gauss–Kronrod quadrature to a
//! relative tolerance `max(tol, 8 ε)` (`tol = 1e-35` unless given), with at
//! most 2000 subintervals unless given. Reaching the cap gives
//! [`SpecialError::NoConvergence`] from the `*_checked` functions and NaN from
//! the others.

use crate::SpecialError;
use crate::numerics::integrate;
use std::num::NonZeroUsize;

const FAMILY: &str = "debye";
/// Subinterval cap of [`debye_function`].
const DEFAULT_SUBINTERVALS: NonZeroUsize = NonZeroUsize::MIN.saturating_add(1999);

/// Generalized Debye function D_n(β, x).
///
/// Defined as
///
/// ```text
/// D_n(β, x) = (n / xⁿ) ∫₀ˣ tⁿ / (eᵗ − 1)^β dt
/// ```
///
/// The standard Debye function used in solid-state physics is the special case
/// β = 1. At zero the limiting value is zero for β < 1, one for β = 1,
/// and positive infinity for β > 1.
///
/// Evaluated by adaptive Gauss–Kronrod quadrature of the defining integral
/// after a change of variables that removes the endpoint singularity. See also
/// <https://doi.org/10.1007/s10765-007-0256-1>.
///
/// Returns NaN unless `n > 0`, `0 < β < n + 1` and `x ≥ 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::debye::debye_function;
///
/// // D_n(β, 0) = 1 by definition
/// assert!((debye_function(2.0, 1.0, 0.0) - 1.0).abs() < 1e-14);
///
/// // D_n(β, x) is strictly less than 1 for x > 0
/// assert!(debye_function(1.0, 1.0, 1.0) < 1.0);
/// ```
pub fn debye_function(n: f64, beta: f64, x: f64) -> f64 {
    debye_function_checked(n, beta, x).unwrap_or(f64::NAN)
}

/// [`debye_function`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if the quadrature needs more than 2000
/// subintervals.
///
/// # Examples
///
/// ```
/// use few_special_functions::debye::{debye_function, debye_function_checked};
///
/// assert_eq!(debye_function_checked(3.0, 1.0, 2.0), Ok(debye_function(3.0, 1.0, 2.0)));
/// ```
pub fn debye_function_checked(n: f64, beta: f64, x: f64) -> Result<f64, SpecialError> {
    debye_function_tol_checked(n, beta, x, 1e-35, DEFAULT_SUBINTERVALS)
}

/// Convenience form of [`debye_function`] with order `n = 1`.
///
/// # Examples
///
/// ```
/// use few_special_functions::debye::debye_function_order_one;
/// use std::f64::consts::PI;
///
/// // D_1(1, x) = (1/x) ∫₀ˣ t/(eᵗ − 1) dt, which tends to π²/(6x) for large x.
/// let x = 50.0;
/// let v = debye_function_order_one(1.0, x);
/// assert!((v - PI * PI / (6.0 * x)).abs() < 1e-14 * v);
/// ```
pub fn debye_function_order_one(beta: f64, x: f64) -> f64 {
    debye_function(1.0, beta, x)
}

/// [`debye_function_order_one`], reporting a numerical failure instead of
/// returning NaN.
///
/// # Errors
///
/// As for [`debye_function_checked`].
///
/// # Examples
///
/// ```
/// use few_special_functions::debye::{debye_function_order_one, debye_function_order_one_checked};
///
/// assert_eq!(debye_function_order_one_checked(1.0, 2.0), Ok(debye_function_order_one(1.0, 2.0)));
/// ```
pub fn debye_function_order_one_checked(beta: f64, x: f64) -> Result<f64, SpecialError> {
    debye_function_checked(1.0, beta, x)
}

/// Like [`debye_function`] but with an explicit relative quadrature tolerance
/// `tol` (floored at `8 ε`) and a limit `max_subintervals` on the number of quadrature
/// subintervals.
///
/// # Examples
///
/// ```
/// use few_special_functions::debye::{debye_function, debye_function_tol};
/// use std::num::NonZeroUsize;
///
/// // Agrees with debye_function to within a few times the tolerance.
/// let subintervals = NonZeroUsize::new(500).expect("500 is nonzero");
/// let v1 = debye_function(2.0, 1.0, 1.0);
/// let v2 = debye_function_tol(2.0, 1.0, 1.0, 1e-8, subintervals);
/// assert!((v1 - v2).abs() < 1e-7);
/// ```
pub fn debye_function_tol(
    n: f64,
    beta: f64,
    x: f64,
    tol: f64,
    max_subintervals: NonZeroUsize,
) -> f64 {
    debye_function_tol_checked(n, beta, x, tol, max_subintervals).unwrap_or(f64::NAN)
}

/// [`debye_function_tol`], reporting a numerical failure instead of returning
/// NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if the quadrature needs more than
/// `max_subintervals` subintervals.
///
/// # Examples
///
/// ```
/// use few_special_functions::SpecialError;
/// use few_special_functions::debye::debye_function_tol_checked;
/// use std::num::NonZeroUsize;
///
/// let one = NonZeroUsize::MIN;
/// assert!(matches!(
///     debye_function_tol_checked(3.0, 1.0, 2.0, 1e-15, one),
///     Err(SpecialError::NoConvergence { .. })
/// ));
/// ```
pub fn debye_function_tol_checked(
    n: f64,
    beta: f64,
    x: f64,
    tol: f64,
    max_subintervals: NonZeroUsize,
) -> Result<f64, SpecialError> {
    let in_domain = n.is_finite()
        && n > 0.0
        && beta.is_finite()
        && beta > 0.0
        && beta < n + 1.0
        && x >= 0.0
        && tol.is_finite()
        && tol > 0.0;
    if !in_domain {
        return Ok(f64::NAN);
    }

    Ok(if x == 0.0 {
        if beta < 1.0 {
            0.0
        } else if beta == 1.0 {
            1.0
        } else {
            f64::INFINITY
        }
    } else if x.is_infinite() {
        0.0
    } else if beta == 1.0 && n < f64::EPSILON.sqrt() {
        1.0
    } else {
        positive_argument(n, beta, x, tol, max_subintervals.get())?
    })
}

/// D_n(β, x) for finite `x > 0` and parameters in the domain.
fn positive_argument(
    n: f64,
    beta: f64,
    x: f64,
    tol: f64,
    max_subintervals: usize,
) -> Result<f64, SpecialError> {
    let q = n + 1.0 - beta;
    let m = q.min(1.0);
    let scale = x.min((n / beta).max(1.0));
    let upper = (1.0 / (1.0 + scale / x)).powf(m);
    let integrand = |v: f64| {
        if v == 0.0 {
            return if q == m { 1.0 } else { 0.0 };
        }
        let s = v.powf(1.0 / m);
        let ratio = s / (1.0 - s);
        let t = scale * ratio;
        if t.is_infinite() {
            return 0.0;
        }
        let log_bose = if t == 0.0 {
            0.0
        } else if t <= 1.0 {
            (t / t.exp_m1()).ln()
        } else {
            t.ln() - t - (-(-t).exp_m1()).ln()
        };
        let log_power = if q == m { 0.0 } else { (q - m) * ratio.ln() };
        (log_power - (m + 1.0) * (-s).ln_1p() + beta * log_bose).exp()
    };
    let relative_tolerance = tol.max(8.0 * f64::EPSILON);
    let integral = integrate(
        integrand,
        0.0,
        upper,
        relative_tolerance,
        max_subintervals,
        FAMILY,
    )?;

    let result = (n / m) * integral * scale.powf(1.0 - beta) * (scale / x).powf(n);
    if result.is_finite() && result > 0.0 {
        return Ok(result);
    }

    Ok(
        ((n / m).ln() + (1.0 - beta) * scale.ln() + n * (scale.ln() - x.ln()) + integral.ln())
            .exp(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_zero() {
        assert_eq!(debye_function(1.0, 1.0, 0.0), 1.0);
        assert_eq!(debye_function(3.0, 2.0, 0.0), f64::INFINITY);
    }

    #[test]
    fn iteration_cap_reports_no_convergence() {
        crate::error::with_iteration_cap(1, || {
            assert_eq!(
                debye_function_checked(3.0, 1.0, 2.0),
                Err(SpecialError::no_convergence(FAMILY, 1))
            );
            assert!(debye_function(3.0, 1.0, 2.0).is_nan());
            assert!(debye_function_order_one(1.0, 2.0).is_nan());
        });
    }

    #[test]
    fn parameters_outside_the_domain_give_nan() {
        for (n, beta, x) in [(0.0, 1.0, 1.0), (1.0, 2.5, 1.0), (1.0, 1.0, -1.0)] {
            assert_eq!(
                debye_function_checked(n, beta, x).map(f64::is_nan),
                Ok(true)
            );
        }
    }
}
