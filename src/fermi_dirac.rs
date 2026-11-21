//! Complete Fermi-Dirac integrals.
//!
//! Written for this crate from the defining integral and its series in `eˣ`
//! (DLMF section 25.12); no code was ported.
//!
//! # Iteration caps and tolerances
//!
//! The series in `eˣ` stops when a term falls below `ε/4` of the sum, with a
//! cap of 10 000 terms. The double-exponential quadrature halves its step until
//! two levels agree to `1e-10` (the rule's error roughly squares with each
//! halving, so the finer level is accurate to rounding), with a cap of 10
//! halvings. Reaching a cap gives [`SpecialError::NoConvergence`] from the
//! `*_checked` functions and NaN from the others.

use crate::SpecialError;
use crate::error::iteration_cap;
use std::f64::consts::FRAC_PI_2;

const FAMILY: &str = "fermi_dirac";
/// Cap on the terms of the series in `eˣ`; `x = −1` needs about 30.
const MAX_SERIES_TERMS: usize = 10_000;
/// Cap on the step halvings of the double-exponential rule.
const MAX_HALVINGS: usize = 10;

/// F_j(x) for any order `j > -1`.
///
/// For `x <= -1` the series `F_j(x) = Γ(j+1) Σ_{k≥1} (−1)^{k+1} e^{kx} / k^{j+1}`
/// converges geometrically. Above that the defining integral is evaluated by
/// double-exponential quadrature: tanh-sinh on `[0, x]` and exp-sinh on
/// `[x, ∞)` for `x > 0`, exp-sinh on `[0, ∞)` otherwise. The integrands are
/// written in logarithmic form, so the `t^j` endpoint singularity and the
/// exponential tail neither overflow nor underflow before they are weighted.
fn general_order(j: f64, x: f64) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x == f64::INFINITY {
        return Ok(f64::INFINITY);
    }
    if x <= -1.0 {
        return Ok(libm::tgamma(j + 1.0) * alternating_series(j, x)?);
    }
    let order = j + 1.0;
    if x > 0.0 {
        // [0, x] with t = x σ(y), x − t = x σ(−y), y = π sinh s and σ the
        // logistic function; t^j dt = x^{j+1} σ(y)^{j+1} σ(−y) π cosh(s) ds.
        let inner = double_exponential(|s| {
            let y = std::f64::consts::PI * s.sinh();
            let log_sigma = -softplus(-y);
            let log_complement = -softplus(y);
            let distance = x * log_complement.exp();
            std::f64::consts::PI * s.cosh() * (order * log_sigma + log_complement).exp()
                / ((-distance).exp() + 1.0)
        })?;
        // [x, ∞) with t = x + w, w = exp(y), y = (π/2) sinh s.
        let outer = double_exponential(|s| {
            let y = FRAC_PI_2 * s.sinh();
            let w = y.exp();
            if !w.is_finite() {
                return 0.0;
            }
            FRAC_PI_2 * s.cosh() * (y + j * (x + w).ln() - w - softplus(-w)).exp()
        })?;
        Ok(x.powf(order) * inner + outer)
    } else {
        // [0, ∞) with t = exp(y), y = (π/2) sinh s; t − x ≥ 0.
        double_exponential(|s| {
            let y = FRAC_PI_2 * s.sinh();
            let excess = y.exp() - x;
            if !excess.is_finite() {
                return 0.0;
            }
            FRAC_PI_2 * s.cosh() * (order * y - excess - softplus(-excess)).exp()
        })
    }
}

/// `Σ_{k≥1} (−1)^{k+1} e^{kx} / k^{j+1}`, for `x <= -1`.
fn alternating_series(j: f64, x: f64) -> Result<f64, SpecialError> {
    let ratio = x.exp();
    let mut power = ratio;
    let mut sum = 0.0;
    let max_terms = iteration_cap(MAX_SERIES_TERMS);
    for k in 1..=max_terms {
        let k = k as f64;
        let term = power / k.powf(j + 1.0);
        sum += if k % 2.0 == 1.0 { term } else { -term };
        if term <= 0.25 * f64::EPSILON * sum.abs() {
            return Ok(sum);
        }
        power *= ratio;
    }
    Err(SpecialError::no_convergence(FAMILY, max_terms))
}

/// `ln(1 + e^z)` without overflow or loss of accuracy.
fn softplus(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

/// `∫ f(s) ds` over the real line for an integrand that decays double
/// exponentially, by the trapezoidal rule with step halving.
///
/// The error of the rule roughly squares with each halving, so once two levels
/// agree to 1e-10 the finer one is accurate to rounding.
fn double_exponential(f: impl Fn(f64) -> f64) -> Result<f64, SpecialError> {
    // Beyond |s| = 8 every integrand here is below the f64 range.
    const S_MAX: f64 = 8.0;
    let max_halvings = iteration_cap(MAX_HALVINGS);
    // Adds f(offset + k·spacing) for k = 0, 1, 2, … in one direction until
    // the terms no longer change `reference`.
    let ray = |offset: f64, spacing: f64, reference: f64| {
        let mut sum = 0.0;
        let mut s = offset;
        while s.abs() <= S_MAX {
            let value = f(s);
            sum += value;
            if value.abs() <= 1e-2 * f64::EPSILON * (reference + sum).abs() && s != offset {
                break;
            }
            s += spacing;
        }
        sum
    };

    let mut step = 0.5;
    let centre = f(0.0);
    let mut nodes = centre + ray(step, step, centre) + ray(-step, -step, centre);
    let mut estimate = step * nodes;
    for _ in 0..max_halvings {
        // The new nodes are the odd multiples of step/2.
        let half = 0.5 * step;
        nodes += ray(half, step, nodes) + ray(-half, -step, nodes);
        step = half;
        let refined = step * nodes;
        if (refined - estimate).abs() <= 1e-10 * refined.abs() {
            return Ok(refined);
        }
        estimate = refined;
    }
    Err(SpecialError::no_convergence(FAMILY, max_halvings))
}

/// Complete Fermi-Dirac integral F_j(x) = ∫₀^∞ tʲ / (exp(t-x) + 1) dt.
///
/// Arises in solid-state physics as a measure of carrier statistics in
/// semiconductors; the index `j` is typically a half-integer related to the
/// dimensionality of the band structure.
///
/// Uses the exact formula ln(1 + eˣ) for j = 0. Every other j uses the
/// alternating series in eˣ for x ≤ −1 and double-exponential quadrature of
/// the defining integral otherwise.
///
/// # Accuracy
///
/// Measured against mpmath for j ∈ [−0.5, 10] and x ∈ [−50, 10⁴] (see the
/// crate README): relative error below 10⁻¹⁴.
///
/// Returns NaN for `j < -0.5`.
///
/// # Examples
///
/// ```
/// use few_special_functions::fermi_dirac::fermi_dirac_integral;
///
/// // F_0(0) = ln(2) exactly
/// let v = fermi_dirac_integral(0.0, 0.0);
/// assert!((v - std::f64::consts::LN_2).abs() < 1e-14);
///
/// // F_{1/2}(0) ≈ 0.6781 (tabulated value)
/// let v = fermi_dirac_integral(0.5, 0.0);
/// assert!((v - 0.6781).abs() < 1e-4);
/// ```
pub fn fermi_dirac_integral(j: f64, x: f64) -> f64 {
    fermi_dirac_integral_checked(j, x).unwrap_or(f64::NAN)
}

/// [`fermi_dirac_integral`], reporting a numerical failure instead of
/// returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if the series or the quadrature reaches its
/// cap (see the module documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::fermi_dirac::{fermi_dirac_integral, fermi_dirac_integral_checked};
///
/// assert_eq!(fermi_dirac_integral_checked(1.5, 2.0), Ok(fermi_dirac_integral(1.5, 2.0)));
/// ```
pub fn fermi_dirac_integral_checked(j: f64, x: f64) -> Result<f64, SpecialError> {
    if j.is_nan() || j < -0.5 {
        return Ok(f64::NAN);
    }
    if j == 0.0 {
        Ok(softplus(x))
    } else {
        general_order(j, x)
    }
}

/// Normalized Fermi-Dirac integral F̃_j(x) = F_j(x) / Γ(j+1).
///
/// Dividing by Γ(j+1) gives a function that approaches eˣ as x → −∞ for
/// every order, and x^{j+1}/Γ(j+2) in the degenerate limit x → +∞. Useful when
/// comparing Fermi-Dirac statistics across different orders.
///
/// # Examples
///
/// ```
/// use few_special_functions::fermi_dirac::fermi_dirac_integral_norm;
///
/// // F̃_{1/2}(0) ≈ 0.7652
/// let v = fermi_dirac_integral_norm(0.5, 0.0);
/// assert!((v - 0.7652).abs() < 1e-4);
/// ```
pub fn fermi_dirac_integral_norm(j: f64, x: f64) -> f64 {
    fermi_dirac_integral_norm_checked(j, x).unwrap_or(f64::NAN)
}

/// [`fermi_dirac_integral_norm`], reporting a numerical failure instead of
/// returning NaN.
///
/// # Errors
///
/// As for [`fermi_dirac_integral_checked`].
///
/// # Examples
///
/// ```
/// use few_special_functions::fermi_dirac::{
///     fermi_dirac_integral_norm, fermi_dirac_integral_norm_checked,
/// };
///
/// assert_eq!(
///     fermi_dirac_integral_norm_checked(1.5, 2.0),
///     Ok(fermi_dirac_integral_norm(1.5, 2.0))
/// );
/// ```
pub fn fermi_dirac_integral_norm_checked(j: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(fermi_dirac_integral_checked(j, x)? / libm::tgamma(j + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::with_iteration_cap;

    #[test]
    fn iteration_caps_report_no_convergence() {
        let expected = Some(SpecialError::no_convergence(FAMILY, 1));
        with_iteration_cap(1, || {
            // x = -2 takes the series, x = 0 and x = 3 the quadrature.
            for x in [-2.0, 0.0, 3.0] {
                assert_eq!(
                    fermi_dirac_integral_checked(0.5, x).err(),
                    expected,
                    "x = {x}"
                );
                assert!(fermi_dirac_integral(0.5, x).is_nan(), "x = {x}");
                assert!(fermi_dirac_integral_norm(0.5, x).is_nan(), "x = {x}");
            }
        });
    }

    #[test]
    fn orders_below_the_domain_give_nan() {
        assert_eq!(
            fermi_dirac_integral_checked(-0.7, 1.0).map(f64::is_nan),
            Ok(true)
        );
    }
}
