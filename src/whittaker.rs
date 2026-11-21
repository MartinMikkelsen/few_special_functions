//! Whittaker M and W functions on the principal complex branch.
//!
//! Ported from `src/Whittaker.jl` of FewSpecialFunctions.jl
//! (<https://github.com/MartinMikkelsen/FewSpecialFunctions.jl>, MIT license),
//! commit b909e7b22696e942762f800ddad34a70d5b9d51d.
//! The Julia code follows DLMF 13.14.2–3, 13.14.33 and 13.19.3, and
//! I. J. Thompson and A. R. Barnett, J. Comput. Phys. 64, 490 (1986),
//! <https://doi.org/10.1016/0021-9991(86)90046-X>. The real-argument `W`
//! (integral representation, terminating series, κ-recurrence and connection
//! formula; DLMF sections 13.3 and 13.14–13.18) is written for this crate.
//!
//! # Iteration caps and tolerances
//!
//! | Method | Stops when | Cap |
//! |---|---|---|
//! | Kummer series for `M` | term below `4 ε max(1, \|sum\|)` | 100 000 terms |
//! | Radial Runge–Kutta propagation (`\|z\| > 10` off the positive axis) | fixed step 0.002 | 10⁷ steps (`\|z\|` up to about 20 000) |
//! | Quadrature of the integral representation of `W` | absolute error `1e-15` of an O(1) integrand | 20 000 subintervals |
//! | Recurrence in `κ` for `W` | after `⌈κ − µ − 1/2⌉` steps | 1 000 steps |
//!
//! Reaching a cap gives [`SpecialError::NoConvergence`] from the `*_checked`
//! functions and NaN from the others. The large-`z` asymptotic series are
//! summed until a term falls below `ε/16` of the sum; where the terms start to
//! grow, or 1 000 terms are not enough, another method is used instead. The
//! terminating series of `W` is summed exactly for orders up to 1 000.

use crate::SpecialError;
use crate::error::iteration_cap;
use crate::numerics::{gamma_complex, integrate, kummer_m, log_gamma_complex, reciprocal_gamma};
use num_complex::Complex64;

const FAMILY: &str = "whittaker";
/// Terms of the large-`z` asymptotic series tried before another method.
const ASYMPTOTIC_TERMS: usize = 1000;
/// Largest order of the terminating series of `W`.
const MAX_TERMINATING_ORDER: usize = 1000;
/// Cap on the Runge–Kutta steps of the radial propagation.
const MAX_PROPAGATION_STEPS: usize = 10_000_000;
/// Step of the radial propagation.
const PROPAGATION_STEP: f64 = 0.002;
/// Cap on the subintervals of the quadrature of the integral representation.
const MAX_QUADRATURE_INTERVALS: usize = 20_000;
/// Cap on the steps of the recurrence in `kappa`.
const MAX_RECURRENCE_STEPS: usize = 1000;

const NAN: Complex64 = Complex64::new(f64::NAN, f64::NAN);

fn finite(z: Complex64) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

/// Whether the complex functions cover these arguments: all finite, `z != 0`,
/// and for M no parameter pole at a negative integer `2 mu`.
fn in_complex_domain(kappa: Complex64, mu: Complex64, z: Complex64, is_m: bool) -> bool {
    if !(finite(kappa) && finite(mu) && finite(z)) || z == Complex64::new(0.0, 0.0) {
        return false;
    }
    let twice_mu = 2.0 * mu.re;
    !(is_m && mu.im == 0.0 && twice_mu <= -1.0 && twice_mu == twice_mu.round())
}

/// Whether the real functions cover these arguments: all finite and `z > 0`.
fn in_real_domain(kappa: f64, mu: f64, z: f64) -> bool {
    kappa.is_finite() && mu.is_finite() && z.is_finite() && z > 0.0
}

fn m_asymptotic(kappa: Complex64, mu: Complex64, z: Complex64) -> Option<Complex64> {
    let a = mu - kappa + 0.5;
    let b = 2.0 * mu + 1.0;
    let mut term = Complex64::new(1.0, 0.0);
    let mut sum = term;
    for n in 1..=ASYMPTOTIC_TERMS {
        let previous = (n - 1) as f64;
        let next = term * (b - a + previous) * (1.0 - a + previous) / (n as f64 * z);
        if next.norm() > term.norm() {
            return None;
        }
        sum += next;
        term = next;
        if term.norm() <= f64::EPSILON * sum.norm() / 16.0 {
            let scale =
                (z / 2.0 - kappa * z.ln() + log_gamma_complex(b) - log_gamma_complex(a)).exp();
            return Some(scale * sum);
        }
    }
    None
}

fn m_formula(kappa: Complex64, mu: Complex64, z: Complex64) -> Result<Complex64, SpecialError> {
    let a = mu - kappa + 0.5;
    let b = 2.0 * mu + 1.0;
    Ok((-z / 2.0 + (mu + 0.5) * z.ln()).exp() * kummer_m(a, b, z, FAMILY)?)
}

fn d_m_formula(kappa: Complex64, mu: Complex64, z: Complex64) -> Result<Complex64, SpecialError> {
    let value = m_formula(kappa, mu, z)?;
    let a = mu - kappa + 0.5;
    let shifted = a / (2.0 * mu + 1.0) * m_formula(kappa - 0.5, mu + 0.5, z)?;
    Ok((-0.5 + (mu + 0.5) / z) * value + shifted / z.sqrt())
}

fn propagate_whittaker(
    kappa: Complex64,
    mu: Complex64,
    z: Complex64,
    is_m: bool,
) -> Result<(Complex64, Complex64), SpecialError> {
    let radius = z.norm();
    let direction = z / radius;
    let mut position = 4.0;
    let max_steps = iteration_cap(MAX_PROPAGATION_STEPS);
    if (radius - position) / PROPAGATION_STEP > max_steps as f64 {
        return Err(SpecialError::no_convergence(FAMILY, max_steps));
    }
    let initial_z = direction * position;
    let (mut value, initial_derivative) = if is_m {
        (
            m_formula(kappa, mu, initial_z)?,
            d_m_formula(kappa, mu, initial_z)?,
        )
    } else {
        (
            w_formula(kappa, mu, initial_z)?,
            d_w_formula(kappa, mu, initial_z)?,
        )
    };
    let mut derivative = direction * initial_derivative;
    let acceleration = |radial_position: f64, amplitude: Complex64| {
        let argument = direction * radial_position;
        direction
            * direction
            * (0.25 - kappa / argument - (0.25 - mu * mu) / (argument * argument))
            * amplitude
    };

    while position < radius {
        let step = PROPAGATION_STEP.min(radius - position);
        let k1_value = derivative;
        let k1_derivative = acceleration(position, value);
        let k2_value = derivative + step * k1_derivative / 2.0;
        let k2_derivative = acceleration(position + step / 2.0, value + step * k1_value / 2.0);
        let k3_value = derivative + step * k2_derivative / 2.0;
        let k3_derivative = acceleration(position + step / 2.0, value + step * k2_value / 2.0);
        let k4_value = derivative + step * k3_derivative;
        let k4_derivative = acceleration(position + step, value + step * k3_value);
        value += step * (k1_value + 2.0 * k2_value + 2.0 * k3_value + k4_value) / 6.0;
        derivative += step
            * (k1_derivative + 2.0 * k2_derivative + 2.0 * k3_derivative + k4_derivative)
            / 6.0;
        position += step;
    }
    Ok((value, derivative / direction))
}

fn m_core(kappa: Complex64, mu: Complex64, z: Complex64) -> Result<Complex64, SpecialError> {
    // At a = 1/2 + mu - kappa a non-positive integer, M(a, b, z) is a
    // polynomial and the exponentially growing asymptotic form vanishes.
    let a = mu - kappa + 0.5;
    let terminating = a.im == 0.0 && a.re <= 0.0 && a.re == a.re.round();
    if !terminating
        && z.re > 0.0
        && z.norm() > 300.0
        && let Some(value) = m_asymptotic(kappa, mu, z)
    {
        return Ok(value);
    }
    if z.norm() > 10.0 && z.re <= z.norm() / 4.0 {
        return Ok(propagate_whittaker(kappa, mu, z, true)?.0);
    }
    m_formula(kappa, mu, z)
}

fn w_connection(kappa: Complex64, mu: Complex64, z: Complex64) -> Result<Complex64, SpecialError> {
    Ok(
        gamma_complex(-2.0 * mu) * reciprocal_gamma(0.5 - mu - kappa) * m_core(kappa, mu, z)?
            + gamma_complex(2.0 * mu) * reciprocal_gamma(0.5 + mu - kappa) * m_core(kappa, -mu, z)?,
    )
}

fn w_removable_limit(
    kappa: Complex64,
    mu: Complex64,
    z: Complex64,
) -> Result<Complex64, SpecialError> {
    let symmetric = |step: f64| -> Result<Complex64, SpecialError> {
        let h = Complex64::new(step, 0.0);
        Ok(0.5 * (w_connection(kappa, mu + h, z)? + w_connection(kappa, mu - h, z)?))
    };
    let s1 = symmetric(0.01)?;
    let s2 = symmetric(0.005)?;
    let s3 = symmetric(0.0025)?;
    let r1 = (4.0 * s2 - s1) / 3.0;
    let r2 = (4.0 * s3 - s2) / 3.0;
    Ok((16.0 * r2 - r1) / 15.0)
}

fn w_asymptotic(kappa: Complex64, mu: Complex64, z: Complex64) -> Option<(Complex64, Complex64)> {
    let mut term = Complex64::new(1.0, 0.0);
    let mut sum = term;
    let mut derivative_sum = Complex64::new(0.0, 0.0);
    for n in 1..=ASYMPTOTIC_TERMS {
        let previous = (n - 1) as f64;
        let next =
            -term * (mu - kappa + 0.5 + previous) / z * (-mu - kappa + 0.5 + previous) / n as f64;
        if next.norm() > term.norm() {
            return None;
        }
        sum += next;
        derivative_sum -= n as f64 * next / z;
        term = next;
        if term.norm() <= f64::EPSILON * sum.norm() / 16.0 {
            let scale = (-z / 2.0 + kappa * z.ln()).exp();
            return Some((
                scale * sum,
                scale * (derivative_sum + (kappa / z - 0.5) * sum),
            ));
        }
    }
    None
}

fn w_formula(kappa: Complex64, mu: Complex64, z: Complex64) -> Result<Complex64, SpecialError> {
    if kappa == Complex64::new(0.0, 0.0)
        && (mu == Complex64::new(0.5, 0.0) || mu == Complex64::new(-0.5, 0.0))
    {
        return Ok((-z / 2.0).exp());
    }
    if kappa == Complex64::new(1.0, 0.0)
        && (mu == Complex64::new(0.5, 0.0) || mu == Complex64::new(-0.5, 0.0))
    {
        return Ok(z * (-z / 2.0).exp());
    }
    if z.re > 0.0
        && let Some((value, _)) = w_asymptotic(kappa, mu, z)
    {
        return Ok(value);
    }
    if mu.im == 0.0 && 2.0 * mu.re == (2.0 * mu.re).round() {
        w_removable_limit(kappa, mu, z)
    } else {
        w_connection(kappa, mu, z)
    }
}

fn d_w_formula(kappa: Complex64, mu: Complex64, z: Complex64) -> Result<Complex64, SpecialError> {
    if kappa == Complex64::new(0.0, 0.0)
        && (mu == Complex64::new(0.5, 0.0) || mu == Complex64::new(-0.5, 0.0))
    {
        return Ok(-0.5 * (-z / 2.0).exp());
    }
    if kappa == Complex64::new(1.0, 0.0)
        && (mu == Complex64::new(0.5, 0.0) || mu == Complex64::new(-0.5, 0.0))
    {
        return Ok((1.0 - z / 2.0) * (-z / 2.0).exp());
    }
    if z.re > 0.0
        && let Some((_, derivative)) = w_asymptotic(kappa, mu, z)
    {
        return Ok(derivative);
    }
    let value = w_formula(kappa, mu, z)?;
    let a = mu - kappa + 0.5;
    let shifted = -a * w_formula(kappa - 0.5, mu + 0.5, z)?;
    Ok((-0.5 + (mu + 0.5) / z) * value + shifted / z.sqrt())
}

fn w_core(kappa: Complex64, mu: Complex64, z: Complex64) -> Result<Complex64, SpecialError> {
    if z.norm() > 10.0 && z.re <= 0.0 {
        Ok(propagate_whittaker(kappa, mu, z, false)?.0)
    } else {
        w_formula(kappa, mu, z)
    }
}

/// `exp(-z/2) z^kappa`, the large-`z` behavior of W, in logarithmic form.
fn w_log_prefactor(kappa: f64, z: f64) -> f64 {
    -z / 2.0 + kappa * z.ln()
}

/// W and W' when `kappa - mu - 1/2` is a non-negative integer `N`, with
/// `mu >= 0`.
///
/// The series `W = exp(-z/2) z^kappa Σ_k (1/2+mu-kappa)_k (1/2-mu-kappa)_k / (k! (-z)^k)`
/// then terminates after `k = N`, which is DLMF 13.18.17 written in powers of
/// `1/z`.
fn w_terminating(kappa: f64, mu: f64, z: f64, order: usize) -> (f64, f64) {
    let a = 0.5 + mu - kappa;
    let mut term = 1.0;
    let mut value = 0.0;
    let mut derivative = 0.0;
    for k in 0..=order {
        let kf = k as f64;
        value += term;
        // d/dz z^(kappa-k) exp(-z/2) = z^(kappa-k) exp(-z/2) ((kappa-k)/z - 1/2)
        derivative += term * ((kappa - kf) / z - 0.5);
        term *= -(a + kf) * (a - 2.0 * mu + kf) / ((kf + 1.0) * z);
    }
    let prefactor = w_log_prefactor(kappa, z).exp();
    (prefactor * value, prefactor * derivative)
}

/// The order `N` of [`w_terminating`] if `kappa - mu - 1/2` is a non-negative
/// integer no larger than [`MAX_TERMINATING_ORDER`].
fn terminating_order(kappa: f64, mu: f64) -> Option<usize> {
    let order = kappa - mu - 0.5;
    (order >= 0.0 && order == order.round() && order <= MAX_TERMINATING_ORDER as f64)
        .then_some(order as usize)
}

/// W for `a = 1/2 + mu - kappa > 0`, from
/// `W = exp(-z/2) z^kappa / Γ(a) ∫_0^∞ e^{-u} u^{a-1} (1 + u/z)^{mu+kappa-1/2} du`
/// (DLMF section 13.16).
///
/// The integral is evaluated without the prefactor, so the quadrature
/// tolerance is relative to an O(1) quantity; the substitution `u = v^(1/m)`,
/// `m = min(a, 1)`, removes the `u^{a-1}` endpoint singularity.
fn w_real_integral(kappa: f64, mu: f64, z: f64) -> Result<f64, SpecialError> {
    let a = mu - kappa + 0.5;
    debug_assert!(a > 0.0);
    let power = mu + kappa - 0.5;
    let log_normalization = -libm::lgamma(a);
    let upper = (a + power.abs() + 50.0).max(60.0);
    let m = a.min(1.0);
    let kernel = |v: f64| {
        if v == 0.0 {
            return if a == m {
                (log_normalization - m.ln()).exp()
            } else {
                0.0
            };
        }
        let u = v.powf(1.0 / m);
        let log_jacobian = -m.ln() + (1.0 - m) * u.ln();
        (log_normalization - u + (a - 1.0) * u.ln() + power * (u / z).ln_1p() + log_jacobian).exp()
    };
    let quadrature = |tolerance: f64| {
        integrate(
            kernel,
            0.0,
            upper.powf(m),
            tolerance,
            MAX_QUADRATURE_INTERVALS,
            FAMILY,
        )
    };
    let mut integral = quadrature(1e-15)?;
    // A small integral needs a tolerance relative to its own size.
    if integral.abs() < 1.0 {
        integral = quadrature(1e-15 * integral.abs())?;
    }
    Ok((w_log_prefactor(kappa, z) + integral.abs().ln()).exp() * integral.signum())
}

/// W from its expression in M (DLMF 13.14.33), accepted only when its two
/// terms cancel by less than a factor of 16.
///
/// The terms have Gamma-function poles at integer `2 mu`, and both grow like
/// `exp(z/2)` while W decays, so the check rejects those cases and large `z`.
fn w_real_connection(kappa: f64, mu: f64, z: f64) -> Result<Option<f64>, SpecialError> {
    let (kappa_c, mu_c, z_c) = (
        Complex64::from(kappa),
        Complex64::from(mu),
        Complex64::from(z),
    );
    let first = (gamma_complex(-2.0 * mu_c)
        * reciprocal_gamma(0.5 - mu_c - kappa_c)
        * m_core(kappa_c, mu_c, z_c)?)
    .re;
    let second = (gamma_complex(2.0 * mu_c)
        * reciprocal_gamma(0.5 + mu_c - kappa_c)
        * m_core(kappa_c, -mu_c, z_c)?)
    .re;
    let value = first + second;
    Ok((value.is_finite() && first.abs() + second.abs() <= 16.0 * value.abs()).then_some(value))
}

/// Largest `z` at which [`w_real_value`] tries [`w_real_connection`]; above it
/// the exponential growth of the M terms makes the cancellation check fail.
const CONNECTION_MAX_Z: f64 = 10.0;

/// W for real parameters, `mu >= 0` and `z > 0`.
fn w_real_value(kappa: f64, mu: f64, z: f64) -> Result<f64, SpecialError> {
    if let Some(order) = terminating_order(kappa, mu) {
        return Ok(w_terminating(kappa, mu, z, order).0);
    }
    if let Some((value, _)) = w_asymptotic(kappa.into(), mu.into(), z.into()) {
        return Ok(value.re);
    }
    if 2.0 * mu != (2.0 * mu).round()
        && z <= CONNECTION_MAX_Z
        && let Some(value) = w_real_connection(kappa, mu, z)?
    {
        return Ok(value);
    }
    let a = 0.5 + mu - kappa;
    if a > 0.0 {
        return w_real_integral(kappa, mu, z);
    }
    // a < 0: start where a lies in (0, 2] and recur upward in kappa with
    // W_{k+1} = (z - 2k) W_k - ((k - 1/2)² - mu²) W_{k-1} (DLMF 13.3.7 for
    // U(a, 1 + 2mu, z) at fixed mu). W is the minimal solution as a → +∞,
    // so decreasing a, which is increasing kappa, is the stable direction.
    let max_steps = iteration_cap(MAX_RECURRENCE_STEPS);
    if (-a).ceil() > max_steps as f64 {
        return Err(SpecialError::no_convergence(FAMILY, max_steps));
    }
    let steps = (-a).ceil() as usize;
    let mut k = kappa - steps as f64;
    let mut previous = w_real_integral(k - 1.0, mu, z)?;
    let mut current = w_real_integral(k, mu, z)?;
    for _ in 0..steps {
        let next = (z - 2.0 * k) * current - ((k - 0.5) * (k - 0.5) - mu * mu) * previous;
        (previous, current) = (current, next);
        k += 1.0;
    }
    Ok(current)
}

/// W and W' for real parameters and `z > 0`.
fn w_real(kappa: f64, mu: f64, z: f64) -> Result<(f64, f64), SpecialError> {
    // W is even in mu.
    let mu = mu.abs();
    if let Some(order) = terminating_order(kappa, mu) {
        return Ok(w_terminating(kappa, mu, z, order));
    }
    let value = w_real_value(kappa, mu, z)?;
    // DLMF section 13.15: W' = (-1/2 + (mu + 1/2)/z) W - a W_{kappa-1/2, mu+1/2} / sqrt(z).
    let a = 0.5 + mu - kappa;
    let shifted = w_real_value(kappa - 0.5, mu + 0.5, z)?;
    Ok((
        value,
        (-0.5 + (mu + 0.5) / z) * value - a * shifted / z.sqrt(),
    ))
}

/// Whittaker M for real parameters and positive real argument.
///
/// # Accuracy
///
/// Measured against mpmath for `kappa ∈ [−5, 5]`, `mu ∈ [−1.3, 3.5]` and
/// `z ∈ [0.01, 1000]` (see the crate README): relative error below 2e-13. The
/// largest errors are at `z = 1000`, where a relative change of `eps` in `z`
/// changes `M` by about `(z/2)·eps`.
///
/// Returns NaN unless the arguments are finite and `z > 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::whittaker_m;
///
/// // M_{−1,1/2}(z) = z e^{z/2} and M_{0,1/2}(z) = 2 sinh(z/2)
/// let z = 1.7_f64;
/// let v = whittaker_m(-1.0, 0.5, z);
/// assert!((v - z * (z / 2.0).exp()).abs() < 1e-14 * v);
/// assert!((whittaker_m(0.0, 0.5, z) - 2.0 * (z / 2.0).sinh()).abs() < 1e-14);
/// ```
pub fn whittaker_m(kappa: f64, mu: f64, z: f64) -> f64 {
    whittaker_m_checked(kappa, mu, z).unwrap_or(f64::NAN)
}

/// [`whittaker_m`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::{whittaker_m, whittaker_m_checked};
///
/// assert_eq!(whittaker_m_checked(0.3, 0.7, 2.0), Ok(whittaker_m(0.3, 0.7, 2.0)));
/// ```
pub fn whittaker_m_checked(kappa: f64, mu: f64, z: f64) -> Result<f64, SpecialError> {
    if !in_real_domain(kappa, mu, z) {
        return Ok(f64::NAN);
    }
    Ok(whittaker_m_complex_checked(kappa.into(), mu.into(), z.into())?.re)
}

/// Whittaker M on the principal complex branch.
///
/// Returns NaN unless the arguments are finite and `z != 0`, and at the
/// parameter poles `2 mu = -1, -2, …`.
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::whittaker_m_complex;
/// use num_complex::Complex64;
///
/// // M_{−1,1/2}(z) = z e^{z/2}
/// let z = Complex64::new(1.0, 2.0);
/// let v = whittaker_m_complex(Complex64::new(-1.0, 0.0), Complex64::new(0.5, 0.0), z);
/// let want = z * (z / 2.0).exp();
/// assert!((v - want).norm() < 1e-13 * want.norm());
/// ```
pub fn whittaker_m_complex(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    whittaker_m_complex_checked(kappa, mu, z).unwrap_or(NAN)
}

/// [`whittaker_m_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::{whittaker_m_complex, whittaker_m_complex_checked};
/// use num_complex::Complex64;
///
/// let (kappa, mu, z) = (Complex64::new(0.3, 0.1), Complex64::new(0.7, 0.0), Complex64::new(2.0, 1.0));
/// assert_eq!(whittaker_m_complex_checked(kappa, mu, z), Ok(whittaker_m_complex(kappa, mu, z)));
/// ```
pub fn whittaker_m_complex_checked(
    kappa: Complex64,
    mu: Complex64,
    z: Complex64,
) -> Result<Complex64, SpecialError> {
    if !in_complex_domain(kappa, mu, z, true) {
        return Ok(NAN);
    }
    m_core(kappa, mu, z)
}

/// Argument derivative of [`whittaker_m`].
///
/// Measured on the same grid as [`whittaker_m`]: relative error below 3e-13.
///
/// Returns NaN unless the arguments are finite and `z > 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::d_whittaker_m;
///
/// // d/dz [z e^{z/2}] = (1 + z/2) e^{z/2}
/// let z = 1.7_f64;
/// let want = (1.0 + z / 2.0) * (z / 2.0).exp();
/// assert!((d_whittaker_m(-1.0, 0.5, z) - want).abs() < 1e-13 * want);
/// ```
pub fn d_whittaker_m(kappa: f64, mu: f64, z: f64) -> f64 {
    d_whittaker_m_checked(kappa, mu, z).unwrap_or(f64::NAN)
}

/// [`d_whittaker_m`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::{d_whittaker_m, d_whittaker_m_checked};
///
/// assert_eq!(d_whittaker_m_checked(0.3, 0.7, 2.0), Ok(d_whittaker_m(0.3, 0.7, 2.0)));
/// ```
pub fn d_whittaker_m_checked(kappa: f64, mu: f64, z: f64) -> Result<f64, SpecialError> {
    if !in_real_domain(kappa, mu, z) {
        return Ok(f64::NAN);
    }
    Ok(d_whittaker_m_complex_checked(kappa.into(), mu.into(), z.into())?.re)
}

/// Argument derivative of [`whittaker_m_complex`].
///
/// Returns NaN unless the arguments are finite and `z != 0`, and at the
/// parameter poles `2 mu = -1, -2, …`.
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::d_whittaker_m_complex;
/// use num_complex::Complex64;
///
/// // d/dz [z e^{z/2}] = (1 + z/2) e^{z/2}
/// let z = Complex64::new(1.0, 2.0);
/// let v = d_whittaker_m_complex(Complex64::new(-1.0, 0.0), Complex64::new(0.5, 0.0), z);
/// let want = (1.0 + z / 2.0) * (z / 2.0).exp();
/// assert!((v - want).norm() < 1e-13 * want.norm());
/// ```
pub fn d_whittaker_m_complex(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    d_whittaker_m_complex_checked(kappa, mu, z).unwrap_or(NAN)
}

/// [`d_whittaker_m_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::{d_whittaker_m_complex, d_whittaker_m_complex_checked};
/// use num_complex::Complex64;
///
/// let (kappa, mu, z) = (Complex64::new(0.3, 0.1), Complex64::new(0.7, 0.0), Complex64::new(2.0, 1.0));
/// assert_eq!(d_whittaker_m_complex_checked(kappa, mu, z), Ok(d_whittaker_m_complex(kappa, mu, z)));
/// ```
pub fn d_whittaker_m_complex_checked(
    kappa: Complex64,
    mu: Complex64,
    z: Complex64,
) -> Result<Complex64, SpecialError> {
    if !in_complex_domain(kappa, mu, z, true) {
        return Ok(NAN);
    }
    if z.norm() > 10.0 && z.re <= z.norm() / 4.0 {
        return Ok(propagate_whittaker(kappa, mu, z, true)?.1);
    }
    let value = m_core(kappa, mu, z)?;
    let a = mu - kappa + 0.5;
    let shifted = a / (2.0 * mu + 1.0) * m_core(kappa - 0.5, mu + 0.5, z)?;
    Ok((-0.5 + (mu + 0.5) / z) * value + shifted / z.sqrt())
}

/// Whittaker W for real parameters and positive real argument.
///
/// W is even in `mu`. Where `kappa - |mu| - 1/2` is a non-negative integer, W
/// is a polynomial in `1/z` times `exp(-z/2) z^kappa` and is summed exactly.
/// Otherwise it comes from the large-`z` asymptotic series where that
/// converges; for `z <= 10`, from its expression in M when that does not
/// cancel; and otherwise from its integral representation when
/// `1/2 + |mu| - kappa > 0` and from the recurrence in `kappa` below that.
///
/// # Accuracy
///
/// Measured against mpmath for `kappa ∈ [−5, 5]`, `mu ∈ [−1.3, 3.5]` and
/// `z ∈ [0.01, 1000]` (see the crate README): relative error below 1e-13.
///
/// Returns NaN unless the arguments are finite and `z > 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::whittaker_w;
///
/// // W_{κ, ±(κ − 1/2)}(z) = e^{−z/2} z^κ
/// let (kappa, z) = (0.3_f64, 2.5_f64);
/// let want = (-z / 2.0).exp() * z.powf(kappa);
/// let v = whittaker_w(kappa, kappa - 0.5, z);
/// assert!((v - want).abs() < 1e-13 * want);
/// assert!((whittaker_w(kappa, 0.5 - kappa, z) - v).abs() < 1e-13 * want);
/// ```
pub fn whittaker_w(kappa: f64, mu: f64, z: f64) -> f64 {
    whittaker_w_checked(kappa, mu, z).unwrap_or(f64::NAN)
}

/// [`whittaker_w`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::{whittaker_w, whittaker_w_checked};
///
/// assert_eq!(whittaker_w_checked(0.3, 0.7, 2.0), Ok(whittaker_w(0.3, 0.7, 2.0)));
/// ```
pub fn whittaker_w_checked(kappa: f64, mu: f64, z: f64) -> Result<f64, SpecialError> {
    if !in_real_domain(kappa, mu, z) {
        return Ok(f64::NAN);
    }
    Ok(w_real(kappa, mu, z)?.0)
}

/// Whittaker W on the principal complex branch.
///
/// Returns NaN unless the arguments are finite and `z != 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::whittaker_w_complex;
/// use num_complex::Complex64;
///
/// // W_{κ, κ − 1/2}(z) = e^{−z/2} z^κ
/// let kappa = Complex64::new(0.3, 0.0);
/// let z = Complex64::new(1.0, 2.0);
/// let v = whittaker_w_complex(kappa, kappa - 0.5, z);
/// let want = (-z / 2.0).exp() * z.powc(kappa);
/// assert!((v - want).norm() < 1e-13 * want.norm());
/// ```
pub fn whittaker_w_complex(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    whittaker_w_complex_checked(kappa, mu, z).unwrap_or(NAN)
}

/// [`whittaker_w_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::{whittaker_w_complex, whittaker_w_complex_checked};
/// use num_complex::Complex64;
///
/// let (kappa, mu, z) = (Complex64::new(0.3, 0.1), Complex64::new(0.7, 0.0), Complex64::new(2.0, 1.0));
/// assert_eq!(whittaker_w_complex_checked(kappa, mu, z), Ok(whittaker_w_complex(kappa, mu, z)));
/// ```
pub fn whittaker_w_complex_checked(
    kappa: Complex64,
    mu: Complex64,
    z: Complex64,
) -> Result<Complex64, SpecialError> {
    if !in_complex_domain(kappa, mu, z, false) {
        return Ok(NAN);
    }
    w_core(kappa, mu, z)
}

/// Argument derivative of [`whittaker_w`].
///
/// Measured on the same grid as [`whittaker_w`]: relative error below 3e-13.
///
/// Returns NaN unless the arguments are finite and `z > 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::d_whittaker_w;
///
/// // d/dz [e^{−z/2} z^κ] = (κ/z − 1/2) e^{−z/2} z^κ
/// let (kappa, z) = (0.3_f64, 2.5_f64);
/// let want = (kappa / z - 0.5) * (-z / 2.0).exp() * z.powf(kappa);
/// assert!((d_whittaker_w(kappa, kappa - 0.5, z) - want).abs() < 1e-13 * want.abs());
/// ```
pub fn d_whittaker_w(kappa: f64, mu: f64, z: f64) -> f64 {
    d_whittaker_w_checked(kappa, mu, z).unwrap_or(f64::NAN)
}

/// [`d_whittaker_w`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::{d_whittaker_w, d_whittaker_w_checked};
///
/// assert_eq!(d_whittaker_w_checked(0.3, 0.7, 2.0), Ok(d_whittaker_w(0.3, 0.7, 2.0)));
/// ```
pub fn d_whittaker_w_checked(kappa: f64, mu: f64, z: f64) -> Result<f64, SpecialError> {
    if !in_real_domain(kappa, mu, z) {
        return Ok(f64::NAN);
    }
    Ok(w_real(kappa, mu, z)?.1)
}

/// Argument derivative of [`whittaker_w_complex`].
///
/// Returns NaN unless the arguments are finite and `z != 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::d_whittaker_w_complex;
/// use num_complex::Complex64;
///
/// // d/dz [e^{−z/2} z^κ] = (κ/z − 1/2) e^{−z/2} z^κ
/// let kappa = Complex64::new(0.3, 0.0);
/// let z = Complex64::new(1.0, 2.0);
/// let v = d_whittaker_w_complex(kappa, kappa - 0.5, z);
/// let want = (kappa / z - 0.5) * (-z / 2.0).exp() * z.powc(kappa);
/// assert!((v - want).norm() < 1e-13 * want.norm());
/// ```
pub fn d_whittaker_w_complex(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    d_whittaker_w_complex_checked(kappa, mu, z).unwrap_or(NAN)
}

/// [`d_whittaker_w_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::whittaker::{d_whittaker_w_complex, d_whittaker_w_complex_checked};
/// use num_complex::Complex64;
///
/// let (kappa, mu, z) = (Complex64::new(0.3, 0.1), Complex64::new(0.7, 0.0), Complex64::new(2.0, 1.0));
/// assert_eq!(d_whittaker_w_complex_checked(kappa, mu, z), Ok(d_whittaker_w_complex(kappa, mu, z)));
/// ```
pub fn d_whittaker_w_complex_checked(
    kappa: Complex64,
    mu: Complex64,
    z: Complex64,
) -> Result<Complex64, SpecialError> {
    if !in_complex_domain(kappa, mu, z, false) {
        return Ok(NAN);
    }
    if z.norm() > 10.0 && z.re <= 0.0 {
        Ok(propagate_whittaker(kappa, mu, z, false)?.1)
    } else {
        d_w_formula(kappa, mu, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::with_iteration_cap;

    #[test]
    fn iteration_caps_report_no_convergence() {
        let expected = Some(SpecialError::no_convergence(FAMILY, 1));
        with_iteration_cap(1, || {
            // The Kummer series; for W through the connection formula.
            assert_eq!(whittaker_m_checked(0.3, 0.7, 2.0).err(), expected);
            assert!(whittaker_m(0.3, 0.7, 2.0).is_nan());
            assert_eq!(whittaker_w_checked(0.3, 0.7, 2.0).err(), expected);
            assert!(d_whittaker_w(0.3, 0.7, 2.0).is_nan());
            // The quadrature of the integral representation (integer 2 mu).
            assert_eq!(whittaker_w_checked(0.3, 1.0, 2.0).err(), expected);
            // The recurrence in kappa.
            assert_eq!(whittaker_w_checked(4.2, 1.0, 2.0).err(), expected);
            // The radial propagation.
            let z = Complex64::new(0.0, 20.0);
            let (kappa, mu) = (Complex64::new(0.3, 0.0), Complex64::new(0.7, 0.0));
            assert_eq!(whittaker_m_complex_checked(kappa, mu, z).err(), expected);
            assert!(d_whittaker_w_complex(kappa, mu, -z).re.is_nan());
        });
    }

    #[test]
    fn arguments_outside_the_domain_give_nan() {
        assert!(whittaker_w(0.2, 0.3, -1.0).is_nan());
        assert!(whittaker_m(0.2, 0.3, 0.0).is_nan());
        assert!(whittaker_m(f64::NAN, 0.3, 1.0).is_nan());
        assert!(d_whittaker_m(0.2, -0.5, 1.0).is_nan());
        let zero = Complex64::new(0.0, 0.0);
        assert!(whittaker_w_complex(zero, zero, zero).re.is_nan());
    }
}
