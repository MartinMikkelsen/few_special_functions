//! Real parabolic-cylinder functions and their derivatives.
//!
//! Ported from `src/parabolic_cylinder.jl` of FewSpecialFunctions.jl
//! (<https://github.com/MartinMikkelsen/FewSpecialFunctions.jl>, MIT license),
//! commit b909e7b22696e942762f800ddad34a70d5b9d51d.
//! The scaled representations follow A. Gil, J. Segura and N. M. Temme, ACM
//! Trans. Math. Softw. 32, 70 (2006), <https://ir.cwi.nl/pub/14654/14654D.pdf>.
//! The reflection to negative `x` (DLMF 12.2.15 and 12.2.16) is written for
//! this crate.
//!
//! # Iteration caps and tolerances
//!
//! | Method | Stops when | Cap |
//! |---|---|---|
//! | Power series at small `\|x\|` | three successive terms below `ε` of the sums | 10 000 terms |
//! | Quadrature of the integral representation of `U` | absolute error `3e-12` to `3e-14` | 20 000 subintervals |
//! | Taylor-series integration of the differential equation | after a step count set by `a` and `x` | 2 000 000 steps |
//! | Recurrence in `a` for `U` at `a < −1/2` | after `⌊−1/2 − a⌋ + 1` steps | 100 000 steps |
//!
//! Reaching a cap gives [`SpecialError::NoConvergence`] from the `*_checked`
//! functions and NaN from the others. The large-`x` asymptotic series are
//! summed until a term falls below `ε/16` of the sum; where the terms start to
//! grow, or 1 000 terms are not enough, another method is used instead.

use crate::SpecialError;
use crate::error::iteration_cap;
use crate::numerics::{integrate, log_gamma_complex};
use num_complex::Complex64;
use std::f64::consts::{LN_2, PI};

const FAMILY: &str = "parabolic_cylinder";
/// Cap on the terms of the power series.
const MAX_SERIES_TERMS: usize = 10_000;
/// Terms of the large-`x` asymptotic series tried before another method.
const ASYMPTOTIC_TERMS: usize = 1000;
/// Cap on the subintervals of each quadrature.
const MAX_QUADRATURE_INTERVALS: usize = 20_000;
/// Cap on the Taylor steps of the integration of the differential equation.
const MAX_ODE_STEPS: usize = 2_000_000;
/// Cap on the steps of the recurrence in `a`.
const MAX_RECURRENCE_STEPS: usize = 100_000;

const NAN_PAIR: (f64, f64) = (f64::NAN, f64::NAN);

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    U,
    V,
    W,
}

fn reciprocal_gamma_scaled(argument: f64, log_scale: f64) -> f64 {
    if argument <= 0.0 && argument == argument.round() {
        return 0.0;
    }
    let sign = if argument > 0.0 || (PI * argument).sin() > 0.0 {
        1.0
    } else {
        -1.0
    };
    sign * (log_scale - libm::lgamma(argument)).exp()
}

fn initial(a: f64, kind: Kind, log_scale: f64) -> (f64, f64) {
    match kind {
        Kind::U => {
            let log_c = 0.5 * PI.ln() + (-a / 2.0 - 0.25) * LN_2 + log_scale;
            (
                reciprocal_gamma_scaled(a / 2.0 + 0.75, log_c),
                -reciprocal_gamma_scaled(a / 2.0 + 0.25, log_c + 0.5 * LN_2),
            )
        }
        Kind::V => {
            let angle_0 = PI * (a / 2.0 + 0.25);
            let angle_1 = PI * (a / 2.0 + 0.75);
            (
                angle_0.sin()
                    * reciprocal_gamma_scaled(0.75 - a / 2.0, (a / 2.0 + 0.25) * LN_2 + log_scale),
                angle_1.sin()
                    * reciprocal_gamma_scaled(0.25 - a / 2.0, (a / 2.0 + 0.75) * LN_2 + log_scale),
            )
        }
        Kind::W => {
            let upper = log_gamma_complex(Complex64::new(0.25, a / 2.0));
            let lower = log_gamma_complex(Complex64::new(0.75, a / 2.0));
            let half_ratio = 0.5 * (upper.re - lower.re);
            (
                (-0.75 * LN_2 + half_ratio + log_scale).exp(),
                -(-0.25 * LN_2 - half_ratio + log_scale).exp(),
            )
        }
    }
}

/// The power series in `x`; a sum that overflows is returned as it stands.
fn series(a: f64, x: f64, kind: Kind, log_scale: f64) -> Result<(f64, f64), SpecialError> {
    let (y0, y1) = initial(a, kind, log_scale);
    if x == 0.0 {
        return Ok((y0, y1));
    }
    let mut even_previous = 0.0;
    let mut even = y0;
    let mut odd_previous = 0.0;
    let mut odd = x * y1;
    let mut value = even + odd;
    let mut derivative = y1;
    let aa = a * x * x;
    let bb = if kind == Kind::W { -1.0 } else { 1.0 } * x.powi(4) / 4.0;
    let mut small = 0;

    let max_terms = iteration_cap(MAX_SERIES_TERMS);
    for k in 1..=max_terms {
        let kf = k as f64;
        (even_previous, even) = (
            even,
            (aa * even + bb * even_previous) / (2.0 * kf * (2.0 * kf - 1.0)),
        );
        (odd_previous, odd) = (
            odd,
            (aa * odd + bb * odd_previous) / ((2.0 * kf + 1.0) * 2.0 * kf),
        );
        let derivative_term = (2.0 * kf * even + (2.0 * kf + 1.0) * odd) / x;
        value += even + odd;
        derivative += derivative_term;
        let value_small = even.abs() + odd.abs() <= f64::EPSILON * value.abs();
        let derivative_small = (2.0 * kf * even.abs() + (2.0 * kf + 1.0) * odd.abs()) / x.abs()
            <= f64::EPSILON * derivative.abs();
        small = if value_small && derivative_small {
            small + 1
        } else {
            0
        };
        if small >= 3 || !value.is_finite() || !derivative.is_finite() {
            return Ok((value, derivative));
        }
    }
    Err(SpecialError::no_convergence(FAMILY, max_terms))
}

fn scale_value(log_scale: f64, value: f64) -> f64 {
    let scale = log_scale.exp();
    if scale < f64::MIN_POSITIVE || !scale.is_finite() {
        if value == 0.0 {
            value
        } else {
            value.signum() * (log_scale + value.abs().ln()).exp()
        }
    } else {
        scale * value
    }
}

fn log_scale(a: f64, x: f64) -> f64 {
    if a == 0.0 {
        return x * x / 4.0;
    }
    let q = x * x / 4.0 + a;
    if q <= 0.0 {
        return a * (a.abs().ln() - 1.0) / 2.0;
    }
    let root = q.sqrt();
    a * (x / 2.0 + root).ln() + x * root / 2.0 - a / 2.0
}

fn asymptotic_scale(a: f64, x: f64, kind: Kind) -> f64 {
    let q = a / ((x * x / 4.0 + a).sqrt() + x / 2.0);
    let correction = a * (q / x).ln_1p() - q * q / 2.0;
    (if kind == Kind::U {
        correction
    } else {
        -correction
    }) - x.ln() / 2.0
}

fn u_asymptotic(a: f64, x: f64, scaled: bool) -> Option<(f64, f64)> {
    let mut ratio = 1.0;
    let mut sum = 1.0;
    let mut derivative_sum = 0.0;
    for k in 1..=ASYMPTOTIC_TERMS {
        let kf = k as f64;
        let next = -ratio * (a + 2.0 * kf - 1.5) * (a + 2.0 * kf - 0.5) / (2.0 * kf * x * x);
        if next.abs() > ratio.abs() {
            return None;
        }
        sum += next;
        derivative_sum -= 2.0 * kf * next / x;
        ratio = next;
        if ratio.abs() <= f64::EPSILON * sum.abs() / 16.0 {
            let exponent = if scaled {
                asymptotic_scale(a, x, Kind::U)
            } else {
                -x * x / 4.0 - (a + 0.5) * x.ln()
            };
            return Some((
                scale_value(exponent, sum),
                scale_value(exponent, derivative_sum - (x / 2.0 + (a + 0.5) / x) * sum),
            ));
        }
    }
    None
}

fn v_asymptotic_scaled(a: f64, x: f64) -> Option<(f64, f64)> {
    let mut ratio = 1.0;
    let mut sum = 1.0;
    let mut derivative_sum = 0.0;
    for k in 1..=ASYMPTOTIC_TERMS {
        let kf = k as f64;
        let next = ratio * (2.0 * kf - 1.5 - a) * (2.0 * kf - 0.5 - a) / (2.0 * kf * x * x);
        if next.abs() > ratio.abs() {
            return None;
        }
        sum += next;
        derivative_sum -= 2.0 * kf * next / x;
        ratio = next;
        if ratio.abs() <= f64::EPSILON * sum.abs() / 16.0 {
            let exponent = asymptotic_scale(a, x, Kind::V) + 0.5 * (2.0 / PI).ln();
            return Some((
                scale_value(exponent, sum),
                scale_value(exponent, derivative_sum + (x / 2.0 + (a - 0.5) / x) * sum),
            ));
        }
    }
    None
}

fn u_integral_scaled(a: f64, x: f64) -> Result<(f64, f64), SpecialError> {
    let alpha = a + 0.5;
    debug_assert!(alpha > 0.0);
    let normalization = log_scale(a, x) - x * x / 4.0 - libm::lgamma(alpha);
    if alpha > 1.0 {
        let saddle = ((x * x + 4.0 * (alpha - 1.0)).sqrt() - x) / 2.0;
        let curvature = 1.0 + (alpha - 1.0) / (saddle * saddle);
        let upper = saddle + 12.0_f64.max(12.0 / curvature.sqrt());
        let kernel = |t: f64| {
            if t == 0.0 {
                0.0
            } else {
                (normalization - x * t - t * t / 2.0 + (alpha - 1.0) * t.ln()).exp()
            }
        };
        let value = integrate(kernel, 0.0, upper, 3e-12, MAX_QUADRATURE_INTERVALS, FAMILY)?;
        let derivative = integrate(
            |t| -(x / 2.0 + t) * kernel(t),
            0.0,
            upper,
            1e-10,
            MAX_QUADRATURE_INTERVALS,
            FAMILY,
        )?;
        return Ok((value, derivative));
    }
    let m = alpha.min(1.0);
    let saddle = ((x * x + 4.0 * (a - 0.5).max(0.0)).sqrt() - x) / 2.0;
    let t_scale = saddle.max(1.0);
    let evaluate = |v: f64, derivative: bool| {
        if v == 0.0 {
            if alpha == m {
                let base = normalization + alpha * t_scale.ln() - m.ln();
                return if derivative {
                    -x / 2.0 * base.exp()
                } else {
                    base.exp()
                };
            }
            return 0.0;
        }
        if v == 1.0 {
            return 0.0;
        }
        let s = v.powf(1.0 / m);
        let t = t_scale * s / (1.0 - s);
        let log_jacobian = t_scale.ln() - m.ln() + (1.0 - m) * s.ln() - 2.0 * (-s).ln_1p();
        let base =
            (normalization - x * t - t * t / 2.0 + (alpha - 1.0) * t.ln() + log_jacobian).exp();
        if derivative {
            -(x / 2.0 + t) * base
        } else {
            base
        }
    };
    let quadrature = |derivative: bool| {
        integrate(
            |v| evaluate(v, derivative),
            0.0,
            1.0,
            3e-14,
            MAX_QUADRATURE_INTERVALS,
            FAMILY,
        )
    };
    Ok((quadrature(false)?, quadrature(true)?))
}

/// `U` scaled back by `exp(-L(a, x))` from [`u_integral_scaled`].
fn u_integral(a: f64, x: f64) -> Result<(f64, f64), SpecialError> {
    let exponent = log_scale(a, x);
    let (value, derivative) = u_integral_scaled(a, x)?;
    Ok((
        scale_value(-exponent, value),
        scale_value(-exponent, derivative),
    ))
}

/// `U` and `U'` at any real `x`.
///
/// For `x < 0` it uses DLMF 12.2.15,
/// `U(a, -x) = -sin(πa) U(a, x) + π/Γ(1/2 + a) V(a, x)`: the power series at
/// negative `x` cancels by up to `exp(x²/2)`.
fn u_pair(a: f64, x: f64) -> Result<(f64, f64), SpecialError> {
    if !(a.is_finite() && x.is_finite()) {
        return Ok(NAN_PAIR);
    }
    if x < 0.0 {
        let (u, du) = u_pair(a, -x)?;
        let (v, dv) = ode_pair(a, -x, Kind::V, 0.0)?;
        let sine = sin_pi(a);
        // π/Γ(1/2 + a) applied to V in logarithmic form: for large a the
        // reciprocal underflows long before the product does.
        let times_v = |value: f64| {
            if value == 0.0 {
                0.0
            } else {
                value.signum() * reciprocal_gamma_scaled(0.5 + a, PI.ln() + value.abs().ln())
            }
        };
        return Ok((-sine * u + times_v(v), sine * du - times_v(dv)));
    }
    if x > 1.0
        && let Some(result) = u_asymptotic(a, x, false)
    {
        return Ok(result);
    }
    if x.abs() > 1.0 {
        if a > -0.5 {
            return u_integral(a, x);
        }
        let max_steps = iteration_cap(MAX_RECURRENCE_STEPS);
        if (-0.5 - a).floor() >= max_steps as f64 {
            return Err(SpecialError::no_convergence(FAMILY, max_steps));
        }
        let steps = ((-0.5 - a).floor() as usize) + 1;
        let upper_order = a + steps as f64;
        let (mut current_value, mut current_derivative) = u_integral(upper_order, x)?;
        let (mut upper_value, mut upper_derivative) = u_integral(upper_order + 1.0, x)?;
        let mut order = upper_order;
        for _ in 0..steps {
            let lower_value = x * current_value + (order + 0.5) * upper_value;
            let lower_derivative =
                current_value + x * current_derivative + (order + 0.5) * upper_derivative;
            upper_value = current_value;
            upper_derivative = current_derivative;
            current_value = lower_value;
            current_derivative = lower_derivative;
            order -= 1.0;
        }
        return Ok((current_value, current_derivative));
    }
    if a > 0.5 && x > 0.0 {
        return u_integral(a, x);
    }
    series(a, x, Kind::U, 0.0)
}

fn ode_advance(
    a: f64,
    position: f64,
    value: f64,
    derivative: f64,
    step: f64,
    kind: Kind,
) -> (f64, f64) {
    const TERMS: usize = 24;
    let equation_sign = if kind == Kind::W { -1.0 } else { 1.0 };
    let q = [
        a + equation_sign * position * position / 4.0,
        equation_sign * position / 2.0,
        equation_sign / 4.0,
    ];
    let mut coefficients = [0.0; TERMS];
    coefficients[0] = value;
    coefficients[1] = derivative;
    for n in 0..TERMS - 2 {
        let mut right_hand_side = q[0] * coefficients[n];
        if n >= 1 {
            right_hand_side += q[1] * coefficients[n - 1];
        }
        if n >= 2 {
            right_hand_side += q[2] * coefficients[n - 2];
        }
        coefficients[n + 2] = right_hand_side / ((n + 1) * (n + 2)) as f64;
    }

    let next_value = coefficients
        .iter()
        .rev()
        .fold(0.0, |sum, coefficient| sum * step + coefficient);
    let next_derivative = (1..TERMS)
        .rev()
        .fold(0.0, |sum, n| sum * step + n as f64 * coefficients[n]);
    (next_value, next_derivative)
}

fn ode_pair(
    a: f64,
    x: f64,
    kind: Kind,
    initial_log_scale: f64,
) -> Result<(f64, f64), SpecialError> {
    let (mut value, mut derivative) = initial(a, kind, initial_log_scale);
    let mut position = 0.0;
    let mut steps = 0_usize;
    let max_steps = iteration_cap(MAX_ODE_STEPS);
    while position != x {
        if steps == max_steps {
            return Err(SpecialError::no_convergence(FAMILY, max_steps));
        }
        let remaining = x - position;
        let local_scale = 1.0 + a.abs().sqrt() + position.abs() / 2.0;
        let step = remaining.signum() * (0.5 / local_scale).min(0.1).min(remaining.abs());
        (value, derivative) = ode_advance(a, position, value, derivative, step, kind);
        position += step;
        steps += 1;
    }
    Ok((value, derivative))
}

fn w_asymptotic(a: f64, x: f64) -> Option<(f64, f64)> {
    let t = x.abs();
    let mut ratio = Complex64::new(1.0, 0.0);
    let mut sum = ratio;
    let mut derivative_sum = Complex64::new(0.0, 0.0);
    for k in 1..=ASYMPTOTIC_TERMS {
        let kf = k as f64;
        let next = -Complex64::i()
            * ratio
            * Complex64::new(2.0 * kf - 1.5, a)
            * Complex64::new(2.0 * kf - 0.5, a)
            / (2.0 * kf * t * t);
        if next.norm() > ratio.norm() {
            return None;
        }
        sum += next;
        derivative_sum -= 2.0 * kf * next / t;
        ratio = next;
        if ratio.norm() <= f64::EPSILON * sum.norm() / 16.0 {
            let log_inverse_k = if a >= 0.0 {
                PI * a + (1.0 + (1.0 + (-2.0 * PI * a).exp()).sqrt()).ln()
            } else {
                (PI * a).exp().asinh()
            };
            let phase = t * t / 4.0 - a * t.ln()
                + PI / 4.0
                + 0.5 * log_gamma_complex(Complex64::new(0.5, a)).im;
            let exponent = 0.5
                * (2.0_f64.ln() - t.ln()
                    + if x > 0.0 {
                        -log_inverse_k
                    } else {
                        log_inverse_k
                    });
            let oscillation = Complex64::from_polar(1.0, phase);
            let derivative =
                derivative_sum + Complex64::new(-1.0 / (2.0 * t), t / 2.0 - a / t) * sum;
            let (value_component, derivative_component) = if x > 0.0 {
                ((oscillation * sum).re, (oscillation * derivative).re)
            } else {
                ((oscillation * sum).im, (oscillation * derivative).im)
            };
            return Some((
                scale_value(exponent, value_component),
                x.signum() * scale_value(exponent, derivative_component),
            ));
        }
    }
    None
}

fn w_pair(a: f64, x: f64) -> Result<(f64, f64), SpecialError> {
    if !(a.is_finite() && x.is_finite()) {
        return Ok(NAN_PAIR);
    }
    if x.abs() > 1.0 && x.abs() <= 10.0 {
        return ode_pair(a, x, Kind::W, 0.0);
    }
    if x.abs() > 1.0
        && let Some(result) = w_asymptotic(a, x)
    {
        return Ok(result);
    }
    series(a, x, Kind::W, 0.0)
}

/// `sin(πa)`, exactly zero at integers.
fn sin_pi(a: f64) -> f64 {
    // r = a mod 2 in [-1, 1]; both reductions below are exact.
    let r = a - 2.0 * (a / 2.0).round();
    let r = if r.abs() > 0.5 { r.signum() - r } else { r };
    (PI * r).sin()
}

/// `cos(πa)`, exactly zero at half-integers.
fn cos_pi(a: f64) -> f64 {
    let r = (a - 2.0 * (a / 2.0).round()).abs();
    sin_pi(0.5 - r)
}

/// `V` and `V'` at any real `x`.
///
/// For `x >= 0`, `V` is the dominant solution of the equation, so it is
/// integrated outward from 0. For `x < 0` it uses DLMF 12.2.16,
/// `V(a, -x) = cos(πa)/Γ(1/2 - a) U(a, x) + sin(πa) V(a, x)`: integrating
/// toward negative `x` is unstable whenever `V` is recessive there, as at
/// integer `a`.
fn v_pair(a: f64, x: f64) -> Result<(f64, f64), SpecialError> {
    if !(a.is_finite() && x.is_finite()) {
        return Ok(NAN_PAIR);
    }
    if x >= 0.0 {
        return ode_pair(a, x, Kind::V, 0.0);
    }
    let (u, du) = u_pair(a, -x)?;
    let (v, dv) = ode_pair(a, -x, Kind::V, 0.0)?;
    let sine = sin_pi(a);
    // cos(πa)/Γ(1/2 - a) = cos²(πa) Γ(1/2 + a)/π, the form used for a > 0
    // so that Γ is never evaluated near its poles; applied to U in
    // logarithmic form because Γ(1/2 + a) overflows long before Γ(1/2 + a) U.
    let times_u = |value: f64| {
        if a <= 0.0 {
            cos_pi(a) * reciprocal_gamma_scaled(0.5 - a, 0.0) * value
        } else {
            let cosine = cos_pi(a);
            if cosine == 0.0 || value == 0.0 {
                return 0.0;
            }
            let log_magnitude =
                2.0 * cosine.abs().ln() + libm::lgamma(0.5 + a) - PI.ln() + value.abs().ln();
            value.signum() * log_magnitude.exp()
        }
    };
    Ok((times_u(u) + sine * v, -(times_u(du) + sine * dv)))
}

/// Parabolic-cylinder function `U(a, x)`.
///
/// Returns NaN unless `a` and `x` are finite; the same holds for every
/// function of this module.
///
/// # Accuracy
///
/// Measured against mpmath for `a ∈ [−10, 20]` and `x ∈ [−20, 10]` (see the
/// crate README): relative error below 2e-13.
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::u;
///
/// // U(−1/2, x) = e^{−x²/4} and U(−3/2, x) = x e^{−x²/4}
/// let x = 1.3_f64;
/// let gauss = (-x * x / 4.0).exp();
/// assert!((u(-0.5, x) - gauss).abs() < 1e-14);
/// assert!((u(-1.5, x) - x * gauss).abs() < 1e-14);
/// ```
pub fn u(a: f64, x: f64) -> f64 {
    u_checked(a, x).unwrap_or(f64::NAN)
}

/// [`u`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{u, u_checked};
///
/// assert_eq!(u_checked(0.7, 1.3), Ok(u(0.7, 1.3)));
/// ```
pub fn u_checked(a: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(u_pair(a, x)?.0)
}

/// Argument derivative of [`u`].
///
/// Has the same accuracy as [`u`].
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::du;
///
/// // d/dx e^{−x²/4} = −(x/2) e^{−x²/4}
/// let x = 1.3_f64;
/// assert!((du(-0.5, x) + x / 2.0 * (-x * x / 4.0).exp()).abs() < 1e-14);
/// ```
pub fn du(a: f64, x: f64) -> f64 {
    du_checked(a, x).unwrap_or(f64::NAN)
}

/// [`du`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{du, du_checked};
///
/// assert_eq!(du_checked(0.7, 1.3), Ok(du(0.7, 1.3)));
/// ```
pub fn du_checked(a: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(u_pair(a, x)?.1)
}

/// Parabolic-cylinder function `V(a, x)`.
///
/// # Accuracy
///
/// Measured against mpmath for `a ∈ [−10, 20]` and `x ∈ [−20, 10]` (see the
/// crate README): relative error below 2e-13.
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::v;
///
/// // mpmath: V(0.7, 1.3) = 1.2250775797975974
/// assert!((v(0.7, 1.3) - 1.225_077_579_797_597_4).abs() < 1e-13);
/// ```
pub fn v(a: f64, x: f64) -> f64 {
    v_checked(a, x).unwrap_or(f64::NAN)
}

/// [`v`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{v, v_checked};
///
/// assert_eq!(v_checked(0.7, 1.3), Ok(v(0.7, 1.3)));
/// ```
pub fn v_checked(a: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(v_pair(a, x)?.0)
}

/// Argument derivative of [`v`].
///
/// Has the same accuracy as [`v`].
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{du, dv, u, v};
/// use std::f64::consts::PI;
///
/// // Wronskian U V′ − U′ V = sqrt(2/π) (DLMF 12.2.20)
/// let (a, x) = (0.7, 1.3);
/// let wronskian = u(a, x) * dv(a, x) - du(a, x) * v(a, x);
/// assert!((wronskian - (2.0 / PI).sqrt()).abs() < 1e-13);
/// ```
pub fn dv(a: f64, x: f64) -> f64 {
    dv_checked(a, x).unwrap_or(f64::NAN)
}

/// [`dv`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{dv, dv_checked};
///
/// assert_eq!(dv_checked(0.7, 1.3), Ok(dv(0.7, 1.3)));
/// ```
pub fn dv_checked(a: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(v_pair(a, x)?.1)
}

/// Oscillatory parabolic-cylinder function `W(a, x)`.
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::w;
///
/// // mpmath: W(0.5, 1.5) = 0.33771491253541248
/// assert!((w(0.5, 1.5) - 0.337_714_912_535_412_48).abs() < 1e-10);
/// ```
pub fn w(a: f64, x: f64) -> f64 {
    w_checked(a, x).unwrap_or(f64::NAN)
}

/// [`w`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{w, w_checked};
///
/// assert_eq!(w_checked(0.7, 1.3), Ok(w(0.7, 1.3)));
/// ```
pub fn w_checked(a: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(w_pair(a, x)?.0)
}

/// Argument derivative of [`w`].
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{dw, w};
///
/// // mpmath: W′(0.5, 1.5) = −0.26726353506748859
/// let (a, x) = (0.5, 1.5);
/// assert!((dw(a, x) + 0.267_263_535_067_488_59).abs() < 1e-10);
///
/// // Wronskian of W(a, x) and W(a, −x) is 1 (DLMF 12.14.5)
/// let wronskian = -w(a, x) * dw(a, -x) - dw(a, x) * w(a, -x);
/// assert!((wronskian - 1.0).abs() < 1e-10);
/// ```
pub fn dw(a: f64, x: f64) -> f64 {
    dw_checked(a, x).unwrap_or(f64::NAN)
}

/// [`dw`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{dw, dw_checked};
///
/// assert_eq!(dw_checked(0.7, 1.3), Ok(dw(0.7, 1.3)));
/// ```
pub fn dw_checked(a: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(w_pair(a, x)?.1)
}

/// Scaled `U(a, x)`: `exp(L(a, x)) U(a, x)` for `x >= 0`; NaN for `x < 0`.
///
/// With `q = x²/4 + a`, the exponent is
/// `L = a ln(x/2 + √q) + x √q / 2 − a/2` for `q > 0`, `L = a (ln|a| − 1)/2`
/// for `q <= 0`, and `L = x²/4` for `a = 0` (A. Gil, J. Segura and N. M. Temme,
/// ACM Trans. Math. Softw. 32, 70 (2006), eqs. (7) and (11)–(13)). It removes
/// the growth and decay in both `a` and `x`, so the scaled value stays in
/// range where `U` itself underflows.
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{u, u_scaled};
///
/// let (a, x) = (2.0_f64, 3.0_f64);
/// let root = (x * x / 4.0 + a).sqrt();
/// let exponent = a * (x / 2.0 + root).ln() + x * root / 2.0 - a / 2.0;
/// assert!((u_scaled(a, x) - exponent.exp() * u(a, x)).abs() < 1e-13 * u_scaled(a, x));
///
/// // U(20, 60) underflows, the scaled value does not.
/// assert_eq!(u(20.0, 60.0), 0.0);
/// assert!(u_scaled(20.0, 60.0) > 0.0);
/// ```
pub fn u_scaled(a: f64, x: f64) -> f64 {
    u_scaled_checked(a, x).unwrap_or(f64::NAN)
}

/// [`u_scaled`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{u_scaled, u_scaled_checked};
///
/// assert_eq!(u_scaled_checked(2.0, 3.0), Ok(u_scaled(2.0, 3.0)));
/// ```
pub fn u_scaled_checked(a: f64, x: f64) -> Result<f64, SpecialError> {
    if !(a.is_finite() && x.is_finite() && x >= 0.0) {
        return Ok(f64::NAN);
    }
    let exponent = log_scale(a, x);
    if x * x / 4.0 + a <= 0.0 {
        return Ok(ode_pair(a, x, Kind::U, exponent)?.0);
    }
    if x > 1.0
        && x * x / 4.0 + a > 0.0
        && let Some(result) = u_asymptotic(a, x, true)
    {
        return Ok(result.0);
    }
    if a > 0.5 || (a > -0.5 && x > 1.0) {
        return Ok(u_integral_scaled(a, x)?.0);
    }
    if x > 1.0 {
        return Ok(scale_value(exponent, u_pair(a, x)?.0));
    }
    Ok(series(a, x, Kind::U, exponent)?.0)
}

/// Scaled `V(a, x)`: `exp(−L(a, x)) V(a, x)` for `x >= 0`, with the exponent
/// `L` of [`u_scaled`]; NaN for `x < 0`.
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{v, v_scaled};
///
/// let (a, x) = (2.0_f64, 3.0_f64);
/// let root = (x * x / 4.0 + a).sqrt();
/// let exponent = a * (x / 2.0 + root).ln() + x * root / 2.0 - a / 2.0;
/// assert!((v_scaled(a, x) - (-exponent).exp() * v(a, x)).abs() < 1e-13 * v_scaled(a, x));
/// ```
pub fn v_scaled(a: f64, x: f64) -> f64 {
    v_scaled_checked(a, x).unwrap_or(f64::NAN)
}

/// [`v_scaled`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if a method reaches its cap (see the module
/// documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{v_scaled, v_scaled_checked};
///
/// assert_eq!(v_scaled_checked(2.0, 3.0), Ok(v_scaled(2.0, 3.0)));
/// ```
pub fn v_scaled_checked(a: f64, x: f64) -> Result<f64, SpecialError> {
    if !(a.is_finite() && x.is_finite() && x >= 0.0) {
        return Ok(f64::NAN);
    }
    if x > 1.0
        && x * x / 4.0 + a > 0.0
        && let Some(result) = v_asymptotic_scaled(a, x)
    {
        return Ok(result.0);
    }
    Ok(ode_pair(a, x, Kind::V, -log_scale(a, x))?.0)
}

/// Standard parabolic-cylinder function `D_nu(x) = U(-nu - 1/2, x)`.
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::parabolic_cylinder_d;
///
/// // D_1(x) = x e^{−x²/4} and D_2(x) = (x² − 1) e^{−x²/4}
/// let x = 1.3_f64;
/// let gauss = (-x * x / 4.0).exp();
/// assert!((parabolic_cylinder_d(1.0, x) - x * gauss).abs() < 1e-14);
/// assert!((parabolic_cylinder_d(2.0, x) - (x * x - 1.0) * gauss).abs() < 1e-14);
/// ```
pub fn parabolic_cylinder_d(nu: f64, x: f64) -> f64 {
    parabolic_cylinder_d_checked(nu, x).unwrap_or(f64::NAN)
}

/// [`parabolic_cylinder_d`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// As for [`u_checked`].
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{parabolic_cylinder_d, parabolic_cylinder_d_checked};
///
/// assert_eq!(parabolic_cylinder_d_checked(1.5, 2.0), Ok(parabolic_cylinder_d(1.5, 2.0)));
/// ```
pub fn parabolic_cylinder_d_checked(nu: f64, x: f64) -> Result<f64, SpecialError> {
    u_checked(-nu - 0.5, x)
}

/// Argument derivative of [`parabolic_cylinder_d`].
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::d_parabolic_cylinder_d;
///
/// // d/dx [x e^{−x²/4}] = (1 − x²/2) e^{−x²/4}
/// let x = 1.3_f64;
/// let want = (1.0 - x * x / 2.0) * (-x * x / 4.0).exp();
/// assert!((d_parabolic_cylinder_d(1.0, x) - want).abs() < 1e-14);
/// ```
pub fn d_parabolic_cylinder_d(nu: f64, x: f64) -> f64 {
    d_parabolic_cylinder_d_checked(nu, x).unwrap_or(f64::NAN)
}

/// [`d_parabolic_cylinder_d`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// As for [`du_checked`].
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{d_parabolic_cylinder_d, d_parabolic_cylinder_d_checked};
///
/// assert_eq!(d_parabolic_cylinder_d_checked(1.5, 2.0), Ok(d_parabolic_cylinder_d(1.5, 2.0)));
/// ```
pub fn d_parabolic_cylinder_d_checked(nu: f64, x: f64) -> Result<f64, SpecialError> {
    du_checked(-nu - 0.5, x)
}

/// Scaled standard parabolic-cylinder function for `x >= 0`:
/// [`u_scaled`]`(−nu − 1/2, x)`.
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{parabolic_cylinder_d_scaled, u_scaled};
///
/// assert_eq!(parabolic_cylinder_d_scaled(1.5, 2.0), u_scaled(-2.0, 2.0));
/// ```
pub fn parabolic_cylinder_d_scaled(nu: f64, x: f64) -> f64 {
    parabolic_cylinder_d_scaled_checked(nu, x).unwrap_or(f64::NAN)
}

/// [`parabolic_cylinder_d_scaled`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// As for [`u_scaled_checked`].
///
/// # Examples
///
/// ```
/// use few_special_functions::parabolic_cylinder::{parabolic_cylinder_d_scaled, parabolic_cylinder_d_scaled_checked};
///
/// assert_eq!(parabolic_cylinder_d_scaled_checked(1.5, 2.0), Ok(parabolic_cylinder_d_scaled(1.5, 2.0)));
/// ```
pub fn parabolic_cylinder_d_scaled_checked(nu: f64, x: f64) -> Result<f64, SpecialError> {
    u_scaled_checked(-nu - 0.5, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::with_iteration_cap;

    #[test]
    fn iteration_caps_report_no_convergence() {
        let expected = Some(SpecialError::no_convergence(FAMILY, 1));
        with_iteration_cap(1, || {
            // The power series, the quadrature, the recurrence in a and the
            // integration of the differential equation.
            for (a, x) in [(0.3, 0.5), (2.0, 0.5), (-3.2, 1.5)] {
                assert_eq!(u_checked(a, x).err(), expected, "a = {a}, x = {x}");
                assert!(du(a, x).is_nan(), "a = {a}, x = {x}");
            }
            assert_eq!(v_checked(0.7, 1.3).err(), expected);
            assert_eq!(w_checked(0.5, 1.5).err(), expected);
            assert!(dw(0.5, 1.5).is_nan());
            assert_eq!(u_scaled_checked(-3.0, 1.0).err(), expected);
            assert!(v_scaled(0.7, 1.3).is_nan());
        });
    }

    #[test]
    fn arguments_outside_the_domain_give_nan() {
        assert!(u_scaled(0.0, -1.0).is_nan());
        assert!(v_scaled(0.0, -1.0).is_nan());
        assert!(u(f64::NAN, 1.0).is_nan());
        assert!(dv(0.5, f64::INFINITY).is_nan());
        assert!(w(f64::NEG_INFINITY, 1.0).is_nan());
        assert!(parabolic_cylinder_d(f64::NAN, 1.0).is_nan());
    }
}
