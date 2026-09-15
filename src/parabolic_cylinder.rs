//! Real parabolic-cylinder functions and their derivatives.

use crate::numerics::{integrate, log_gamma_complex};
use num_complex::Complex64;
use std::f64::consts::{LN_2, PI};

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    U,
    V,
    W,
}

fn check_finite(a: f64, x: f64) {
    assert!(
        a.is_finite() && x.is_finite(),
        "cylinder parameters must be finite"
    );
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

fn series(a: f64, x: f64, kind: Kind, log_scale: f64) -> (f64, f64) {
    let (y0, y1) = initial(a, kind, log_scale);
    if x == 0.0 {
        return (y0, y1);
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

    for k in 1..=10_000 {
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
        if small >= 3 {
            return (value, derivative);
        }
        if !value.is_finite() || !derivative.is_finite() {
            break;
        }
    }
    panic!("parabolic-cylinder series did not converge")
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
    for k in 1..=1000 {
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
    for k in 1..=1000 {
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

fn u_integral_scaled(a: f64, x: f64) -> (f64, f64) {
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
        let value = integrate(kernel, 0.0, upper, 3e-12, 20_000)
            .unwrap_or_else(|message| panic!("parabolic-cylinder U quadrature failed: {message}"));
        let derivative = integrate(|t| -(x / 2.0 + t) * kernel(t), 0.0, upper, 1e-10, 20_000)
            .unwrap_or_else(|message| panic!("parabolic-cylinder U quadrature failed: {message}"));
        return (value, derivative);
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
    let value = integrate(|v| evaluate(v, false), 0.0, 1.0, 3e-14, 8000)
        .unwrap_or_else(|message| panic!("parabolic-cylinder U quadrature failed: {message}"));
    let derivative = integrate(|v| evaluate(v, true), 0.0, 1.0, 3e-14, 8000)
        .unwrap_or_else(|message| panic!("parabolic-cylinder U quadrature failed: {message}"));
    (value, derivative)
}

fn u_pair(a: f64, x: f64) -> (f64, f64) {
    check_finite(a, x);
    if x < 0.0 && a <= -0.5 && a + 0.5 == (a + 0.5).round() {
        let (value, derivative) = u_pair(a, -x);
        let parity = -(PI * a).sin();
        return (parity * value, -parity * derivative);
    }
    if x < 0.0 {
        return series(a, x, Kind::U, 0.0);
    }
    if x > 1.0
        && let Some(result) = u_asymptotic(a, x, false)
    {
        return result;
    }
    if x.abs() > 1.0 {
        if a > -0.5 {
            let exponent = log_scale(a, x);
            let (value, derivative) = u_integral_scaled(a, x);
            return (
                scale_value(-exponent, value),
                scale_value(-exponent, derivative),
            );
        }
        let steps = ((-0.5 - a).floor() as usize) + 1;
        let upper_order = a + steps as f64;
        let (mut current_value, mut current_derivative) = {
            let exponent = log_scale(upper_order, x);
            let (value, derivative) = u_integral_scaled(upper_order, x);
            (
                scale_value(-exponent, value),
                scale_value(-exponent, derivative),
            )
        };
        let (mut upper_value, mut upper_derivative) = {
            let exponent = log_scale(upper_order + 1.0, x);
            let (value, derivative) = u_integral_scaled(upper_order + 1.0, x);
            (
                scale_value(-exponent, value),
                scale_value(-exponent, derivative),
            )
        };
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
        return (current_value, current_derivative);
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

fn ode_pair(a: f64, x: f64, kind: Kind, initial_log_scale: f64) -> (f64, f64) {
    let (mut value, mut derivative) = initial(a, kind, initial_log_scale);
    let mut position = 0.0;
    let mut steps = 0_usize;
    while position != x {
        let remaining = x - position;
        let local_scale = 1.0 + a.abs().sqrt() + position.abs() / 2.0;
        let step = remaining.signum() * (0.5 / local_scale).min(0.1).min(remaining.abs());
        (value, derivative) = ode_advance(a, position, value, derivative, step, kind);
        position += step;
        steps += 1;
        assert!(
            steps <= 2_000_000,
            "parabolic-cylinder ODE work limit exceeded"
        );
    }
    (value, derivative)
}

fn w_asymptotic(a: f64, x: f64) -> Option<(f64, f64)> {
    let t = x.abs();
    let mut ratio = Complex64::new(1.0, 0.0);
    let mut sum = ratio;
    let mut derivative_sum = Complex64::new(0.0, 0.0);
    for k in 1..=1000 {
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

fn w_pair(a: f64, x: f64) -> (f64, f64) {
    check_finite(a, x);
    if x.abs() > 1.0 && x.abs() <= 10.0 {
        return ode_pair(a, x, Kind::W, 0.0);
    }
    if x.abs() > 1.0
        && let Some(result) = w_asymptotic(a, x)
    {
        return result;
    }
    series(a, x, Kind::W, 0.0)
}

/// Parabolic-cylinder function `U(a, x)`.
pub fn u(a: f64, x: f64) -> f64 {
    u_pair(a, x).0
}

/// Argument derivative of [`u`].
pub fn du(a: f64, x: f64) -> f64 {
    u_pair(a, x).1
}

/// Parabolic-cylinder function `V(a, x)`.
pub fn v(a: f64, x: f64) -> f64 {
    check_finite(a, x);
    ode_pair(a, x, Kind::V, 0.0).0
}

/// Argument derivative of [`v`].
pub fn dv(a: f64, x: f64) -> f64 {
    check_finite(a, x);
    ode_pair(a, x, Kind::V, 0.0).1
}

/// Oscillatory parabolic-cylinder function `W(a, x)`.
pub fn w(a: f64, x: f64) -> f64 {
    w_pair(a, x).0
}

/// Argument derivative of [`w`].
pub fn dw(a: f64, x: f64) -> f64 {
    w_pair(a, x).1
}

/// Order- and argument-scaled `U(a, x)` for `x >= 0`.
pub fn u_scaled(a: f64, x: f64) -> f64 {
    check_finite(a, x);
    assert!(x >= 0.0, "scaled cylinder functions require x >= 0");
    let exponent = log_scale(a, x);
    if x * x / 4.0 + a <= 0.0 {
        return ode_pair(a, x, Kind::U, exponent).0;
    }
    if x > 1.0
        && x * x / 4.0 + a > 0.0
        && let Some(result) = u_asymptotic(a, x, true)
    {
        return result.0;
    }
    if a > 0.5 || (a > -0.5 && x > 1.0) {
        return u_integral_scaled(a, x).0;
    }
    if x > 1.0 {
        return scale_value(exponent, u_pair(a, x).0);
    }
    series(a, x, Kind::U, exponent).0
}

/// Order- and argument-scaled `V(a, x)` for `x >= 0`.
pub fn v_scaled(a: f64, x: f64) -> f64 {
    check_finite(a, x);
    assert!(x >= 0.0, "scaled cylinder functions require x >= 0");
    if x > 1.0
        && x * x / 4.0 + a > 0.0
        && let Some(result) = v_asymptotic_scaled(a, x)
    {
        return result.0;
    }
    ode_pair(a, x, Kind::V, -log_scale(a, x)).0
}

/// Standard parabolic-cylinder function `D_nu(x) = U(-nu - 1/2, x)`.
pub fn parabolic_cylinder_d(nu: f64, x: f64) -> f64 {
    check_finite(nu, x);
    u(-nu - 0.5, x)
}

/// Argument derivative of [`parabolic_cylinder_d`].
pub fn d_parabolic_cylinder_d(nu: f64, x: f64) -> f64 {
    check_finite(nu, x);
    du(-nu - 0.5, x)
}

/// Scaled standard parabolic-cylinder function for `x >= 0`.
pub fn parabolic_cylinder_d_scaled(nu: f64, x: f64) -> f64 {
    check_finite(nu, x);
    u_scaled(-nu - 0.5, x)
}
