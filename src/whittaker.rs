//! Whittaker M and W functions on the principal complex branch.

use crate::numerics::{gamma_complex, integrate, kummer_m, log_gamma_complex, reciprocal_gamma};
use num_complex::Complex64;

fn finite(z: Complex64) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

fn check_complex(kappa: Complex64, mu: Complex64, z: Complex64, is_m: bool) {
    assert!(
        finite(kappa) && finite(mu) && finite(z),
        "Whittaker parameters must be finite"
    );
    assert!(
        z != Complex64::new(0.0, 0.0),
        "Whittaker functions require a nonzero argument"
    );
    if is_m && mu.im == 0.0 {
        let twice_mu = 2.0 * mu.re;
        assert!(
            !(twice_mu <= -1.0 && twice_mu == twice_mu.round()),
            "Whittaker M has a parameter pole at negative integer 2*mu"
        );
    }
}

fn check_real(kappa: f64, mu: f64, z: f64) {
    assert!(
        kappa.is_finite() && mu.is_finite() && z.is_finite(),
        "Whittaker parameters must be finite"
    );
    assert!(z > 0.0, "real Whittaker arguments must be positive");
}

fn m_asymptotic(kappa: Complex64, mu: Complex64, z: Complex64) -> Option<Complex64> {
    let a = mu - kappa + 0.5;
    let b = 2.0 * mu + 1.0;
    let mut term = Complex64::new(1.0, 0.0);
    let mut sum = term;
    for n in 1..=1000 {
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

fn m_formula(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    let a = mu - kappa + 0.5;
    let b = 2.0 * mu + 1.0;
    (-z / 2.0 + (mu + 0.5) * z.ln()).exp() * kummer_m(a, b, z)
}

fn d_m_formula(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    let value = m_formula(kappa, mu, z);
    let a = mu - kappa + 0.5;
    let shifted = a / (2.0 * mu + 1.0) * m_formula(kappa - 0.5, mu + 0.5, z);
    (-0.5 + (mu + 0.5) / z) * value + shifted / z.sqrt()
}

fn propagate_whittaker(
    kappa: Complex64,
    mu: Complex64,
    z: Complex64,
    is_m: bool,
) -> (Complex64, Complex64) {
    let radius = z.norm();
    let direction = z / radius;
    let mut position = 4.0;
    let initial_z = direction * position;
    let (mut value, initial_derivative) = if is_m {
        (
            m_formula(kappa, mu, initial_z),
            d_m_formula(kappa, mu, initial_z),
        )
    } else {
        (
            w_formula(kappa, mu, initial_z),
            d_w_formula(kappa, mu, initial_z),
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
        let step = 0.002_f64.min(radius - position);
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
    (value, derivative / direction)
}

fn m_core(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    // At a = 1/2 + mu - kappa a non-positive integer, M(a, b, z) is a
    // polynomial and the exponentially growing asymptotic form vanishes.
    let a = mu - kappa + 0.5;
    let terminating = a.im == 0.0 && a.re <= 0.0 && a.re == a.re.round();
    if !terminating
        && z.re > 0.0
        && z.norm() > 300.0
        && let Some(value) = m_asymptotic(kappa, mu, z)
    {
        return value;
    }
    if z.norm() > 10.0 && z.re <= z.norm() / 4.0 {
        return propagate_whittaker(kappa, mu, z, true).0;
    }
    m_formula(kappa, mu, z)
}

fn w_connection(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    gamma_complex(-2.0 * mu) * reciprocal_gamma(0.5 - mu - kappa) * m_core(kappa, mu, z)
        + gamma_complex(2.0 * mu) * reciprocal_gamma(0.5 + mu - kappa) * m_core(kappa, -mu, z)
}

fn w_removable_limit(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    let symmetric = |step: f64| {
        let h = Complex64::new(step, 0.0);
        0.5 * (w_connection(kappa, mu + h, z) + w_connection(kappa, mu - h, z))
    };
    let s1 = symmetric(0.01);
    let s2 = symmetric(0.005);
    let s3 = symmetric(0.0025);
    let r1 = (4.0 * s2 - s1) / 3.0;
    let r2 = (4.0 * s3 - s2) / 3.0;
    (16.0 * r2 - r1) / 15.0
}

fn w_asymptotic(kappa: Complex64, mu: Complex64, z: Complex64) -> Option<(Complex64, Complex64)> {
    let mut term = Complex64::new(1.0, 0.0);
    let mut sum = term;
    let mut derivative_sum = Complex64::new(0.0, 0.0);
    for n in 1..=1000 {
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

fn w_formula(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    if kappa == Complex64::new(0.0, 0.0)
        && (mu == Complex64::new(0.5, 0.0) || mu == Complex64::new(-0.5, 0.0))
    {
        return (-z / 2.0).exp();
    }
    if kappa == Complex64::new(1.0, 0.0)
        && (mu == Complex64::new(0.5, 0.0) || mu == Complex64::new(-0.5, 0.0))
    {
        return z * (-z / 2.0).exp();
    }
    if z.re > 0.0
        && let Some((value, _)) = w_asymptotic(kappa, mu, z)
    {
        return value;
    }
    if mu.im == 0.0 && 2.0 * mu.re == (2.0 * mu.re).round() {
        w_removable_limit(kappa, mu, z)
    } else {
        w_connection(kappa, mu, z)
    }
}

fn d_w_formula(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    if kappa == Complex64::new(0.0, 0.0)
        && (mu == Complex64::new(0.5, 0.0) || mu == Complex64::new(-0.5, 0.0))
    {
        return -0.5 * (-z / 2.0).exp();
    }
    if kappa == Complex64::new(1.0, 0.0)
        && (mu == Complex64::new(0.5, 0.0) || mu == Complex64::new(-0.5, 0.0))
    {
        return (1.0 - z / 2.0) * (-z / 2.0).exp();
    }
    if z.re > 0.0
        && let Some((_, derivative)) = w_asymptotic(kappa, mu, z)
    {
        return derivative;
    }
    let value = w_formula(kappa, mu, z);
    let a = mu - kappa + 0.5;
    let shifted = -a * w_formula(kappa - 0.5, mu + 0.5, z);
    (-0.5 + (mu + 0.5) / z) * value + shifted / z.sqrt()
}

fn w_core(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    if z.norm() > 10.0 && z.re <= 0.0 {
        propagate_whittaker(kappa, mu, z, false).0
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
/// integer.
fn terminating_order(kappa: f64, mu: f64) -> Option<usize> {
    let order = kappa - mu - 0.5;
    (order >= 0.0 && order == order.round()).then_some(order as usize)
}

/// W for `a = 1/2 + mu - kappa > 0`, from
/// `W = exp(-z/2) z^kappa / Γ(a) ∫_0^∞ e^{-u} u^{a-1} (1 + u/z)^{mu+kappa-1/2} du`
/// (DLMF section 13.16).
///
/// The integral is evaluated without the prefactor, so the quadrature
/// tolerance is relative to an O(1) quantity; the substitution `u = v^(1/m)`,
/// `m = min(a, 1)`, removes the `u^{a-1}` endpoint singularity.
fn w_real_integral(kappa: f64, mu: f64, z: f64) -> f64 {
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
        integrate(kernel, 0.0, upper.powf(m), tolerance, 20_000).unwrap_or_else(|message| {
            panic!("Whittaker W quadrature failed (kappa = {kappa}, mu = {mu}, z = {z}): {message}")
        })
    };
    let mut integral = quadrature(1e-15);
    // A small integral needs a tolerance relative to its own size.
    if integral.abs() < 1.0 {
        integral = quadrature(1e-15 * integral.abs());
    }
    (w_log_prefactor(kappa, z) + integral.abs().ln()).exp() * integral.signum()
}

/// W from its expression in M (DLMF 13.14.33), accepted only when its two
/// terms cancel by less than a factor of 16.
///
/// The terms have Gamma-function poles at integer `2 mu`, and both grow like
/// `exp(z/2)` while W decays, so the check rejects those cases and large `z`.
fn w_real_connection(kappa: f64, mu: f64, z: f64) -> Option<f64> {
    let (kappa_c, mu_c, z_c) = (
        Complex64::from(kappa),
        Complex64::from(mu),
        Complex64::from(z),
    );
    let first = (gamma_complex(-2.0 * mu_c)
        * reciprocal_gamma(0.5 - mu_c - kappa_c)
        * m_core(kappa_c, mu_c, z_c))
    .re;
    let second = (gamma_complex(2.0 * mu_c)
        * reciprocal_gamma(0.5 + mu_c - kappa_c)
        * m_core(kappa_c, -mu_c, z_c))
    .re;
    let value = first + second;
    (value.is_finite() && first.abs() + second.abs() <= 16.0 * value.abs()).then_some(value)
}

/// Largest `z` at which [`w_real_value`] tries [`w_real_connection`]; above it
/// the exponential growth of the M terms makes the cancellation check fail.
const CONNECTION_MAX_Z: f64 = 10.0;

/// W for real parameters, `mu >= 0` and `z > 0`.
fn w_real_value(kappa: f64, mu: f64, z: f64) -> f64 {
    if let Some(order) = terminating_order(kappa, mu) {
        return w_terminating(kappa, mu, z, order).0;
    }
    if let Some((value, _)) = w_asymptotic(kappa.into(), mu.into(), z.into()) {
        return value.re;
    }
    if 2.0 * mu != (2.0 * mu).round()
        && z <= CONNECTION_MAX_Z
        && let Some(value) = w_real_connection(kappa, mu, z)
    {
        return value;
    }
    let a = 0.5 + mu - kappa;
    if a > 0.0 {
        return w_real_integral(kappa, mu, z);
    }
    // a < 0: start where a lies in (0, 2] and recur upward in kappa with
    // W_{k+1} = (z - 2k) W_k - ((k - 1/2)² - mu²) W_{k-1} (DLMF 13.3.7 for
    // U(a, 1 + 2mu, z) at fixed mu). W is the minimal solution as a → +∞,
    // so decreasing a, which is increasing kappa, is the stable direction.
    let steps = (-a).ceil() as usize;
    let mut k = kappa - steps as f64;
    let mut previous = w_real_integral(k - 1.0, mu, z);
    let mut current = w_real_integral(k, mu, z);
    for _ in 0..steps {
        let next = (z - 2.0 * k) * current - ((k - 0.5) * (k - 0.5) - mu * mu) * previous;
        (previous, current) = (current, next);
        k += 1.0;
    }
    current
}

/// W and W' for real parameters and `z > 0`.
fn w_real(kappa: f64, mu: f64, z: f64) -> (f64, f64) {
    // W is even in mu.
    let mu = mu.abs();
    if let Some(order) = terminating_order(kappa, mu) {
        return w_terminating(kappa, mu, z, order);
    }
    let value = w_real_value(kappa, mu, z);
    // DLMF section 13.15: W' = (-1/2 + (mu + 1/2)/z) W - a W_{kappa-1/2, mu+1/2} / sqrt(z).
    let a = 0.5 + mu - kappa;
    let shifted = w_real_value(kappa - 0.5, mu + 0.5, z);
    (
        value,
        (-0.5 + (mu + 0.5) / z) * value - a * shifted / z.sqrt(),
    )
}

/// Whittaker M for real parameters and positive real argument.
///
/// # Accuracy
///
/// Measured against mpmath for `kappa ∈ [−5, 5]`, `mu ∈ [−1.3, 3.5]` and
/// `z ∈ [0.01, 1000]` (see the crate README): relative error below 2e-13. The
/// largest errors are at `z = 1000`, where a relative change of `eps` in `z`
/// changes `M` by about `(z/2)·eps`.
pub fn whittaker_m(kappa: f64, mu: f64, z: f64) -> f64 {
    assert!(z > 0.0, "real Whittaker arguments must be positive");
    whittaker_m_complex(kappa.into(), mu.into(), z.into()).re
}

/// Whittaker M on the principal complex branch.
pub fn whittaker_m_complex(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    check_complex(kappa, mu, z, true);
    m_core(kappa, mu, z)
}

/// Argument derivative of [`whittaker_m`].
///
/// Measured on the same grid as [`whittaker_m`]: relative error below 3e-13.
pub fn d_whittaker_m(kappa: f64, mu: f64, z: f64) -> f64 {
    assert!(z > 0.0, "real Whittaker arguments must be positive");
    d_whittaker_m_complex(kappa.into(), mu.into(), z.into()).re
}

/// Argument derivative of [`whittaker_m_complex`].
pub fn d_whittaker_m_complex(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    check_complex(kappa, mu, z, true);
    if z.norm() > 10.0 && z.re <= z.norm() / 4.0 {
        return propagate_whittaker(kappa, mu, z, true).1;
    }
    let value = m_core(kappa, mu, z);
    let a = mu - kappa + 0.5;
    let shifted = a / (2.0 * mu + 1.0) * m_core(kappa - 0.5, mu + 0.5, z);
    (-0.5 + (mu + 0.5) / z) * value + shifted / z.sqrt()
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
/// # Panics
///
/// Panics if the arguments are not finite, if `z <= 0`, or if the quadrature
/// of the integral representation does not converge.
pub fn whittaker_w(kappa: f64, mu: f64, z: f64) -> f64 {
    check_real(kappa, mu, z);
    w_real(kappa, mu, z).0
}

/// Whittaker W on the principal complex branch.
pub fn whittaker_w_complex(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    check_complex(kappa, mu, z, false);
    w_core(kappa, mu, z)
}

/// Argument derivative of [`whittaker_w`].
///
/// Measured on the same grid as [`whittaker_w`]: relative error below 3e-13.
pub fn d_whittaker_w(kappa: f64, mu: f64, z: f64) -> f64 {
    check_real(kappa, mu, z);
    w_real(kappa, mu, z).1
}

/// Argument derivative of [`whittaker_w_complex`].
pub fn d_whittaker_w_complex(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    check_complex(kappa, mu, z, false);
    if z.norm() > 10.0 && z.re <= 0.0 {
        propagate_whittaker(kappa, mu, z, false).1
    } else {
        d_w_formula(kappa, mu, z)
    }
}
