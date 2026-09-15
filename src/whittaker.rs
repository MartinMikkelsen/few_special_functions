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
    if z.re > 0.0
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

fn w_real_integral(kappa: f64, mu: f64, z: f64) -> Option<f64> {
    let a = mu - kappa + 0.5;
    if a <= 0.0 {
        return None;
    }
    let power = mu + kappa - 0.5;
    let normalization = -z / 2.0 + kappa * z.ln() - libm::lgamma(a);
    let upper = (a + power.abs() + 50.0).max(60.0);
    let m = a.min(1.0);
    let kernel = |v: f64| {
        if v == 0.0 {
            return if a == m {
                (normalization - m.ln()).exp()
            } else {
                0.0
            };
        }
        let u = v.powf(1.0 / m);
        let log_jacobian = -m.ln() + (1.0 - m) * u.ln();
        (normalization - u + (a - 1.0) * u.ln() + power * (u / z).ln_1p() + log_jacobian).exp()
    };
    integrate(kernel, 0.0, upper.powf(m), 3e-14, 10_000).ok()
}

/// Whittaker M for real parameters and positive real argument.
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
pub fn whittaker_w(kappa: f64, mu: f64, z: f64) -> f64 {
    assert!(
        kappa.is_finite() && mu.is_finite() && z.is_finite(),
        "Whittaker parameters must be finite"
    );
    assert!(z > 0.0, "real Whittaker arguments must be positive");
    if kappa == 0.0 && mu.abs() == 0.5 {
        return (-z / 2.0).exp();
    }
    if kappa == 1.0 && mu.abs() == 0.5 {
        return z * (-z / 2.0).exp();
    }
    if let Some((value, _)) = w_asymptotic(kappa.into(), mu.into(), z.into()) {
        return value.re;
    }
    if let Some(value) = w_real_integral(kappa, mu, z) {
        value
    } else {
        w_core(kappa.into(), mu.into(), z.into()).re
    }
}

/// Whittaker W on the principal complex branch.
pub fn whittaker_w_complex(kappa: Complex64, mu: Complex64, z: Complex64) -> Complex64 {
    check_complex(kappa, mu, z, false);
    w_core(kappa, mu, z)
}

/// Argument derivative of [`whittaker_w`].
pub fn d_whittaker_w(kappa: f64, mu: f64, z: f64) -> f64 {
    assert!(
        kappa.is_finite() && mu.is_finite() && z.is_finite(),
        "Whittaker parameters must be finite"
    );
    assert!(z > 0.0, "real Whittaker arguments must be positive");
    if kappa == 0.0 && mu.abs() == 0.5 {
        return -0.5 * (-z / 2.0).exp();
    }
    if kappa == 1.0 && mu.abs() == 0.5 {
        return (1.0 - z / 2.0) * (-z / 2.0).exp();
    }
    let value = whittaker_w(kappa, mu, z);
    let a = mu - kappa + 0.5;
    let shifted = -a * whittaker_w(kappa - 0.5, mu + 0.5, z);
    (-0.5 + (mu + 0.5) / z) * value + shifted / z.sqrt()
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
