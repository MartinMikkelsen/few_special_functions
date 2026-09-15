//! Bose–Einstein integrals for integer and half-integer orders.

use crate::numerics::zeta_real;

fn check_domain(k: f64, eta: f64) {
    assert!(
        k.is_finite() && k >= -4.5 && 2.0 * k == (2.0 * k).round(),
        "k must be an integer or half-integer at least -9/2, got {k}"
    );
    assert!(
        !eta.is_nan() && eta <= 0.0,
        "eta must be nonpositive, got {eta}"
    );
}

fn fugacity_scaled(k: f64, eta: f64) -> f64 {
    let z = eta.exp();
    let q = -eta.exp_m1();
    let mut power = 1.0;
    let mut total = 1.0;
    let mut correction = 0.0;
    for n in 2..=100_000 {
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
            return total;
        }
    }
    panic!("Bose-Einstein fugacity series did not converge")
}

fn near_zero_series(k: f64, eta: f64) -> f64 {
    let integer_order = k >= 0.0 && k == k.round();
    let mut total = if integer_order {
        let order = k as usize;
        let harmonic = (1..=order).map(|j| 1.0 / j as f64).sum::<f64>();
        eta.powi(order as i32) / libm::tgamma(k + 1.0) * (harmonic - (-eta).ln())
    } else {
        libm::tgamma(-k) * (-eta).powf(k)
    };
    if total.is_infinite() {
        return total;
    }

    let mut coefficient = 1.0;
    let mut correction = 0.0;
    let mut small_terms = 0;
    for m in 0..=100_000 {
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
                return total;
            }
        }
        coefficient *= eta / (m + 1) as f64;
    }
    panic!("Bose-Einstein expansion at eta = 0 did not converge")
}

fn normalized(k: f64, eta: f64) -> f64 {
    if eta == f64::NEG_INFINITY {
        return 0.0;
    }
    if eta == 0.0 {
        return if k > 0.0 {
            zeta_real(k + 1.0)
        } else {
            f64::INFINITY
        };
    }
    if k == 0.0 {
        return if eta < -std::f64::consts::LN_2 {
            -(-eta.exp()).ln_1p()
        } else {
            -(-eta.exp_m1()).ln()
        };
    }
    if k == k.round() && (-4.0..0.0).contains(&k) {
        let z = eta.exp();
        let q = -eta.exp_m1();
        return match k as i32 {
            -1 => z / q,
            -2 => z / q.powi(2),
            -3 => z * (1.0 + z) / q.powi(3),
            -4 => z * (1.0 + 4.0 * z + z * z) / q.powi(4),
            _ => unreachable!(),
        };
    }
    if eta <= -1.0 || k >= 20.0 {
        return eta.exp() * fugacity_scaled(k, eta);
    }
    near_zero_series(k, eta)
}

/// Normalized Bose–Einstein integral `Li_(k+1)(exp(eta))`.
///
/// `k` must be an integer or half-integer with `k >= -4.5`, and `eta <= 0`.
/// Negative orders use the polylogarithm continuation.
pub fn bose_einstein_integral_norm(k: f64, eta: f64) -> f64 {
    check_domain(k, eta);
    normalized(k, eta)
}

/// Unnormalized Bose–Einstein integral.
///
/// This is `Gamma(k+1) * bose_einstein_integral_norm(k, eta)` and requires
/// `k > -1` so that the defining integral exists.
pub fn bose_einstein_integral(k: f64, eta: f64) -> f64 {
    check_domain(k, eta);
    assert!(k > -1.0, "the unnormalized integral requires k > -1");
    if eta == f64::NEG_INFINITY {
        return 0.0;
    }
    let value = normalized(k, eta);
    if k <= 170.5 && value >= f64::MIN_POSITIVE {
        let scale = libm::tgamma(k + 1.0);
        if scale.is_finite() {
            return scale * value;
        }
    }
    let log_value = if eta < -1.0 {
        eta + fugacity_scaled(k, eta).ln()
    } else {
        value.ln()
    };
    (libm::lgamma(k + 1.0) + log_value).exp()
}
