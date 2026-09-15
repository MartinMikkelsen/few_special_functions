/// Regularized upper incomplete gamma Q(a, x) = Γ(a,x) / Γ(a).
///
/// Uses a convergent series for small `x` and a continued fraction in the tail.
pub(crate) fn inc_gamma_q(a: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && x >= 0.0);
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - inc_gamma_series(a, x)
    } else {
        inc_gamma_cf(a, x)
    }
}

/// Series for P(a, x): exp(-x + a·ln(x) - ln Γ(a)) · Σ xᵏ / (a·(a+1)·…·(a+k))
fn inc_gamma_series(a: f64, x: f64) -> f64 {
    let log_prefix = -x + a * x.ln() - libm::lgamma(a);
    let mut ap = a;
    let mut term = 1.0 / a;
    let mut sum = term;
    for _ in 0..300 {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * 3e-15 {
            break;
        }
    }
    log_prefix.exp() * sum
}

/// Lentz CF for Q(a, x) = Γ(a,x)/Γ(a).
/// From NR §6.2: Q = exp(-x + a·ln(x) - ln Γ(a)) · h
/// where h is the CF value.
fn inc_gamma_cf(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let log_prefix = -x + a * x.ln() - libm::lgamma(a);

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = if b.abs() < TINY { TINY } else { 1.0 / b };
    let mut h = d;

    for i in 1_usize..=300 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        d = 1.0 / d;
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 3e-15 {
            break;
        }
    }

    log_prefix.exp() * h
}

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
/// Uses the series expansion from doi:10.1007/s10765-007-0256-1.
///
/// # Panics
/// Panics if n ≤ 0, β ≤ 0, or x < 0.
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
    debye_function_tol(n, beta, x, 1e-35, 2000)
}

/// Convenience form of [`debye_function`] with order `n = 1`.
pub fn debye_function_order_one(beta: f64, x: f64) -> f64 {
    debye_function(1.0, beta, x)
}

/// Like [`debye_function`] but with explicit convergence tolerance `tol` and
/// maximum number of series terms `max_terms`.
///
/// Useful when you need a looser tolerance for speed, or want to guard against
/// slow convergence for unusual parameter combinations.
///
/// # Examples
///
/// ```
/// use few_special_functions::debye::debye_function_tol;
///
/// // Should agree with debye_function to within a few times the tolerance
/// use few_special_functions::debye::debye_function;
/// let v1 = debye_function(2.0, 1.0, 1.0);
/// let v2 = debye_function_tol(2.0, 1.0, 1.0, 1e-8, 500);
/// assert!((v1 - v2).abs() < 1e-5);
/// ```
pub fn debye_function_tol(n: f64, beta: f64, x: f64, tol: f64, max_terms: usize) -> f64 {
    assert!(
        n.is_finite() && n > 0.0,
        "n must be positive and finite, got {n}"
    );
    assert!(
        beta.is_finite() && beta > 0.0 && beta < n + 1.0,
        "beta must satisfy 0 < beta < n + 1, got {beta}"
    );
    assert!(x >= 0.0, "x must be non-negative, got {x}");
    assert!(
        tol.is_finite() && tol > 0.0,
        "tol must be positive and finite"
    );
    assert!(max_terms > 0, "max_terms must be positive");

    if x == 0.0 {
        return if beta < 1.0 {
            0.0
        } else if beta == 1.0 {
            1.0
        } else {
            f64::INFINITY
        };
    }
    if x.is_infinite() {
        return 0.0;
    }
    if beta == 1.0 && n < f64::EPSILON.sqrt() {
        return 1.0;
    }

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
    let integral = crate::numerics::integrate(integrand, 0.0, upper, relative_tolerance, max_terms)
        .unwrap_or_else(|message| panic!("Debye quadrature failed: {message}"));

    let result = (n / m) * integral * scale.powf(1.0 - beta) * (scale / x).powf(n);
    if result.is_finite() && result > 0.0 {
        return result;
    }

    ((n / m).ln() + (1.0 - beta) * scale.ln() + n * (scale.ln() - x.ln()) + integral.ln()).exp()
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
    fn inc_gamma_q_relative_accuracy_in_the_tail() {
        assert_eq!(inc_gamma_q(1.0, 0.0), 1.0);
        // Q(1, x) = e^{-x} exactly, over a range where 1 - P(1, x) rounds to zero.
        for &x in &[0.5_f64, 1.5, 2.0, 10.0, 40.0, 100.0, 300.0, 700.0] {
            let want = (-x).exp();
            let rel = (inc_gamma_q(1.0, x) - want).abs() / want;
            assert!(rel < 1e-12, "Q(1,{x}) rel {rel:.2e}");
        }
        // Q(½, x) = erfc(√x)
        #[allow(clippy::excessive_precision)]
        let half = [
            (0.5_f64, 0.317310507862914226_f64),
            (2.0, 0.0455002638963585105),
            (20.0, 2.53962858946993629e-10),
            (100.0, 2.08848758376254478e-45),
            (400.0, 5.39586561160790086e-176),
        ];
        for (x, want) in half {
            let rel = (inc_gamma_q(0.5, x) - want).abs() / want;
            assert!(rel < 1e-11, "Q(0.5,{x}) rel {rel:.2e}");
        }
    }
}
