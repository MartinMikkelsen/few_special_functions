/// Regularized lower incomplete gamma P(a, x) = γ(a,x) / Γ(a).
///
/// Uses a convergent series for x < a+1 and the Lentz continued fraction
/// for Q(a,x) = 1 - P(a,x) when x ≥ a+1.
pub(crate) fn inc_gamma_p(a: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && x >= 0.0);
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        inc_gamma_series(a, x)
    } else {
        1.0 - inc_gamma_cf(a, x)
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
/// Defined as D_n(β, x) = (n/xⁿ) ∫₀ˣ tⁿ / (eᵝᵗ - 1) dt, normalized so
/// D_n(β, 0) = 1.
///
/// Uses the series from doi:10.1007/s10765-007-0256-1.
///
/// # Panics
/// Panics if n ≤ 0, β ≤ 0, or x < 0.
pub fn debye_function(n: f64, beta: f64, x: f64) -> f64 {
    debye_function_tol(n, beta, x, 1e-15, 2000)
}

/// Like [`debye_function`] but with explicit convergence tolerance and
/// maximum number of series terms.
pub fn debye_function_tol(n: f64, beta: f64, x: f64, tol: f64, max_terms: usize) -> f64 {
    assert!(n > 0.0, "n must be positive, got {n}");
    assert!(beta > 0.0, "beta must be positive, got {beta}");
    assert!(x >= 0.0, "x must be non-negative, got {x}");

    if x == 0.0 {
        return 1.0;
    }

    let a = n + 1.0;
    let gamma_n1 = libm::tgamma(a); // Γ(n+1)

    let mut sum = 0.0;
    let mut c = 1.0_f64; // c_i = (β)_i / i! (Pochhammer / factorial)

    for i in 0..=max_terms {
        let psi = beta + i as f64;
        let p = inc_gamma_p(a, psi * x);
        let gamma_lower = p * gamma_n1; // γ(n+1, ψx)

        let term = c * gamma_lower / psi.powf(a);
        sum += term;

        if term.abs() < tol * sum.abs() {
            return n * sum / x.powf(n);
        }

        c *= psi / (i as f64 + 1.0);
    }

    n * sum / x.powf(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_zero() {
        assert_eq!(debye_function(1.0, 1.0, 0.0), 1.0);
        assert_eq!(debye_function(3.0, 2.0, 0.0), 1.0);
    }

    #[test]
    fn inc_gamma_p_known() {
        // P(1, 1) = 1 - e⁻¹ ≈ 0.6321
        assert!((inc_gamma_p(1.0, 1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
        // P(2, 1) = 1 - 2e⁻¹ ≈ 0.2642
        assert!((inc_gamma_p(2.0, 1.0) - (1.0 - 2.0 * (-1.0_f64).exp())).abs() < 1e-12);
    }
}
