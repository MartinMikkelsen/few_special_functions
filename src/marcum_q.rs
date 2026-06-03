use std::f64::consts::{FRAC_2_SQRT_PI, PI};

use crate::debye::inc_gamma_p;

// ─── Mathematical primitives ─────────────────────────────────────────────────

/// Real complementary error function erfc(x) = 1 − erf(x).
fn erfc_real(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_real(-x);
    }
    if x < 4.0 {
        // Series: erf(x) = (2/√π) Σ (-1)^n x^{2n+1} / (n!(2n+1))
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        for n in 1_usize..=80 {
            term *= -x2 / n as f64;
            let contrib = term / (2 * n + 1) as f64;
            sum += contrib;
            if contrib.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        1.0 - FRAC_2_SQRT_PI * sum
    } else {
        // Asymptotic: erfc(x) ~ exp(-x²)/(x√π) · Σ (-1)^m (2m-1)!!/(2x²)^m
        let x2 = x * x;
        let inv_2x2 = 1.0 / (2.0 * x2);
        let mut sum = 1.0;
        let mut term = 1.0;
        let mut min_norm = f64::INFINITY;
        for m in 1_usize..=60 {
            term *= -((2 * m - 1) as f64) * inv_2x2;
            if term.abs() > min_norm {
                break;
            }
            min_norm = term.abs();
            sum += term;
        }
        (-x2).exp() / (x * PI.sqrt()) * sum
    }
}

/// Regularized upper incomplete gamma Q(a, x) = 1 − P(a, x).
fn upper_inc_gamma(a: f64, x: f64) -> f64 {
    1.0 - inc_gamma_p(a, x)
}

/// Scaled modified Bessel function exp(−x) · I_n(x) for integer n ≥ 0.
///
/// Uses the series for x ≤ 700 and the asymptotic expansion for x > 700.
fn bessel_i_scaled(n: usize, x: f64) -> f64 {
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if x > 700.0 {
        // Asymptotic: exp(-x)·I_n(x) ~ 1/√(2πx)·(1 - (4n²-1)/(8x) + ...)
        let n2 = (n as f64).powi(2);
        let inv_8x = 1.0 / (8.0 * x);
        let mut sum = 1.0;
        let mut term = 1.0;
        let mut min_norm = f64::INFINITY;
        for k in 1_usize..=30 {
            let num = 4.0 * n2 - ((2 * k - 1) as f64).powi(2);
            term *= -num * inv_8x / k as f64;
            if term.abs() > min_norm {
                break;
            }
            min_norm = term.abs();
            sum += term;
            if term.abs() < 1e-16 {
                break;
            }
        }
        sum / (2.0 * PI * x).sqrt()
    } else {
        // Series via the log of the first term to avoid overflow:
        //   exp(-x)·I_n(x) = Σ_{k=0}^∞ (x/2)^{n+2k} / (k!·(n+k)!) · exp(-x)
        let half_x = x / 2.0;
        let log_term0 = n as f64 * half_x.ln() - libm::lgamma(n as f64 + 1.0) - x;
        if log_term0 < -745.0 {
            return 0.0;
        }
        let mut term = log_term0.exp();
        let mut sum = term;
        let h2 = half_x * half_x;
        for k in 1_usize..=600 {
            term *= h2 / (k as f64 * (n + k) as f64);
            sum += term;
            if term < sum * 3e-16 {
                break;
            }
        }
        sum
    }
}

/// Ratio I_μ(ξ) / I_{μ−1}(ξ) via the modified Lentz continued fraction.
///
/// CF: I_μ(ξ)/I_{μ−1}(ξ) = 1/(2μ/ξ + 1/(2(μ+1)/ξ + 1/(2(μ+2)/ξ + ...)))
fn bessel_i_ratio(mu: f64, xi: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut f = TINY;
    let mut c = f;
    let mut d = 0.0_f64;
    for j in 1_usize..=300 {
        let b = 2.0 * (mu + j as f64 - 1.0) / xi;
        d += b;
        if d.abs() < TINY {
            d = TINY;
        }
        d = 1.0 / d;
        c = b + 1.0 / c;
        if c.abs() < TINY {
            c = TINY;
        }
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 3e-15 {
            break;
        }
    }
    f
}

// ─── Marcum Q helper functions ────────────────────────────────────────────────

/// log of A_n from eq. (32): lnΓ(μ+½+n) − lnΓ(μ+½−n) − n·ln 2 − lnΓ(n+1)
fn ln_a(n: i32, mu: f64) -> f64 {
    libm::lgamma(mu + 0.5 + n as f64)
        - libm::lgamma(mu + 0.5 - n as f64)
        - n as f64 * 2.0_f64.ln()
        - libm::lgamma(n as f64 + 1.0)
}

/// ζ²/2 from eq. (84), with Taylor expansion near δ = y − x − 1 ≈ 0.
fn half_zeta2(x: f64, y: f64) -> f64 {
    let delta = y - x - 1.0;
    if delta.abs() < 1e-3 {
        let z = delta / (2.0 * x + 1.0).powi(2);
        let c = [
            1.0,
            -(3.0 * x + 1.0) / 3.0,
            ((72.0 * x + 42.0) * x + 7.0) / 36.0,
            -(((2700.0 * x + 2142.0) * x + 657.0) * x + 73.0) / 540.0,
            ((((181440.0 * x + 177552.0) * x + 76356.0) * x + 15972.0) * x + 1331.0) / 12960.0,
        ];
        let s: f64 = c
            .iter()
            .enumerate()
            .map(|(k, &ck)| ck * z.powi(k as i32))
            .sum();
        (2.0 * x + 1.0).powi(3) * s * s / 2.0
    } else {
        let r = (1.0 + 4.0 * x * y).sqrt();
        x + y - r + ((1.0 + r) / (2.0 * y)).ln()
    }
}

fn zeta(x: f64, y: f64) -> f64 {
    (2.0 * half_zeta2(x, y)).sqrt().copysign(x + 1.0 - y)
}

fn theta_over_sin(theta: f64) -> f64 {
    if theta < 1e-4 {
        1.0 + theta * theta / 6.0
    } else {
        theta / theta.sin()
    }
}

fn theta_prime_sin(theta: f64) -> f64 {
    if theta < 1e-4 {
        theta * theta * (theta * theta / 45.0 + 1.0 / 3.0)
    } else {
        1.0 - theta / theta.tan()
    }
}

fn rho_val(theta_o: f64, xi: f64) -> f64 {
    (theta_o * theta_o + xi * xi).sqrt()
}

fn r_val(theta: f64, y: f64, xi: f64) -> f64 {
    let tos = theta_over_sin(theta);
    let xs = xi / tos;
    (1.0 + (1.0 + xs * xs).sqrt()) * tos / (2.0 * y)
}

fn r_prime_sin(theta: f64, y: f64, xi: f64) -> f64 {
    let tos = theta_over_sin(theta);
    let xs = xi / tos;
    (1.0 + 1.0 / (1.0 + xs * xs).sqrt()) * theta_prime_sin(theta) / (2.0 * y)
}

fn f_integrand(theta: f64, y: f64, xi: f64) -> f64 {
    let r0 = r_val(theta, y, xi);
    let d = r0 - theta.cos();
    (r_prime_sin(theta, y, xi) - d * r0) / (d * d + theta.sin().powi(2))
}

fn psi_integrand(theta: f64, xi: f64) -> f64 {
    let tos = theta_over_sin(theta);
    let rv = rho_val(tos, xi);
    theta.cos() * rv - (1.0 + xi * xi).sqrt() - ((tos + rv) / (1.0 + (1.0 + xi * xi).sqrt())).ln()
}

fn f1_f2(x: f64, m: f64) -> (f64, f64) {
    let sq = (4.0 * x + 2.0 * m).sqrt();
    (x + m - sq, x + m + sq)
}

// ─── Algorithm branches ───────────────────────────────────────────────────────

/// Series expansion (section 3) — used when x < 30.
///
/// Q_M(a,b) = exp(−x) · Σ_{n=0}^∞ x^n/n! · Q(M+n, y)
fn marcum_q_small_x(m: f64, x: f64, y: f64) -> f64 {
    let mut s = 0.0;
    let mut term_factor = 1.0_f64;
    for n in 0_usize.. {
        let t = term_factor * upper_inc_gamma(m + n as f64, y);
        s += t;
        if t.abs() <= f64::EPSILON * s.abs() {
            break;
        }
        term_factor *= x / (n as f64 + 1.0);
        if n > 5000 {
            break;
        }
    }
    (-x).exp() * s
}

/// Asymptotic for large ξ = 2√(xy) with M² < 2ξ  (section 4.1).
fn marcum_q_large_xy(m: f64, x: f64, y: f64, xi: f64) -> f64 {
    let delta = y.sqrt() - x.sqrt();
    let sigma = delta * delta / xi;
    let rho0 = (y / x).sqrt();
    let rho_fac = (y / x).powf(m / 2.0) / (8.0 * PI).sqrt();
    let ef = (-delta * delta).exp() * xi.sqrt();

    let phi = if delta.abs() < 1e-5 {
        if sigma == 0.0 {
            0.0
        } else {
            (PI / sigma).sqrt() - 2.0 * xi.sqrt()
        }
    } else {
        (PI / sigma).sqrt() * erfc_real(delta.abs())
    };

    let mut big_phi = phi;
    let mut big_psi = if (rho0 - 1.0).abs() < f64::EPSILON {
        0.5
    } else {
        (rho0.powf(m - 0.5) / 2.0 * erfc_real(delta.abs())).copysign(rho0 - 1.0)
    };

    let mut s = if x > y { 1.0 } else { 0.0 };
    let mut rho_t = rho_fac;
    let mut ef_cur = ef;

    for n in 1_i32..=500 {
        s += big_psi;
        if big_psi.abs() <= f64::EPSILON * s.abs() {
            break;
        }
        rho_t = -rho_t;
        ef_cur /= xi;
        if (m - 1.0) + 0.5 - n as f64 <= 0.0 || m + 0.5 - n as f64 <= 0.0 {
            break;
        }
        big_phi = (ef_cur - sigma * big_phi) / (n as f64 - 0.5);
        let ln_an = ln_a(n, m - 1.0);
        big_psi = rho_t * ln_an.exp() * (1.0 - (ln_a(n, m) - ln_an).exp() / rho0) * big_phi;
    }

    s.max(0.0)
}

/// Recurrence (eq. 14) — stable when y is near the expected range.
fn marcum_q_recurrence(m: f64, x: f64, y: f64, xi: f64) -> f64 {
    let root = (y / x).sqrt();
    let mu_start = m - (m - (2.0 * xi).sqrt() + 1.0).ceil();
    let mut mu = mu_start;
    let mut qm1 = marcum_q_modified(mu - 1.0, x, y);
    let mut q0 = marcum_q_modified(mu, x, y);

    while mu < m - f64::EPSILON * m {
        let cm = root * bessel_i_ratio(mu, xi);
        let q1 = (1.0 + cm) * q0 - cm * qm1;
        qm1 = q0;
        q0 = q1;
        mu += 1.0;
    }
    q0
}

/// Asymptotic expansion for large M (section 4.2).
fn marcum_q_large_m(m: f64, x: f64, y: f64) -> f64 {
    let zv = zeta(x, y);
    let ehalf = (-m * half_zeta2(x, y)).exp();
    let max_k = 100;
    let mut psi = vec![0.0_f64; max_k];
    psi[0] = (PI / (2.0 * m)).sqrt() * erfc_real(-zv * (m / 2.0).sqrt());
    psi[1] = ehalf / m;

    let mut s = 0.0_f64;
    let mut k = 1_usize;
    while k < max_k {
        let bk: f64 = (1..=k).map(|j| psi[j - 1] / m.powi((k - j) as i32)).sum();
        s += bk;
        if bk.abs() <= f64::EPSILON * s.abs() {
            break;
        }
        k += 1;
        if k >= max_k {
            break;
        }
        psi[k] = (k as f64 - 1.0) / m * psi[k - 1] + (-zv).powi(k as i32 - 1) / m * ehalf;
    }

    let result = erfc_real(-zv * (m / 2.0).sqrt()) / 2.0 - (m / (2.0 * PI)).sqrt() * s;
    result.clamp(0.0, 1.0)
}

/// Quadrature fallback (section 5) using composite 4-point GL on 256 panels.
fn marcum_q_quadrature(m: f64, x: f64, y: f64, xi: f64) -> f64 {
    // 4-point GL nodes/weights on [−1,1]
    const GL4_X: [f64; 4] = [
        -0.861136311594952,
        -0.339981043584856,
        0.339981043584856,
        0.861136311594952,
    ];
    const GL4_W: [f64; 4] = [
        0.347854845137454,
        0.652145154862626,
        0.652145154862626,
        0.347854845137454,
    ];

    let upper = PI - 1.0 / 512.0;
    let n_panels = 256_usize;
    let h = upper / n_panels as f64;

    let mut integral = 0.0;
    for panel in 0..n_panels {
        let a = panel as f64 * h;
        let b = a + h;
        let mid = (a + b) / 2.0;
        let half = (b - a) / 2.0;
        for (&t, &w) in GL4_X.iter().zip(GL4_W.iter()) {
            let theta = mid + half * t;
            integral += w * half * (m * psi_integrand(theta, xi)).exp() * f_integrand(theta, y, xi);
        }
    }
    integral *= (-m * half_zeta2(x, y)).exp() / PI;
    if x + 1.0 < y {
        integral
    } else {
        1.0 + integral
    }
}

// ─── Core dispatch ────────────────────────────────────────────────────────────

fn marcum_q_modified(m: f64, x: f64, y: f64) -> f64 {
    let xi = 2.0 * (x * y).sqrt();
    let (f1, f2) = f1_f2(x, m);

    let mut qv = if x < 30.0 {
        marcum_q_small_x(m, x, y)
    } else if xi > 30.0 && m * m < 2.0 * xi {
        marcum_q_large_xy(m, x, y, xi)
    } else if f1 < y && y < f2 && m < 135.0 {
        marcum_q_recurrence(m, x, y, xi)
    } else if f1 < y && y < f2 {
        marcum_q_large_m(m, x, y)
    } else {
        marcum_q_quadrature(m, x / m, y / m, xi / m)
    };

    if qv > 1.0 && qv < 1.0 + f64::EPSILON {
        qv = 1.0;
    }
    qv
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Generalized Marcum Q-function Q_μ(a, b).
///
/// Defined as
///
/// ```text
/// Q_μ(a, b) = exp(−(a²+b²)/2) · Σ_{k=0}^∞ (a/b)^{μ−1+k} · I_{μ−1+k}(ab)
/// ```
///
/// where I_ν is the modified Bessel function of the first kind. Returns a
/// value in \[0, 1\].
///
/// Widely used in communications and radar signal processing — for integer
/// order M it gives the probability that a non-central chi-squared random
/// variable with 2M degrees of freedom and non-centrality parameter a²
/// exceeds the threshold b².
///
/// The implementation automatically selects among series expansion, asymptotic
/// expansion, recurrence, and quadrature fallback depending on the input
/// parameters.
///
/// Reference: <https://arxiv.org/pdf/1311.0681v1>
///
/// # Panics
/// Panics if μ < 0.5, a < 0, or b < 0.
///
/// # Examples
///
/// ```
/// use few_special_functions::marcum_q::marcum_q;
///
/// // Q_1(a, 0) = 1 for any a ≥ 0 (integrating the full distribution)
/// assert!((marcum_q(1.0, 2.0, 0.0) - 1.0).abs() < 1e-12);
///
/// // Q_1(0, b) = exp(−b²/2) (Rayleigh tail)
/// let b = 2.0_f64;
/// assert!((marcum_q(1.0, 0.0, b) - (-b * b / 2.0).exp()).abs() < 1e-10);
/// ```
pub fn marcum_q(mu: f64, a: f64, b: f64) -> f64 {
    assert!(mu >= 0.5, "μ must be ≥ 0.5, got {mu}");
    assert!(a >= 0.0, "a must be ≥ 0, got {a}");
    assert!(b >= 0.0, "b must be ≥ 0, got {b}");
    marcum_q_modified(mu, a * a / 2.0, b * b / 2.0)
}

/// Derivative ∂Q_M(a,b)/∂b of the Marcum Q-function (integer order M).
///
/// ```text
/// dQ_M/db = −(b^M / a^{M−1}) · exp(−(a²+b²)/2) · I_{M−1}(ab)
/// ```
///
/// Computed in a numerically stable form using the scaled Bessel function
/// exp(−ab) · I_{M−1}(ab) to avoid overflow for large arguments.
///
/// # Panics
/// Panics if M < 1 or a = 0.
///
/// # Examples
///
/// ```
/// use few_special_functions::marcum_q::dq_db;
///
/// // dQ/db is always ≤ 0 (Q is non-increasing in b)
/// assert!(dq_db(1, 1.0, 1.0) <= 0.0);
/// ```
pub fn dq_db(m: u32, a: f64, b: f64) -> f64 {
    assert!(m >= 1, "M must be ≥ 1, got {m}");
    assert!(a != 0.0, "a must be nonzero");
    let n = m as usize - 1;
    let ab = a * b;
    // dQ/db = −b^M/a^{M−1} · exp(−(a−b)²/2) · [exp(−ab)·I_{M−1}(ab)]
    let log_coeff = m as f64 * b.ln() - (m as f64 - 1.0) * a.ln();
    let scaled = bessel_i_scaled(n, ab);
    -(log_coeff.exp() * (-(a - b).powi(2) / 2.0).exp() * scaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_known() {
        // erfc(0) = 1, erfc(∞) = 0
        assert!((erfc_real(0.0) - 1.0).abs() < 1e-15);
        // erfc(1) ≈ 0.157299
        assert!((erfc_real(1.0) - 0.15729920705028513).abs() < 1e-12);
        // erfc(-1) = 2 - erfc(1)
        assert!((erfc_real(-1.0) - (2.0 - erfc_real(1.0))).abs() < 1e-15);
    }

    #[test]
    fn bessel_i_scaled_known() {
        // exp(-x)·I_0(x): at x=1 ≈ 0.46575
        let v = bessel_i_scaled(0, 1.0);
        assert!((v - 0.46575960759364743).abs() < 1e-10, "got {v}");
    }

    #[test]
    fn marcum_q_known() {
        assert!((marcum_q(1.0, 0.2, 0.6) - 0.838249985438908).abs() < 1e-9);
        assert!((marcum_q(5.0, 0.2, 0.6) - 0.999998670306184).abs() < 1e-9);
        assert!((marcum_q(1.0, 1.2, 1.6) - 0.501536568390858).abs() < 1e-9);
    }
}
