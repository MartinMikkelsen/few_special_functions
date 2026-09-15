use std::f64::consts::PI;

use crate::debye::inc_gamma_q;

// ─── Mathematical primitives ─────────────────────────────────────────────────

/// Real complementary error function erfc(x) = 1 − erf(x).
fn erfc_real(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_real(-x);
    }
    if x < 6.0 {
        // erfc(x) = Q(½, x²). The erf series this replaces reached terms of order 4·10⁴
        // before being subtracted from 1, and the asymptotic series below x ≈ 6 was
        // truncated at its smallest term while that term was still of order 10⁻⁸.
        inc_gamma_q(0.5, x * x)
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

/// Ratio A_n(μ)/A_{n−1}(μ) for the coefficients of eq. (32).
///
/// A_n(μ) = 2^{−n} Γ(μ+½+n) / (n! Γ(μ+½−n)) telescopes to (μ² − (n−½)²)/(2n). Unlike
/// evaluating the Γ quotient through lnΓ, this keeps the sign: Γ(μ+½−n) is negative for
/// half of the negative half-integer arguments reached when n > μ + ½.
fn a_ratio(n: i32, mu: f64) -> f64 {
    let nh = n as f64 - 0.5;
    (mu * mu - nh * nh) / (2.0 * n as f64)
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
        let t = term_factor * inc_gamma_q(m + n as f64, y);
        s += t;
        // s stays 0 while the leading Q(m+n, y) underflow; testing then would stop
        // the sum before it reaches its terms of largest magnitude
        if s > 0.0 && t <= f64::EPSILON * s {
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
    let mut a_lo = 1.0_f64; // A_n(μ−1), starting from A_0 = 1
    let mut a_hi = 1.0_f64; // A_n(μ)

    for n in 1_i32..=500 {
        s += big_psi;
        if big_psi.abs() <= f64::EPSILON * s.abs() {
            break;
        }
        rho_t = -rho_t;
        ef_cur /= xi;
        big_phi = (ef_cur - sigma * big_phi) / (n as f64 - 0.5);
        a_lo *= a_ratio(n, m - 1.0);
        a_hi *= a_ratio(n, m);
        big_psi = rho_t * (a_lo - a_hi / rho0) * big_phi;
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

/// Centered Poisson-gamma mixture for large M.
fn marcum_q_large_m(m: f64, x: f64, y: f64) -> f64 {
    let mode = x.floor() as usize;
    let mut sum = inc_gamma_q(m + mode as f64, y);
    let mut weights = 1.0;

    let mut weight = 1.0;
    for n in (1..=mode).rev() {
        weight *= n as f64 / x;
        sum += weight * inc_gamma_q(m + (n - 1) as f64, y);
        weights += weight;
        let ratio = (n - 1) as f64 / x;
        if weight * ratio <= f64::EPSILON * weights * (1.0 - ratio) {
            break;
        }
    }

    weight = 1.0;
    let mut n = mode;
    loop {
        n += 1;
        weight *= x / n as f64;
        sum += weight * inc_gamma_q(m + n as f64, y);
        weights += weight;
        let ratio = x / (n + 1) as f64;
        if weight * ratio <= f64::EPSILON * weights * (1.0 - ratio) {
            break;
        }
    }
    sum / weights
}

// 4-point Gauss-Legendre nodes/weights on [−1,1] (Abramowitz & Stegun table 25.4).
const GL4_X: [f64; 4] = [
    -0.86113631159405258,
    -0.33998104358485626,
    0.33998104358485626,
    0.86113631159405258,
];
const GL4_W: [f64; 4] = [
    0.34785484513745386,
    0.65214515486254614,
    0.65214515486254614,
    0.34785484513745386,
];

/// Quadrature fallback (section 5) using composite 4-point GL on 256 panels.
fn marcum_q_quadrature(m: f64, x: f64, y: f64, xi: f64) -> f64 {
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
    if y == 0.0 {
        // Q_μ(a, 0) = 1; the quadrature branch would divide by y here.
        return 1.0;
    }
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
/// Q_μ(a, b) = exp(−(a²+b²)/2) · Σ_{k=1−μ}^∞ (a/b)^k · I_k(ab)
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

/// Standard Marcum Q-function of order one.
pub fn marcum_q_order_one(a: f64, b: f64) -> f64 {
    marcum_q(1.0, a, b)
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

/// Derivative with respect to `b` of the order-one Marcum Q-function.
pub fn dq_db_order_one(a: f64, b: f64) -> f64 {
    dq_db(1, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl4_rule_is_exact_to_degree_7() {
        // An n-point Gauss-Legendre rule integrates polynomials up to degree
        // 2n−1 exactly; ∫_{−1}^{1} x^k dx = 2/(k+1) for even k, 0 for odd k.
        for k in 0..=7u32 {
            let got: f64 = GL4_X
                .iter()
                .zip(GL4_W.iter())
                .map(|(&x, &w)| w * x.powi(k as i32))
                .sum();
            let want = if k % 2 == 0 {
                2.0 / (k as f64 + 1.0)
            } else {
                0.0
            };
            assert!(
                (got - want).abs() < 1e-15,
                "GL4 not exact for x^{k}: got {got:.17e}, want {want:.17e}"
            );
        }
    }

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
    fn erfc_relative_accuracy_below_the_asymptotic_split() {
        // The erf series this replaces was subtracted from 1, costing five digits near x = 4.
        #[allow(clippy::excessive_precision)]
        let refs = [
            (0.5_f64, 0.479500122186953462_f64),
            (1.5, 0.0338948535246892729),
            (2.5, 0.00040695201744495894),
            (3.0, 0.0000220904969985854414),
            (3.5, 7.43098372341412746e-7),
            (3.9, 3.47922485972317423e-8),
            (3.999, 1.55447494909950056e-8),
            (4.5, 1.96616044154288748e-10),
            (6.0, 2.15197367124989131e-17),
            (20.0, 5.39586561160790086e-176),
        ];
        for (x, want) in refs {
            let got = erfc_real(x);
            let rel = (got - want).abs() / want;
            assert!(
                rel < 1e-11,
                "erfc({x}): got {got:.17e}, want {want:.17e}, rel {rel:.2e}"
            );
            // erfc(−x) = 2 − erfc(x)
            assert!((erfc_real(-x) - (2.0 - got)).abs() <= 4.0 * f64::EPSILON);
        }
    }

    #[test]
    fn a_ratio_keeps_the_sign_past_the_gamma_pole() {
        // A_n(μ) = 2^{−n} Γ(μ+½+n)/(n! Γ(μ+½−n)); for μ = 0 the first ratio is negative
        // because Γ(−½) < 0, which lnΓ cannot express.
        assert!(
            (a_ratio(1, 0.0) - -0.125).abs() < 1e-15,
            "{}",
            a_ratio(1, 0.0)
        );
        assert!((a_ratio(1, 19.0) - 180.375).abs() < 1e-12);
        // μ = n − ½ sits on the pole of Γ(μ+½−n): A_n and every later term vanish.
        assert_eq!(a_ratio(1, 0.5), 0.0);
        assert_eq!(a_ratio(3, 2.5), 0.0);
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
