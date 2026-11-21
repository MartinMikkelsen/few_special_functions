//! Real Coulomb wave functions `F` and `G` for `ell > -1`, real `eta` and
//! `rho > 0`.
//!
//! Outside the turning point this is Steed's method: the continued fraction
//! CF1 gives `F'/F`, the complex continued fraction CF2 gives `H+'/H+ = p + iq`
//! with `H+ = G + iF`, and the Wronskian `F'G - FG' = 1` fixes the
//! normalization (DLMF sections 33.2, 33.4 and 33.8; A. R. Barnett, Comput. Phys.
//! Commun. 27, 147 (1982); I. J. Thompson and A. R. Barnett, J. Comput. Phys.
//! 64, 490 (1986)). Inside the turning point CF2 determines `q` only to an
//! absolute precision while `q` itself becomes tiny, so `G` and `G'` are
//! taken from CF2 at the switching radius and integrated inward with a Taylor
//! series of the Coulomb equation; `G` grows inward, so the integration is
//! stable. `F` then follows from CF1 at `rho` and the Wronskian.
//!
//! Written from the published formulas; no code was ported.

use num_complex::Complex64;

/// `F`, `G` and `G'` at one point.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Waves {
    pub f: f64,
    pub g: f64,
    pub dg: f64,
}

/// Whether [`waves`] covers these arguments.
pub(crate) fn in_domain(ell: f64, eta: f64, rho: f64) -> bool {
    ell > -1.0 && ell.is_finite() && eta.is_finite() && rho > 0.0 && rho.is_finite()
}

/// `F`, `G` and `G'`; requires [`in_domain`].
pub(crate) fn waves(ell: f64, eta: f64, rho: f64) -> Waves {
    debug_assert!(in_domain(ell, eta, rho));
    // Steed's method loses accuracy inside the turning point and, through slow
    // CF2 convergence, at small rho; below `switch` it is used only to start
    // the inward integration.
    let turning_point = eta + (eta * eta + ell * (ell + 1.0)).max(0.0).sqrt();
    let switch = turning_point.max(1.0);
    if rho >= switch {
        return steed(ell, eta, rho);
    }

    let start = steed(ell, eta, switch);
    let (mut g, mut dg) = (start.g, start.dg);
    let mut position = switch;
    while position > rho {
        // Keep every Taylor term within a modest factor of the result: near the
        // singularity at 0 the solution behaves like rho^-ell, whose series at
        // step s·position grows like binomial(k + ell, k) s^k, and elsewhere
        // like exp(sqrt|Q| step) with Q the coefficient of the equation.
        let q = (ell * (ell + 1.0) / (position * position) + 2.0 * eta / position - 1.0).abs();
        let step = (position - rho)
            .min(position / (2.0 + ell.max(0.0)))
            .min(2.0 / q.max(1.0).sqrt());
        (g, dg) = taylor_step(ell, eta, position, g, dg, -step);
        position = if step == position - rho {
            rho
        } else {
            position - step
        };
    }
    let (ratio, _) = cf1(ell, eta, rho);
    let f = 1.0 / (ratio * g - dg);
    Waves { f, g, dg }
}

/// Steed's method at `rho`.
fn steed(ell: f64, eta: f64, rho: f64) -> Waves {
    let (ratio, sign) = cf1(ell, eta, rho);
    let pq = cf2(ell, eta, rho);
    let (p, q) = (pq.re, pq.im);
    // G' + iF' = (p + iq)(G + iF) and F' = ratio F give G = gamma F, and the
    // Wronskian F'G - FG' = 1 gives F² ((ratio - p) gamma + q) = 1.
    let gamma = (ratio - p) / q;
    let f = sign / ((ratio - p) * gamma + q).sqrt();
    let g = gamma * f;
    Waves {
        f,
        g,
        dg: p * g - q * f,
    }
}

/// CF1: `F'/F` at order `ell`, and the sign of `F`.
///
/// With `S_k = k/rho + eta/k`, `R_k² = 1 + eta²/k²` and `T_k = S_k + S_{k+1}`,
/// `F'/F = S_{ell+1} - R_{ell+1}² / (T_{ell+1} - R_{ell+2}² / (T_{ell+2} - …))`.
/// The fraction is evaluated from a truncation depth `n` upward. Its partial
/// tails `t_k` satisfy `F_{ell+k}/F_{ell+k-1} = R_{ell+k}/t_k`, and `F` is
/// positive at high order, so the sign of `F` is the product of the signs of
/// the tails. The depth is doubled until the result stops changing.
fn cf1(ell: f64, eta: f64, rho: f64) -> (f64, f64) {
    let s = |k: f64| k / rho + eta / k;
    let r2 = |k: f64| 1.0 + (eta / k) * (eta / k);
    let evaluate = |depth: usize| {
        let mut tail = s(ell + depth as f64) + s(ell + depth as f64 + 1.0);
        let mut sign = 1.0;
        for k in (1..depth).rev() {
            let k = ell + k as f64;
            tail = s(k) + s(k + 1.0) - r2(k + 1.0) / tail;
            if tail < 0.0 {
                sign = -sign;
            }
        }
        (s(ell + 1.0) - r2(ell + 1.0) / tail, sign)
    };

    let mut depth = (rho + eta.abs()) as usize + 40;
    let mut previous = evaluate(depth);
    loop {
        depth *= 2;
        let next = evaluate(depth);
        if (next.0 - previous.0).abs() <= 4.0 * f64::EPSILON * next.0.abs() {
            return next;
        }
        assert!(
            depth < 1 << 26,
            "Coulomb CF1 did not converge (ell = {ell}, eta = {eta}, rho = {rho})"
        );
        previous = next;
    }
}

/// CF2: `H+'/H+ = p + iq`.
///
/// `p + iq = i (1 - eta/rho) + (i/rho) a c / X` with
/// `X = b_1 + (a+1)(c+1) / (b_2 + (a+2)(c+2) / (b_3 + …))`,
/// `a = i eta - ell`, `c = i eta + ell + 1` and `b_k = 2 (rho - eta + i k)`.
/// `X` is evaluated by the modified Lentz method; its leading term `b_1` never
/// vanishes, so no artificial starting value is needed.
fn cf2(ell: f64, eta: f64, rho: f64) -> Complex64 {
    const MAX_TERMS: usize = 1_000_000;
    // Replaces an exactly zero Lentz denominator; its square must not underflow.
    const SMALL: f64 = 1e-150;
    let a = Complex64::new(-ell, eta);
    let c = Complex64::new(ell + 1.0, eta);
    let b = |k: usize| Complex64::new(2.0 * (rho - eta), 2.0 * k as f64);
    let leading = Complex64::new(0.0, 1.0 - eta / rho);
    let numerator = a * c;
    if numerator == Complex64::new(0.0, 0.0) {
        return leading;
    }

    let mut x = b(1);
    let mut lentz_c = x;
    let mut lentz_d = Complex64::new(0.0, 0.0);
    for k in 2..=MAX_TERMS {
        let shift = (k - 1) as f64;
        let partial_numerator = (a + shift) * (c + shift);
        lentz_d = b(k) + partial_numerator * lentz_d;
        if lentz_d == Complex64::new(0.0, 0.0) {
            lentz_d = Complex64::new(SMALL, 0.0);
        }
        lentz_c = b(k) + partial_numerator / lentz_c;
        if lentz_c == Complex64::new(0.0, 0.0) {
            lentz_c = Complex64::new(SMALL, 0.0);
        }
        lentz_d = 1.0 / lentz_d;
        let delta = lentz_c * lentz_d;
        x *= delta;
        if (delta - 1.0).norm() <= 2.0 * f64::EPSILON {
            return leading + Complex64::new(0.0, 1.0 / rho) * numerator / x;
        }
    }
    panic!("Coulomb CF2 did not converge (ell = {ell}, eta = {eta}, rho = {rho})")
}

/// One Taylor-series step of `u'' = (ell(ell+1)/rho² + 2 eta/rho - 1) u` from
/// `center` to `center + step`, returning `(u, u')`.
///
/// Multiplied by `rho²`, the equation has polynomial coefficients, so with
/// `u = Σ c_k x^k`, `x = rho - center`, the coefficients obey
/// `center² (k+2)(k+1) c_{k+2} = A c_k + B c_{k-1} - c_{k-2}
///  - 2 center (k+1) k c_{k+1} - k(k-1) c_k`
/// with `A = ell(ell+1) + 2 eta center - center²` and `B = 2 eta - 2 center`.
/// The series converges for `|step| < center`; callers keep `|step| <= center/2`.
fn taylor_step(ell: f64, eta: f64, center: f64, u: f64, du: f64, step: f64) -> (f64, f64) {
    const MAX_TERMS: usize = 2000;
    let a = ell * (ell + 1.0) + 2.0 * eta * center - center * center;
    let b = 2.0 * eta - 2.0 * center;
    let scale = center * center;
    // Coefficients c_{k-2}, c_{k-1}, c_k, c_{k+1} in a sliding window.
    let (mut c_km2, mut c_km1, mut c_k, mut c_kp1) = (0.0, 0.0, u, du);
    let mut value = u + du * step;
    let mut derivative = du;
    let mut power = step; // step^(k+1)
    let mut quiet = 0;
    for k in 0..MAX_TERMS {
        let kf = k as f64;
        let next = (a * c_k + b * c_km1
            - c_km2
            - 2.0 * center * (kf + 1.0) * kf * c_kp1
            - kf * (kf - 1.0) * c_k)
            / (scale * (kf + 2.0) * (kf + 1.0));
        let derivative_term = (kf + 2.0) * next * power;
        power *= step;
        let value_term = next * power;
        value += value_term;
        derivative += derivative_term;
        if value_term.abs() <= 0.25 * f64::EPSILON * value.abs()
            && derivative_term.abs() <= 0.25 * f64::EPSILON * derivative.abs()
        {
            quiet += 1;
            if quiet == 2 {
                return (value, derivative);
            }
        } else {
            quiet = 0;
        }
        (c_km2, c_km1, c_k, c_kp1) = (c_km1, c_k, c_kp1, next);
    }
    panic!("Coulomb Taylor series did not converge (ell = {ell}, eta = {eta}, rho = {center})")
}
