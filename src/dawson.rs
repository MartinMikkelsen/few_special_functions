//! Dawson's integral D(x) = e^{-x²} ∫₀ˣ e^{t²} dt.
//!
//! Adapted from the algorithms in:
//! M. R. Zaghloul, *Numerical Algorithms* **95**, 1291-1308 (2023).
//! <https://doi.org/10.1007/s11075-023-01608-8>
//! Source: <https://github.com/mofrehzaghloul/Dawson> (MIT)
//!
//! This is a faithful `f64` port of the Julia implementation in
//! `FewSpecialFunctions.jl` (`src/Dawson.jl`).

/// Maximum number of continued-fraction / series iterations before giving up.
const DAWSON_MAXITER: usize = 10_000;

/// Convergence tolerance: 4·eps.
fn dawson_tolerance() -> f64 {
    4.0 * f64::EPSILON
}

/// Upper bound of the "small" region evaluated with the continued fraction.
fn dawson_small_start() -> f64 {
    f64::EPSILON.sqrt()
}

/// Upper bound of the "middle" region evaluated with the Maclaurin series.
fn dawson_large_start() -> f64 {
    8.0_f64.max((2.0 * (1.0 / f64::EPSILON).ln()).sqrt())
}

/// Threshold at or above which the leading asymptotic term 1/(2x) suffices.
fn dawson_asymptotic_start() -> f64 {
    1.0 / f64::EPSILON.sqrt()
}

/// One order-`m` approximant of the small-argument continued fraction.
fn dawson_small_cf_order(m: usize, x: f64) -> f64 {
    let x2 = x * x;
    let sign = if m.is_multiple_of(2) { -1.0 } else { 1.0 };
    let mut y = sign * (2 * m) as f64 * x2 / (4 * m * m - 1) as f64;
    for k in (1..m).rev() {
        let sign = if k.is_multiple_of(2) { -1.0 } else { 1.0 };
        y = sign * (2 * k) as f64 * x2 / ((4 * k * k - 1) as f64 * (1.0 + y));
    }
    x / (1.0 + y)
}

/// Small-argument continued fraction, converged over the approximant order `m`.
pub(crate) fn dawson_small_cf(x: f64) -> f64 {
    let tol = dawson_tolerance();
    let mut previous = dawson_small_cf_order(1, x);
    for m in 2..=DAWSON_MAXITER {
        let current = dawson_small_cf_order(m, x);
        if (current - previous).abs() <= tol * 1.0_f64.max(current.abs()) {
            return current;
        }
        previous = current;
    }
    panic!("Dawson small continued fraction failed to converge for x = {x}");
}

/// Middle-region Maclaurin series D(x) = x·e^{-x²}·Σ … .
pub(crate) fn dawson_middle_series(x: f64) -> f64 {
    let x2 = x * x;
    let tol = dawson_tolerance();
    let mut sum = 1.0;
    let mut term = 1.0;
    for n in 0..DAWSON_MAXITER {
        term *= x2 * (2 * n + 1) as f64 / ((2 * n + 3) * (n + 1)) as f64;
        sum += term;
        if term.abs() <= tol * sum.abs() {
            return x * (-x2).exp() * sum;
        }
    }
    panic!("Dawson middle series failed to converge for x = {x}");
}

/// One order-`m` approximant of the large-argument continued fraction.
fn dawson_large_cf_order(m: usize, x: f64) -> f64 {
    let half = 0.5_f64;
    let mut y = m as f64 / x;
    for k in (1..m).rev() {
        y = k as f64 / (-half).mul_add(y, x);
    }
    1.0 / 2.0_f64.mul_add(x, -y)
}

/// Large-argument continued fraction, converged over the approximant order `m`.
pub(crate) fn dawson_large_cf(x: f64) -> f64 {
    let tol = dawson_tolerance();
    let mut previous = dawson_large_cf_order(1, x);
    for m in 2..=DAWSON_MAXITER {
        let current = dawson_large_cf_order(m, x);
        let scale = current.abs();
        if scale == 0.0 {
            if (current - previous).abs() <= tol {
                return current;
            }
        } else if (current - previous).abs() <= tol * scale {
            return current;
        }
        previous = current;
    }
    panic!("Dawson large continued fraction failed to converge for x = {x}");
}

/// Dawson's integral
///
/// ```text
/// D(x) = e^{-x²} ∫₀ˣ e^{t²} dt
/// ```
///
/// evaluated by piecewise selection between a small-argument continued
/// fraction, a middle-region Maclaurin series, a large-argument continued
/// fraction, and the leading asymptotic term 1/(2x), following Zaghloul (2023).
///
/// `D` is an odd function; signed zeros, `NaN`, and `±∞` are propagated exactly
/// as in the Julia reference (`D(±∞) = ±0`).
///
/// # Examples
///
/// ```
/// use few_special_functions::dawson::dawson;
///
/// // D(1) ≈ 0.5380795069127684 (maximum of the integral)
/// assert!((dawson(1.0) - 0.5380795069127684).abs() < 1e-14);
/// assert_eq!(dawson(0.0), 0.0);
/// assert_eq!(dawson(-2.0), -dawson(2.0));
/// ```
pub fn dawson(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == 0.0 {
        return x; // preserves the sign of a signed zero
    }
    if x.is_infinite() {
        return 1.0 / (x + x); // D(±∞) = ±0
    }

    let ax = x.abs();
    let value = if ax <= dawson_small_start() {
        dawson_small_cf(ax)
    } else if ax <= dawson_large_start() {
        dawson_middle_series(ax)
    } else if ax >= dawson_asymptotic_start() {
        1.0 / (ax + ax)
    } else {
        dawson_large_cf(ax)
    };
    value.copysign(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    // These exercise the individual numerical regions directly, mirroring the
    // "Dawson numerical regions" testset in the Julia suite.
    #[test]
    fn small_cf_region() {
        assert!((dawson_small_cf(0.1) - 0.09933599239785286).abs() <= 2e-14 * 0.1);
    }

    #[test]
    fn middle_series_region() {
        assert!((dawson_middle_series(2.0) - 0.30134038892379197).abs() <= 2e-14 * 0.30134);
    }

    #[test]
    fn large_cf_region() {
        assert!((dawson_large_cf(20.0) - 0.02503136792640367).abs() <= 2e-14 * 0.0250314);
    }
}
