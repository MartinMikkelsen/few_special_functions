//! Internal numerical building blocks.
//!
//! Written for this crate from published formulas; no code was ported.
//!
//! - complex log-gamma: the Lanczos approximation with the common g = 7,
//!   nine-coefficient set, and the reflection formula (DLMF section 5.5);
//! - digamma: recurrence and reflection (DLMF section 5.5) and the asymptotic
//!   expansion (DLMF section 5.11);
//! - Kummer `M`, regularized `M` and Tricomi `U`: the power series, Kummer's
//!   transformation and the connection formula for `U` (DLMF section 13.2);
//! - Riemann zeta for real `s`: Euler–Maclaurin summation (DLMF section 25.2)
//!   and the functional equation (DLMF section 25.4);
//! - upper incomplete gamma: DLMF 8.7.1 and 8.9.2;
//! - adaptive 15-point Gauss–Kronrod quadrature.

use crate::SpecialError;
use crate::error::iteration_cap;
use num_complex::Complex64;
use std::f64::consts::PI;

const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

fn sin_pi_complex(z: Complex64) -> Complex64 {
    let nearest = z.re.round();
    let reduced = z.re - nearest;
    let parity = if nearest.rem_euclid(2.0) == 0.0 {
        1.0
    } else {
        -1.0
    };
    Complex64::new(
        parity * (PI * reduced).sin() * (PI * z.im).cosh(),
        parity * (PI * reduced).cos() * (PI * z.im).sinh(),
    )
}

pub(crate) fn log_gamma_complex(z: Complex64) -> Complex64 {
    if z.re < 0.5 {
        return Complex64::new(PI.ln(), 0.0)
            - sin_pi_complex(z).ln()
            - log_gamma_complex(Complex64::new(1.0, 0.0) - z);
    }

    let shifted = z - Complex64::new(1.0, 0.0);
    let mut series = Complex64::new(LANCZOS_COEFFICIENTS[0], 0.0);
    for (index, coefficient) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        series += coefficient / (shifted + Complex64::new(index as f64, 0.0));
    }
    let t = shifted + Complex64::new(7.5, 0.0);
    Complex64::new(0.5 * (2.0 * PI).ln(), 0.0) + (shifted + Complex64::new(0.5, 0.0)) * t.ln() - t
        + series.ln()
}

pub(crate) fn gamma_complex(z: Complex64) -> Complex64 {
    log_gamma_complex(z).exp()
}

pub(crate) fn digamma_complex(z: Complex64) -> Complex64 {
    if z.re < 0.5 {
        let pi_z = Complex64::new(PI, 0.0) * z;
        return digamma_complex(Complex64::new(1.0, 0.0) - z)
            - Complex64::new(PI, 0.0) * pi_z.cos() / pi_z.sin();
    }

    let mut value = Complex64::new(0.0, 0.0);
    let mut w = z;
    while w.re < 12.0 {
        value -= Complex64::new(1.0, 0.0) / w;
        w += Complex64::new(1.0, 0.0);
    }

    let inverse = Complex64::new(1.0, 0.0) / w;
    let inverse_squared = inverse * inverse;
    value + w.ln()
        - 0.5 * inverse
        - inverse_squared
            * (1.0 / 12.0
                + inverse_squared
                    * (-1.0 / 120.0
                        + inverse_squared
                            * (1.0 / 252.0
                                + inverse_squared
                                    * (-1.0 / 240.0
                                        + inverse_squared
                                            * (1.0 / 132.0
                                                + inverse_squared * (-691.0 / 32_760.0))))))
}

/// Cap on the terms of the Kummer series.
pub(crate) const MAX_KUMMER_TERMS: usize = 100_000;

/// Whether every component of the arguments is finite.
fn all_finite(values: &[Complex64]) -> bool {
    values
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
}

const COMPLEX_NAN: Complex64 = Complex64::new(f64::NAN, f64::NAN);

/// Kummer's function `M(a, b, z)` by its power series (DLMF 13.2.2), after
/// Kummer's transformation (DLMF 13.2.39) when `Re z < 0`. The series stops
/// when a term falls below `4 ε max(1, |sum|)`; reaching
/// [`MAX_KUMMER_TERMS`] reports [`SpecialError::NoConvergence`] for `family`.
/// Arguments that are not finite give NaN.
pub(crate) fn kummer_m(
    a: Complex64,
    b: Complex64,
    z: Complex64,
    family: &'static str,
) -> Result<Complex64, SpecialError> {
    if !all_finite(&[a, b, z]) {
        return Ok(COMPLEX_NAN);
    }
    if z.re < 0.0 {
        return Ok(z.exp() * kummer_m_series(b - a, b, -z, family)?);
    }
    kummer_m_series(a, b, z, family)
}

fn kummer_m_series(
    a: Complex64,
    b: Complex64,
    z: Complex64,
    family: &'static str,
) -> Result<Complex64, SpecialError> {
    let mut sum = Complex64::new(1.0, 0.0);
    let mut compensation = Complex64::new(0.0, 0.0);
    let mut term = Complex64::new(1.0, 0.0);

    let max_terms = iteration_cap(MAX_KUMMER_TERMS);
    for index in 1..=max_terms {
        let previous = (index - 1) as f64;
        term *= (a + previous) * z / ((b + previous) * index as f64);

        let corrected = term - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
        if term.norm() <= 4.0 * f64::EPSILON * sum.norm().max(1.0) {
            return Ok(sum);
        }
    }
    Err(SpecialError::no_convergence(family, max_terms))
}

pub(crate) fn reciprocal_gamma(z: Complex64) -> Complex64 {
    if z.im == 0.0 && z.re <= 0.0 && z.re == z.re.round() {
        Complex64::new(0.0, 0.0)
    } else if z.re < 0.5 {
        sin_pi_complex(z) * gamma_complex(1.0 - z) / PI
    } else {
        (-log_gamma_complex(z)).exp()
    }
}

fn cos_pi_complex(z: Complex64) -> Complex64 {
    let nearest = z.re.round();
    let reduced = z.re - nearest;
    let parity = if nearest.rem_euclid(2.0) == 0.0 {
        1.0
    } else {
        -1.0
    };
    Complex64::new(
        parity * (PI * reduced).cos() * (PI * z.im).cosh(),
        -parity * (PI * reduced).sin() * (PI * z.im).sinh(),
    )
}

/// `1/Γ(z)` and its derivative `−ψ(z)/Γ(z)`. Both are entire; for `Re z < 1/2`
/// they come from the reflection formula `1/Γ(z) = Γ(1 − z) sin(πz)/π`, so the
/// derivative is finite at the poles of Γ, where it equals `(−1)^k k!` at
/// `z = −k`.
fn reciprocal_gamma_with_derivative(z: Complex64) -> (Complex64, Complex64) {
    if z.re < 0.5 {
        let one_minus_z = Complex64::new(1.0, 0.0) - z;
        let gamma_reflected = gamma_complex(one_minus_z);
        let sin = sin_pi_complex(z);
        (
            sin * gamma_reflected / PI,
            gamma_reflected * (cos_pi_complex(z) - digamma_complex(one_minus_z) * sin / PI),
        )
    } else {
        let value = (-log_gamma_complex(z)).exp();
        (value, -digamma_complex(z) * value)
    }
}

/// Whether the series of [`kummer_m_regularized`] may stop at `index`: past
/// the leading terms, and past the zeros of `1/Γ(b + n)` at a nonpositive
/// integer `b`, which would otherwise end the sum at a run of zero terms.
fn regularized_series_may_stop(index: usize, b: Complex64) -> bool {
    index >= 8 && b.re + index as f64 > 0.0
}

/// A Kahan compensated sum that also records the largest term added.
struct CompensatedSum {
    sum: Complex64,
    compensation: Complex64,
    peak: f64,
}

impl CompensatedSum {
    fn new(start: Complex64) -> Self {
        Self {
            sum: start,
            compensation: Complex64::new(0.0, 0.0),
            peak: start.norm(),
        }
    }

    fn add(&mut self, term: Complex64) {
        let corrected = term - self.compensation;
        let next = self.sum + corrected;
        self.compensation = (next - self.sum) - corrected;
        self.sum = next;
        self.peak = self.peak.max(term.norm());
    }

    /// Whether `term` is below `4 ε max(|sum|, ε peak)`.
    ///
    /// The bound is relative: a regularized sum can be as small as `1/Γ(b)`,
    /// so an absolute floor would stop it early. The `ε peak` floor ends a
    /// sum that cancels to zero.
    fn negligible(&self, term: Complex64) -> bool {
        term.norm() <= 4.0 * f64::EPSILON * self.sum.norm().max(f64::EPSILON * self.peak)
    }
}

/// `M(a, b, z) / Γ(b)` by its power series (DLMF 13.2.4), with the same
/// transformation, cap and treatment of non-finite arguments as
/// [`kummer_m`]. The series stops when a term falls below
/// `4 ε max(|sum|, ε max_n |term_n|)`.
pub(crate) fn kummer_m_regularized(
    a: Complex64,
    b: Complex64,
    z: Complex64,
    family: &'static str,
) -> Result<Complex64, SpecialError> {
    if !all_finite(&[a, b, z]) {
        return Ok(COMPLEX_NAN);
    }
    if z.re < 0.0 {
        return Ok(z.exp() * kummer_m_regularized(b - a, b, -z, family)?);
    }

    let mut coefficient = Complex64::new(1.0, 0.0);
    let mut sum = CompensatedSum::new(reciprocal_gamma(b));
    let max_terms = iteration_cap(MAX_KUMMER_TERMS);
    for index in 1..=max_terms {
        let previous = (index - 1) as f64;
        coefficient *= (a + previous) * z / index as f64;
        let term = coefficient * reciprocal_gamma(b + index as f64);
        sum.add(term);
        if regularized_series_may_stop(index, b) && sum.negligible(term) {
            return Ok(sum.sum);
        }
    }
    Err(SpecialError::no_convergence(family, max_terms))
}

/// `M(a, b, z) / Γ(b)` and its partial derivatives in `a` and `b`, as
/// `[value, ∂/∂a, ∂/∂b]`.
///
/// The series of [`kummer_m_regularized`] is differentiated term by term:
/// with `c_n = (a)_n zⁿ / n!`, the terms are `c_n / Γ(b + n)`,
/// `(∂c_n/∂a) / Γ(b + n)` and `c_n (1/Γ)′(b + n)`. `∂c_n/∂a` follows the
/// recurrence of `c_n`, so it stays finite where `(a)_n` vanishes. For
/// `Re z < 0`, Kummer's transformation `M(a, b, z)/Γ(b) = e^z M(b − a, b, −z)/Γ(b)`
/// is applied first and the chain rule maps its derivatives back. The series
/// stops when each of the three terms meets the stopping rule of
/// [`kummer_m_regularized`] for its own sum, with the same cap and treatment
/// of non-finite arguments as [`kummer_m`].
pub(crate) fn kummer_m_regularized_with_parameter_derivatives(
    a: Complex64,
    b: Complex64,
    z: Complex64,
    family: &'static str,
) -> Result<[Complex64; 3], SpecialError> {
    if !all_finite(&[a, b, z]) {
        return Ok([COMPLEX_NAN; 3]);
    }
    if z.re < 0.0 {
        let [value, d_first, d_b] =
            kummer_m_regularized_with_parameter_derivatives(b - a, b, -z, family)?;
        let exp_z = z.exp();
        return Ok([exp_z * value, -exp_z * d_first, exp_z * (d_first + d_b)]);
    }

    let mut coefficient = Complex64::new(1.0, 0.0);
    let mut coefficient_d_a = Complex64::new(0.0, 0.0);
    let (reciprocal, reciprocal_derivative) = reciprocal_gamma_with_derivative(b);
    let mut value = CompensatedSum::new(reciprocal);
    let mut d_a = CompensatedSum::new(Complex64::new(0.0, 0.0));
    let mut d_b = CompensatedSum::new(reciprocal_derivative);
    let max_terms = iteration_cap(MAX_KUMMER_TERMS);
    for index in 1..=max_terms {
        let factor = z / index as f64;
        let a_previous = a + (index - 1) as f64;
        coefficient_d_a = (coefficient_d_a * a_previous + coefficient) * factor;
        coefficient *= a_previous * factor;
        let (reciprocal, reciprocal_derivative) =
            reciprocal_gamma_with_derivative(b + index as f64);
        let terms = [
            coefficient * reciprocal,
            coefficient_d_a * reciprocal,
            coefficient * reciprocal_derivative,
        ];
        value.add(terms[0]);
        d_a.add(terms[1]);
        d_b.add(terms[2]);
        if regularized_series_may_stop(index, b)
            && value.negligible(terms[0])
            && d_a.negligible(terms[1])
            && d_b.negligible(terms[2])
        {
            return Ok([value.sum, d_a.sum, d_b.sum]);
        }
    }
    Err(SpecialError::no_convergence(family, max_terms))
}

/// Tricomi's function `U(a, b, z)` from Kummer's function (DLMF 13.2.42).
///
/// The connection formula divides by `sin(πb)`, so its relative error grows
/// like `ε / |sin(πb)|`. Where `|sin(πb)| < 1e-7` it is replaced by the mean
/// of the values at `b ± 1e-5`, whose relative error is of order `1e-10`.
/// Arguments that are not finite give NaN.
pub(crate) fn tricomi_u(
    a: Complex64,
    b: Complex64,
    z: Complex64,
    family: &'static str,
) -> Result<Complex64, SpecialError> {
    if !all_finite(&[a, b, z]) {
        return Ok(COMPLEX_NAN);
    }
    let sin_pi_b = (Complex64::new(PI, 0.0) * b).sin();
    if sin_pi_b.norm() < 1e-7 {
        let step = Complex64::new(1e-5, 0.0);
        return Ok(0.5 * (tricomi_u(a, b - step, z, family)? + tricomi_u(a, b + step, z, family)?));
    }

    Ok(Complex64::new(PI, 0.0) / sin_pi_b
        * (kummer_m(a, b, z, family)?
            * reciprocal_gamma(Complex64::new(1.0, 0.0) + a - b)
            * reciprocal_gamma(b)
            - z.powc(Complex64::new(1.0, 0.0) - b)
                * kummer_m(
                    Complex64::new(1.0, 0.0) + a - b,
                    Complex64::new(2.0, 0.0) - b,
                    z,
                    family,
                )?
                * reciprocal_gamma(a)
                * reciprocal_gamma(Complex64::new(2.0, 0.0) - b)))
}

/// Riemann zeta function for real `s`; NaN at the pole `s = 1`.
pub(crate) fn zeta_real(s: f64) -> f64 {
    if s == 1.0 {
        return f64::NAN;
    }
    if s < 0.0 && s == s.round() && (s as i64) % 2 == 0 {
        return 0.0;
    }
    if s < 0.0 {
        return 2.0_f64.powf(s)
            * PI.powf(s - 1.0)
            * (0.5 * PI * s).sin()
            * gamma_complex(Complex64::new(1.0 - s, 0.0)).re
            * zeta_real(1.0 - s);
    }

    const TAIL_COEFFICIENTS: [f64; 10] = [
        1.0 / 12.0,
        -1.0 / 720.0,
        1.0 / 30_240.0,
        -1.0 / 1_209_600.0,
        1.0 / 47_900_160.0,
        -691.0 / 1_307_674_368_000.0,
        1.0 / 74_724_249_600.0,
        -3_617.0 / 10_670_622_842_880_000.0,
        43_867.0 / 510_909_421_717_094_400.0,
        -174_611.0 / 802_857_662_698_291_200_000.0,
    ];

    let cutoff = 32_u32;
    let mut sum = (1..cutoff).map(|n| (n as f64).powf(-s)).sum::<f64>();
    let cutoff = cutoff as f64;
    sum += cutoff.powf(1.0 - s) / (s - 1.0) + 0.5 * cutoff.powf(-s);

    let mut rising = s;
    for (index, coefficient) in TAIL_COEFFICIENTS.iter().enumerate() {
        if index > 0 {
            let order = 2.0 * index as f64;
            rising *= (s + order - 1.0) * (s + order);
        }
        sum += coefficient * rising * cutoff.powf(-s - 2.0 * index as f64 - 1.0);
    }
    sum
}

/// Regularized upper incomplete gamma function `Q(a, x) = Γ(a, x) / Γ(a)`
/// for `a > 0` and `x >= 0`.
///
/// For `x < a + 1` this is `1 - P(a, x)` with the power series
/// `P(a, x) = x^a e^{-x} Σ_{k≥0} x^k / Γ(a + k + 1)` (DLMF 8.7.1). Otherwise
/// it is Legendre's continued fraction (DLMF 8.9.2),
/// `Γ(a, x) = x^a e^{-x} / (x + (1 - a)/(1 + 1/(x + (2 - a)/(1 + 2/(x + …)))))`,
/// which keeps full relative accuracy far into the tail.
pub(crate) fn inc_gamma_q(a: f64, x: f64, family: &'static str) -> Result<f64, SpecialError> {
    debug_assert!(a > 0.0 && x >= 0.0);
    if x == 0.0 {
        return Ok(1.0);
    }
    let log_prefactor = a * x.ln() - x - libm::lgamma(a);
    if x < a + 1.0 {
        // Γ(a) Σ x^k / Γ(a + k + 1) = Σ x^k / (a (a + 1) ⋯ (a + k)); every
        // ratio x / (a + k) with k ≥ 1 is below one.
        let mut term = 1.0 / a;
        let mut sum = term;
        let mut k = 0.0;
        while term > 0.5 * f64::EPSILON * sum {
            k += 1.0;
            term *= x / (a + k);
            sum += term;
        }
        Ok(1.0 - log_prefactor.exp() * sum)
    } else {
        Ok(log_prefactor.exp() / legendre_fraction(a, x, family)?)
    }
}

/// Denominator `x + (1 - a)/(1 + 1/(x + (2 - a)/(1 + 2/(x + …))))` of DLMF 8.9.2
/// for `x >= a + 1`.
///
/// With `T_k = x + (k - a) / (1 + k / T_{k+1})` the denominator is `T_1`.
/// The recursion runs downward from `T_{n+1} = x`, and the depth `n` is doubled
/// until two depths agree, up to [`MAX_FRACTION_DEPTH`]. Every `T_k` exceeds
/// `x - a + 1 >= 2`, so no division is by zero.
fn legendre_fraction(a: f64, x: f64, family: &'static str) -> Result<f64, SpecialError> {
    let max_depth = iteration_cap(MAX_FRACTION_DEPTH);
    let evaluate = |depth: usize| {
        let mut tail = x;
        for k in (1..=depth).rev() {
            let k = k as f64;
            tail = x + (k - a) / (1.0 + k / tail);
        }
        tail
    };
    let mut depth = 16;
    let mut previous = evaluate(depth);
    loop {
        if depth >= max_depth {
            return Err(SpecialError::no_convergence(family, max_depth));
        }
        depth *= 2;
        let next = evaluate(depth);
        if (next - previous).abs() <= 4.0 * f64::EPSILON * next.abs() {
            return Ok(next);
        }
        previous = next;
    }
}

/// Cap on the depth of the incomplete-gamma continued fraction.
const MAX_FRACTION_DEPTH: usize = 1 << 24;

const KRONROD_NODES: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_5,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_48,
    0.0,
];
const KRONROD_WEIGHTS: [f64; 8] = [
    0.022_935_322_010_529_224,
    0.063_092_092_629_978_56,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_42,
    0.204_432_940_075_298_89,
    0.209_482_141_084_727_82,
];
const GAUSS_WEIGHTS: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_64,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

fn gauss_kronrod_15(function: &impl Fn(f64) -> f64, lower: f64, upper: f64) -> (f64, f64) {
    let midpoint = 0.5 * (lower + upper);
    let half_width = 0.5 * (upper - lower);
    let midpoint_value = function(midpoint);
    let mut kronrod = KRONROD_WEIGHTS[7] * midpoint_value;
    let mut gauss = GAUSS_WEIGHTS[3] * midpoint_value;

    for index in 0..7 {
        let offset = half_width * KRONROD_NODES[index];
        let pair_sum = function(midpoint - offset) + function(midpoint + offset);
        kronrod += KRONROD_WEIGHTS[index] * pair_sum;
        if index % 2 == 1 {
            gauss += GAUSS_WEIGHTS[(index - 1) / 2] * pair_sum;
        }
    }

    let kronrod = half_width * kronrod;
    let gauss = half_width * gauss;
    (kronrod, (kronrod - gauss).abs())
}

/// `∫ function` over `[lower, upper]` by adaptive 15-point Gauss–Kronrod
/// quadrature to an absolute `tolerance`, with the error budget halved on each
/// bisection and a floor of `32 ε |estimate|` per subinterval. Visiting more
/// than `max_intervals` subintervals reports [`SpecialError::NoConvergence`]
/// for `family`.
pub(crate) fn integrate(
    function: impl Fn(f64) -> f64,
    lower: f64,
    upper: f64,
    tolerance: f64,
    max_intervals: usize,
    family: &'static str,
) -> Result<f64, SpecialError> {
    if lower == upper {
        return Ok(0.0);
    }
    if upper < lower {
        return integrate(function, upper, lower, tolerance, max_intervals, family)
            .map(|value| -value);
    }
    let max_intervals = iteration_cap(max_intervals);

    let mut total = 0.0;
    let mut intervals = vec![(lower, upper, tolerance)];
    let mut visited = 0;
    while let Some((left, right, local_tolerance)) = intervals.pop() {
        visited += 1;
        if visited > max_intervals {
            return Err(SpecialError::no_convergence(family, max_intervals));
        }
        let (estimate, error) = gauss_kronrod_15(&function, left, right);
        let roundoff_floor = 32.0 * f64::EPSILON * estimate.abs();
        if error <= local_tolerance.max(roundoff_floor) || left == (left + right) * 0.5 {
            total += estimate;
        } else {
            let midpoint = 0.5 * (left + right);
            intervals.push((midpoint, right, 0.5 * local_tolerance));
            intervals.push((left, midpoint, 0.5 * local_tolerance));
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn complex_gamma_family_matches_julia() {
        let z = Complex64::new(0.7, 0.4);
        let log_gamma = log_gamma_complex(z);
        assert!(
            (log_gamma - Complex64::new(0.057044748102826226, -0.42944066621229504)).norm() < 2e-14
        );
        let digamma = digamma_complex(z);
        assert!((digamma - Complex64::new(-0.8219871048796334, 0.9236223307792655)).norm() < 2e-14);
    }

    #[test]
    fn confluent_hypergeometric_functions_match_julia() {
        let a = Complex64::new(0.8, -0.3);
        let b = Complex64::new(1.7, 0.2);
        let z = Complex64::new(1.0, 2.0);
        let m = kummer_m(a, b, z, "test").unwrap();
        assert!((m - Complex64::new(1.3922407178086509, 1.6474537115932955)).norm() < 2e-13);
        let u = tricomi_u(a, b, z, "test").unwrap();
        assert!((u - Complex64::new(0.36857118978677744, -0.22472938298119105)).norm() < 2e-12);
    }

    #[test]
    fn regularized_kummer_at_negative_integer_b() {
        // mpmath: Σ_k (a)_k z^k / (k! Γ(b + k)) summed to k = 200 at 40 digits.
        let a = Complex64::new(1.0, 0.5);
        let cases = [
            (
                (0.0, -1.0),
                0.0,
                Complex64::new(-0.67793149901985677, -1.240945285117965),
            ),
            (
                (0.0, -1.0),
                3.0,
                Complex64::new(1.2992077336620862, 0.0080102013423026438),
            ),
            (
                (0.0, -1.0),
                7.0,
                Complex64::new(1.197529132517262, 0.39340393147269915),
            ),
            (
                (0.0, -1.0),
                8.0,
                Complex64::new(0.45684569628005396, -1.1694798349519364),
            ),
            (
                (0.0, -1.0),
                12.0,
                Complex64::new(0.64618404067662552, -1.0615267535985611),
            ),
            (
                (-1.5, 0.7),
                0.0,
                Complex64::new(-0.3087429430974647, -0.065430700590213218),
            ),
            (
                (-1.5, 0.7),
                3.0,
                Complex64::new(1.7606441487515656, -0.38825606056442416),
            ),
            (
                (-1.5, 0.7),
                7.0,
                Complex64::new(0.051152103412455235, -14.337086751555422),
            ),
            (
                (-1.5, 0.7),
                8.0,
                Complex64::new(8.597908788180188, 22.289036584516115),
            ),
            (
                (-1.5, 0.7),
                12.0,
                Complex64::new(171.59162354352644, -61.620063755826685),
            ),
        ];
        for ((re, im), n, expected) in cases {
            let (b, z) = (Complex64::new(-n, 0.0), Complex64::new(re, im));
            let actual = kummer_m_regularized(a, b, z, "test").unwrap();
            let [value, _, _] =
                kummer_m_regularized_with_parameter_derivatives(a, b, z, "test").unwrap();
            for v in [actual, value] {
                let error = (v - expected).norm() / expected.norm();
                assert!(error < 1e-14, "b = {b}, z = {z}: relative error {error:e}");
            }
        }
    }

    #[test]
    fn reciprocal_gamma_derivative_at_the_poles() {
        // (1/Γ)′(−k) = (−1)^k k!
        let mut factorial = 1.0;
        for k in 0..8 {
            if k > 0 {
                factorial *= k as f64;
            }
            let (value, derivative) =
                reciprocal_gamma_with_derivative(Complex64::new(-(k as f64), 0.0));
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(value, Complex64::new(0.0, 0.0));
            assert!(
                (derivative - sign * factorial).norm() < 1e-13 * factorial,
                "k = {k}"
            );
        }
    }

    #[test]
    fn regularized_kummer_parameter_derivatives_match_central_differences() {
        let step = 1e-5;
        let cases = [
            (
                Complex64::new(2.0, 0.5),
                Complex64::new(4.0, 0.0),
                Complex64::new(0.0, -4.0),
            ),
            (
                Complex64::new(-1.0, 0.5),
                Complex64::new(-2.0, 0.0),
                Complex64::new(0.0, -4.0),
            ),
            (
                Complex64::new(-3.0, 0.5),
                Complex64::new(-8.0, 0.0),
                Complex64::new(0.0, -2.0),
            ),
            (
                Complex64::new(0.8, -0.3),
                Complex64::new(1.7, 0.2),
                Complex64::new(-1.0, 2.0),
            ),
            (
                Complex64::new(0.5, 0.0),
                Complex64::new(-3.0, 0.0),
                Complex64::new(-2.5, -0.5),
            ),
        ];
        for (a, b, z) in cases {
            let m = |a, b| kummer_m_regularized(a, b, z, "test").unwrap();
            let [value, d_a, d_b] =
                kummer_m_regularized_with_parameter_derivatives(a, b, z, "test").unwrap();
            let expected_d_a = (m(a + step, b) - m(a - step, b)) / (2.0 * step);
            let expected_d_b = (m(a, b + step) - m(a, b - step)) / (2.0 * step);
            let scale = value.norm().max(d_a.norm()).max(d_b.norm());
            assert!(
                (value - m(a, b)).norm() < 1e-14 * scale,
                "a = {a}, b = {b}, z = {z}"
            );
            assert!(
                (d_a - expected_d_a).norm() < 1e-8 * scale,
                "a = {a}, b = {b}, z = {z}"
            );
            assert!(
                (d_b - expected_d_b).norm() < 1e-8 * scale,
                "a = {a}, b = {b}, z = {z}"
            );
        }
    }

    #[test]
    fn real_zeta_matches_reference_values() {
        let cases = [
            (2.0, std::f64::consts::PI.powi(2) / 6.0),
            (0.5, -1.4603545088095873),
            (-0.5, -0.20788622497735462),
            (-1.0, -1.0 / 12.0),
            (-2.0, 0.0),
        ];
        for (s, expected) in cases {
            let actual = zeta_real(s);
            assert!(
                (actual - expected).abs() < 3e-14,
                "s={s}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn adaptive_quadrature_integrates_smooth_and_endpoint_peaked_functions() {
        let polynomial = integrate(|x| x * x, 0.0, 1.0, 1e-14, 1000, "test").unwrap();
        assert!((polynomial - 1.0 / 3.0).abs() < 2e-14);

        let peaked = integrate(|x| x.sqrt(), 0.0, 1.0, 1e-13, 4000, "test").unwrap();
        assert!((peaked - 2.0 / 3.0).abs() < 2e-13);
    }

    #[test]
    fn inc_gamma_q_relative_accuracy_in_the_tail() {
        assert_eq!(inc_gamma_q(1.0, 0.0, "test"), Ok(1.0));
        // Q(1, x) = e^{-x} exactly, over a range where 1 - P(1, x) rounds to zero.
        for &x in &[0.5_f64, 1.5, 2.0, 10.0, 40.0, 100.0, 300.0, 700.0] {
            let want = (-x).exp();
            let rel = (inc_gamma_q(1.0, x, "test").unwrap() - want).abs() / want;
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
            let rel = (inc_gamma_q(0.5, x, "test").unwrap() - want).abs() / want;
            assert!(rel < 1e-11, "Q(0.5,{x}) rel {rel:.2e}");
        }
    }

    #[test]
    fn iteration_caps_report_no_convergence() {
        use crate::error::with_iteration_cap;
        let expected = Err(SpecialError::no_convergence("test", 1));
        with_iteration_cap(1, || {
            let (a, b, z) = (
                Complex64::new(0.5, 0.2),
                Complex64::new(1.5, 0.0),
                Complex64::new(2.0, 1.0),
            );
            assert_eq!(kummer_m(a, b, z, "test"), expected.map(|_: ()| a));
            assert_eq!(
                kummer_m_regularized(a, b, z, "test"),
                expected.map(|_: ()| a)
            );
            assert_eq!(
                kummer_m_regularized_with_parameter_derivatives(a, b, z, "test"),
                expected.map(|_: ()| [a; 3])
            );
            assert_eq!(tricomi_u(a, b, z, "test"), expected.map(|_: ()| a));
            assert_eq!(
                integrate(f64::sqrt, 0.0, 1.0, 1e-13, 4000, "test"),
                expected.map(|_: ()| 0.0)
            );
        });
        with_iteration_cap(16, || {
            assert_eq!(
                inc_gamma_q(0.5, 20.0, "test"),
                Err(SpecialError::no_convergence("test", 16))
            );
        });
    }
}
