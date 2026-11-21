//! Clausen functions Cl_n(θ).
//!
//! Ported from `src/Clausen.jl` of FewSpecialFunctions.jl
//! (<https://github.com/MartinMikkelsen/FewSpecialFunctions.jl>, MIT license),
//! commit b909e7b22696e942762f800ddad34a70d5b9d51d.
//! The algorithm is from <https://doi.org/10.1007/s10543-023-00944-4>. The
//! complex exponential integral `E₁`, which the Julia code takes from
//! SpecialFunctions.jl, is written for this crate from its power series and
//! continued fraction (DLMF 6.6.2 and 6.9.1).
//!
//! # Iteration caps and tolerances
//!
//! Where `|z| + Re z < 4`, E₁(z) sums its power series until a term falls below
//! `1e-17 (1 + |sum|)`, with a cap of 4000 terms; elsewhere it evaluates its
//! continued fraction by the modified Lentz method until a correction factor is
//! within `1e-15` of one, with a cap of 300 terms. Reaching either cap gives
//! [`SpecialError::NoConvergence`] from the `*_checked` functions and NaN from
//! the others. The caps are set so that only arguments at which E₁
//! overflows reach them.

use crate::SpecialError;
use crate::error::iteration_cap;
use num_complex::Complex;
use std::f64::consts::{EULER_GAMMA, PI};
use std::num::NonZeroUsize;

// Gauss rule for the Euler-Maclaurin tail weight w(t) = 1/(sqrt(t) cosh^2(pi sqrt(t)/2))
// on (0, inf), where t = 4y^2 and y = Im(z). Sum of the weights is 4/pi.
// Nodes (xi) and weights (a) for N = 10.
const XI_10: [f64; 10] = [
    1.02058363572825669e-1,
    1.10087335344093285e0,
    3.87538426676163313e0,
    9.49089900532915982e0,
    1.92369217680950509e1,
    3.48489807870071239e1,
    5.88422971202513756e1,
    9.52057687864990326e1,
    1.51245498588511132e2,
    2.44769266678480451e2,
];
const A_10: [f64; 10] = [
    1.03011687962788351e0,
    2.2580220852505727e-1,
    1.67040618192678355e-2,
    6.0460703772163201e-4,
    1.16718411051188568e-5,
    1.1536011315284419e-7,
    5.23125205372251291e-10,
    8.88585842372452499e-13,
    3.78823474931504689e-16,
    1.45749170427449731e-20,
];

// Same weight, N = 20 (extended precision).
const XI_20: [f64; 20] = [
    7.8332741180217739859429592034215121e-2,
    7.99480557580280868888504305965764455e-1,
    2.6335359843707676053063024600681851e0,
    6.067125354800522713778706639849939e0,
    1.1578014308014972867250241260973853e1,
    1.967574219682007399786022325652155e1,
    3.0929840293273144262920338413642651e1,
    4.5988905145672958078531429470368746e1,
    6.5602854796804040341325251118403039e1,
    9.0652152903729413194007204939563872e1,
    1.2218773346692348159267034032123833e2,
    1.6148763659034474884235758941294436e2,
    2.1014062487874825104404326118075493e2,
    2.7017538453427198248632043154940478e2,
    3.4427146853197803740611415515669254e2,
    4.3612864159248546858458013080006752e2,
    5.5117670639883226012194621182330183e2,
    6.9813208096151573112006881397004676e2,
    8.931889848952827736674475594859238e2,
    1.1765731082977506193890547798997328e3,
];
const A_20: [f64; 20] = [
    9.4521844770213813522739498751084446e-1,
    2.8673730910005180272566734208681658e-1,
    3.7990715886134892035536939814054822e-2,
    3.1053688857936936821748901103472756e-3,
    1.7979523985634735267965413590172828e-4,
    7.6599061254187375464405113636273785e-6,
    2.4224398021113042502544194520811499e-7,
    5.6724641616616866033851184587197347e-9,
    9.740052786086906643974014704731242e-11,
    1.2068824524541185034179793734503826e-12,
    1.0549929160747151930065637444912904e-14,
    6.3107861649521708824465976542847217e-17,
    2.4806740961948186082790889707150879e-19,
    6.068944075643647849080969319008971e-22,
    8.5770997286711710319485672039762078e-25,
    6.296750414836763110694582719147875e-28,
    2.0455706071903958823179960004147795e-31,
    2.258006419692752335532584347360038e-35,
    5.1214712907227134883116239441893341e-40,
    6.6554055933455521526556015293110494e-46,
];

// ζ(n) for odd n = 1, 3, 5 — used when θ = 0
const ZETA_3: f64 = 1.2020569031595942854_f64;
const ZETA_5: f64 = 1.0369277551433699341_f64;

const FAMILY: &str = "clausen";

/// Order `n` of the Clausen function Cl_n and of the tail primitive F_n.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClausenOrder {
    /// Cl_1(θ) = −ln|2 sin(θ/2)|.
    One = 1,
    /// Cl_2, whose value at π/2 is Catalan's constant.
    Two = 2,
    /// Cl_3.
    Three = 3,
    /// Cl_4.
    Four = 4,
    /// Cl_5.
    Five = 5,
    /// Cl_6.
    Six = 6,
}

impl ClausenOrder {
    /// Every order, in increasing order.
    pub const ALL: [Self; 6] = [
        Self::One,
        Self::Two,
        Self::Three,
        Self::Four,
        Self::Five,
        Self::Six,
    ];

    /// The order for `n`, or `None` unless `n` is in `1..=6`.
    ///
    /// # Examples
    ///
    /// ```
    /// use few_special_functions::clausen::ClausenOrder;
    ///
    /// assert_eq!(ClausenOrder::new(2), Some(ClausenOrder::Two));
    /// assert_eq!(ClausenOrder::new(7), None);
    /// assert_eq!(ClausenOrder::Two.get(), 2);
    /// ```
    pub const fn new(n: usize) -> Option<Self> {
        match n {
            1 => Some(Self::One),
            2 => Some(Self::Two),
            3 => Some(Self::Three),
            4 => Some(Self::Four),
            5 => Some(Self::Five),
            6 => Some(Self::Six),
            _ => None,
        }
    }

    /// The order as an integer.
    pub const fn get(self) -> usize {
        self as usize
    }
}

/// Gauss rule used for the Euler–Maclaurin tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TailNodes {
    /// The 10-node rule used by [`clausen`].
    Ten,
    /// The 20-node rule used by [`clausen_n20`].
    Twenty,
}

impl TailNodes {
    fn rule(self) -> (&'static [f64], &'static [f64]) {
        match self {
            Self::Ten => (&XI_10, &A_10),
            Self::Twenty => (&XI_20, &A_20),
        }
    }
}

/// Cap on the terms of the E₁ power series; `|z| = 700` needs about 1000.
const MAX_SERIES_TERMS: usize = 4_000;
/// Cap on the terms of the E₁ continued fraction.
const MAX_FRACTION_TERMS: usize = 300;

/// Exponential integral E₁(z) for complex z, z ≠ 0, |arg(z)| < π.
///
/// Where `|z| + Re z < 4` it sums the power series
/// `E₁(z) = −γ − ln z − Σ_{k≥1} (−z)^k / (k · k!)` (DLMF 6.6.2), whose terms
/// add up to at most `e^{|z| + Re z} ≈ 55` times the result. Elsewhere it
/// evaluates the continued fraction (DLMF 6.9.1)
///   E₁(z) = exp(-z) · 1/(z+ 1/(1+ 1/(z+ 2/(1+ 2/(z+ 3/...)))))
/// by the modified Lentz method; on that region it converges within about
/// 100 terms, while near the negative real axis it would need up to 10⁶.
///
/// CF coefficients: j=1 → a=1, b=z;
///   j=2k (even) → a=k, b=1;
///   j=2k+1 (odd, k≥1) → a=k, b=z.
fn expint_e1(z: Complex<f64>) -> Result<Complex<f64>, SpecialError> {
    // 1e-150 keeps TINY² = 1e-300 > 0 in f64 (avoids norm_sqr underflow in division)
    const TINY: Complex<f64> = Complex {
        re: 1e-150,
        im: 0.0,
    };

    if z.norm() + z.re < 4.0 {
        let max_terms = iteration_cap(MAX_SERIES_TERMS);
        // Σ_{k≥1} (−1)^{k+1} z^k / (k · k!); consecutive terms differ by the
        // factor −z (k − 1) / k².
        let mut term = z;
        let mut sum = z;
        for k in 2..=max_terms {
            let k = k as f64;
            term *= -z * ((k - 1.0) / (k * k));
            sum += term;
            if term.norm() < 1e-17 * (1.0 + sum.norm()) {
                return Ok(Complex::new(-EULER_GAMMA, 0.0) - z.ln() + sum);
            }
        }
        Err(SpecialError::no_convergence(FAMILY, max_terms))
    } else {
        let max_terms = iteration_cap(MAX_FRACTION_TERMS);
        // Modified Lentz algorithm.  b_0 = 0, so start f at TINY.
        let mut f = TINY;
        let mut c = f;
        let mut d = Complex::new(0.0, 0.0);

        for j in 1..=max_terms {
            let (a, b) = if j == 1 {
                (Complex::new(1.0, 0.0), z)
            } else if j.is_multiple_of(2) {
                let k = (j / 2) as f64;
                (Complex::new(k, 0.0), Complex::new(1.0, 0.0))
            } else {
                let k = ((j - 1) / 2) as f64;
                (Complex::new(k, 0.0), z)
            };

            d = b + a * d;
            if d.norm() < 1e-150 {
                d = TINY;
            }
            d = d.inv();

            c = b + a / c;
            if c.norm() < 1e-150 {
                c = TINY;
            }

            let delta = c * d;
            f *= delta;
            if (delta - Complex::new(1.0, 0.0)).norm() < 1e-15 {
                return Ok((-z).exp() * f);
            }
        }
        Err(SpecialError::no_convergence(FAMILY, max_terms))
    }
}

const COMPLEX_NAN: Complex<f64> = Complex {
    re: f64::NAN,
    im: f64::NAN,
};

/// Complex cosine integral Ci(z) = -½ (E₁(iz) + E₁(-iz)), with branch
/// correction +πi when Re(z) < 0.
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::ci_complex;
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// // Ci(1) = 0.33740392290096813… (mpmath)
/// let v = ci_complex(Complex64::new(1.0, 0.0));
/// assert!((v - Complex64::new(0.337_403_922_900_968_1, 0.0)).norm() < 1e-15);
///
/// // Ci(−x) = Ci(x) + iπ for x > 0
/// let w = ci_complex(Complex64::new(-1.0, 0.0));
/// assert!((w - v - Complex64::new(0.0, PI)).norm() < 1e-14);
/// ```
pub fn ci_complex(z: Complex<f64>) -> Complex<f64> {
    ci_complex_checked(z).unwrap_or(COMPLEX_NAN)
}

/// [`ci_complex`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if the series or the continued fraction
/// of E₁ reaches its cap (see the module documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::{ci_complex, ci_complex_checked};
/// use num_complex::Complex64;
///
/// let z = Complex64::new(1.0, 0.5);
/// assert_eq!(ci_complex_checked(z), Ok(ci_complex(z)));
/// ```
pub fn ci_complex_checked(z: Complex<f64>) -> Result<Complex<f64>, SpecialError> {
    if z.is_nan() || z == Complex::new(0.0, 0.0) {
        return Ok(COMPLEX_NAN);
    }
    if z.im == 0.0 && z.re.is_infinite() {
        return Ok(if z.re > 0.0 {
            Complex::new(0.0, 0.0)
        } else {
            Complex::new(0.0, PI)
        });
    }
    if z.is_infinite() {
        return Ok(COMPLEX_NAN);
    }

    let i = Complex::new(0.0, 1.0);
    let mut v = -(expint_e1(i * z)? + expint_e1(-i * z)?) * 0.5;
    if z.re < 0.0 {
        v += Complex::new(0.0, PI);
    }
    Ok(v)
}

/// Euler-Maclaurin tail primitive F_n(z, θ).
///
/// `F_1(z, θ) = Ci(zθ)`, and `F_2` is its antiderivative in θ.
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::{ci_complex, f_clausen, ClausenOrder};
/// use num_complex::Complex64;
///
/// let z = Complex64::new(1.0, 0.5);
/// let theta = 0.8;
/// assert_eq!(f_clausen(ClausenOrder::One, z, theta), ci_complex(z * theta));
///
/// // F_2(z, θ) is an antiderivative of F_1(z, θ) in θ.
/// let h = 1e-5;
/// let slope = (f_clausen(ClausenOrder::Two, z, theta + h) - f_clausen(ClausenOrder::Two, z, theta - h)) / (2.0 * h);
/// assert!((slope - f_clausen(ClausenOrder::One, z, theta)).norm() < 1e-9);
/// ```
pub fn f_clausen(order: ClausenOrder, z: Complex<f64>, theta: f64) -> Complex<f64> {
    f_clausen_checked(order, z, theta).unwrap_or(COMPLEX_NAN)
}

/// [`f_clausen`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if the evaluation of Ci reaches an
/// iteration cap (see the module documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::{f_clausen, f_clausen_checked, ClausenOrder};
/// use num_complex::Complex64;
///
/// let z = Complex64::new(1.0, 0.5);
/// assert_eq!(
///     f_clausen_checked(ClausenOrder::Three, z, 0.8),
///     Ok(f_clausen(ClausenOrder::Three, z, 0.8))
/// );
/// ```
pub fn f_clausen_checked(
    order: ClausenOrder,
    z: Complex<f64>,
    theta: f64,
) -> Result<Complex<f64>, SpecialError> {
    let tz = z * theta; // θz ∈ ℂ
    let ci = ci_complex_checked(tz)?;
    let s = tz.sin();
    let c = tz.cos();

    Ok(match order {
        ClausenOrder::One => ci,
        ClausenOrder::Two => (tz * ci - s) / z,
        ClausenOrder::Three => {
            let z2 = z * z;
            -(z2 * 2.0).inv() * (tz * tz * ci + c - tz * s)
        }
        ClausenOrder::Four => {
            let z3 = z * z * z;
            -(z3 * 6.0).inv()
                * (tz * tz * tz * ci + (Complex::new(2.0, 0.0) - tz * tz) * s + tz * c)
        }
        ClausenOrder::Five => {
            let z4 = z * z * z * z;
            (z4 * 24.0).inv()
                * (tz * tz * tz * tz * ci
                    + tz * (Complex::new(2.0, 0.0) - tz * tz) * s
                    + (tz * tz - Complex::new(6.0, 0.0)) * c)
        }
        ClausenOrder::Six => {
            let z5 = z * z * z * z * z;
            (z5 * 120.0).inv()
                * (tz * tz * tz * tz * tz * ci + tz * (tz * tz - Complex::new(6.0, 0.0)) * c
                    - (tz * tz * tz * tz - tz * tz * 2.0 + Complex::new(24.0, 0.0)) * s)
        }
    })
}

/// Clausen series summand `sin(kθ)/k^n` for even `n`, and
/// `cos(kθ)/k^n` for odd `n`.
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::f_n;
///
/// let theta = 0.7_f64;
/// assert_eq!(f_n(2, 3, theta), (3.0 * theta).sin() / 9.0);
/// assert_eq!(f_n(1, 3, theta), (3.0 * theta).cos() / 3.0);
/// ```
pub fn f_n(n: usize, k: usize, theta: f64) -> f64 {
    let kt = k as f64 * theta;
    if n.is_multiple_of(2) {
        kt.sin() / (k as f64).powi(n as i32)
    } else {
        kt.cos() / (k as f64).powi(n as i32)
    }
}

/// Cl_n at θ ≡ 0 (mod 2π).
fn at_zero(order: ClausenOrder) -> f64 {
    match order {
        ClausenOrder::One => f64::INFINITY,
        ClausenOrder::Two | ClausenOrder::Four | ClausenOrder::Six => 0.0,
        ClausenOrder::Three => ZETA_3,
        ClausenOrder::Five => ZETA_5,
    }
}

fn clausen_impl(
    order: ClausenOrder,
    theta: f64,
    nodes: TailNodes,
    m: NonZeroUsize,
) -> Result<f64, SpecialError> {
    let n = order.get();
    let m = m.get();
    if !theta.is_finite() {
        return Ok(f64::NAN);
    }
    if theta == 0.0 {
        return Ok(at_zero(order));
    }

    let theta_mod = theta.rem_euclid(2.0 * PI);

    let (phi, sign) = if theta_mod <= PI {
        (theta_mod, 1.0_f64)
    } else {
        (2.0 * PI - theta_mod, (-1.0_f64).powi((n as i32) + 1))
    };

    // phi = 0 after folding means theta is a multiple of 2π
    if phi == 0.0 {
        return Ok(at_zero(order));
    }

    if order == ClausenOrder::One {
        if phi.abs() < 1e-14 {
            return Ok(f64::INFINITY);
        }
        return Ok(sign * (-(2.0 * (phi / 2.0).sin()).abs().ln()));
    }

    let s1: f64 = (1..m).map(|k| f_n(n, k, phi)).sum();

    let (xi, weights) = nodes.rule();
    let mut s2 = 0.0;
    for (&xi_v, &w) in xi.iter().zip(weights) {
        let z = Complex::new(m as f64 - 0.5, 0.5 * xi_v.sqrt());
        s2 += w * f_clausen_checked(order, z, phi)?.re;
    }

    Ok(sign * (s1 - PI / 4.0 * s2))
}

/// Terms summed directly before the Euler–Maclaurin tail: 20.
const DIRECT_TERMS: NonZeroUsize = NonZeroUsize::MIN.saturating_add(19);

/// Clausen function Cl_n(θ) for orders n = 1..=6.
///
/// The Clausen functions generalize the dilogarithm and are defined by the
/// Fourier-like series:
///
/// ```text
/// Cl_{2m}(θ)   = Σ_{k=1}^∞ sin(kθ) / k^{2m}
/// Cl_{2m+1}(θ) = Σ_{k=1}^∞ cos(kθ) / k^{2m+1}
/// ```
///
/// They appear in the calculation of loop integrals in quantum field theory
/// and in certain lattice sums in condensed matter physics.
///
/// Uses N = 10 Euler-Maclaurin quadrature nodes (relative error < 10⁻¹⁴
/// for most arguments; use [`clausen_n20`] for extended precision).
///
/// Reference: doi:10.1007/s10543-023-00944-4
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::{clausen, ClausenOrder};
///
/// // Cl_2(π/2) = Catalan's constant G ≈ 0.9159655941
/// let g = clausen(ClausenOrder::Two, std::f64::consts::FRAC_PI_2);
/// assert!((g - 0.9159655941).abs() < 1e-9);
/// ```
pub fn clausen(order: ClausenOrder, theta: f64) -> f64 {
    clausen_checked(order, theta).unwrap_or(f64::NAN)
}

/// [`clausen`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// [`SpecialError::NoConvergence`] if the evaluation of E₁ in the tail reaches
/// an iteration cap (see the module documentation).
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::{clausen, clausen_checked, ClausenOrder};
///
/// assert_eq!(clausen_checked(ClausenOrder::Four, 1.5), Ok(clausen(ClausenOrder::Four, 1.5)));
/// ```
pub fn clausen_checked(order: ClausenOrder, theta: f64) -> Result<f64, SpecialError> {
    clausen_impl(order, theta, TailNodes::Ten, DIRECT_TERMS)
}

/// Clausen function Cl_n(θ) using N = 20 quadrature nodes for extended precision.
///
/// Same algorithm as [`clausen`] but with more nodes, giving roughly 2× more
/// significant digits. Use when you need accuracy below ~10⁻¹⁴.
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::{clausen_n20, ClausenOrder};
///
/// // Cl_2(π/3) ≈ 1.01494 (Clausen's original value)
/// let v = clausen_n20(ClausenOrder::Two, std::f64::consts::PI / 3.0);
/// assert!((v - 1.01494).abs() < 1e-5);
/// ```
pub fn clausen_n20(order: ClausenOrder, theta: f64) -> f64 {
    clausen_n20_checked(order, theta).unwrap_or(f64::NAN)
}

/// [`clausen_n20`], reporting a numerical failure instead of returning NaN.
///
/// # Errors
///
/// As for [`clausen_checked`].
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::{clausen_n20, clausen_n20_checked, ClausenOrder};
///
/// assert_eq!(
///     clausen_n20_checked(ClausenOrder::Four, 1.5),
///     Ok(clausen_n20(ClausenOrder::Four, 1.5))
/// );
/// ```
pub fn clausen_n20_checked(order: ClausenOrder, theta: f64) -> Result<f64, SpecialError> {
    clausen_impl(order, theta, TailNodes::Twenty, DIRECT_TERMS)
}

/// Clausen function with Julia-compatible quadrature and direct-sum controls.
///
/// `nodes` selects the Gauss rule of the Euler–Maclaurin tail, and the first
/// `direct_terms − 1` terms of the series are summed directly.
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::{clausen, clausen_with_options, ClausenOrder, TailNodes};
/// use std::f64::consts::FRAC_PI_2;
/// use std::num::NonZeroUsize;
///
/// // Cl_2(π/2) is Catalan's constant.
/// let catalan = 0.915_965_594_177_219;
/// let terms = NonZeroUsize::new(40).expect("40 is nonzero");
/// let v = clausen_with_options(ClausenOrder::Two, FRAC_PI_2, TailNodes::Twenty, terms);
/// assert!((v - catalan).abs() < 1e-14);
/// assert!((v - clausen(ClausenOrder::Two, FRAC_PI_2)).abs() < 1e-14);
/// ```
pub fn clausen_with_options(
    order: ClausenOrder,
    theta: f64,
    nodes: TailNodes,
    direct_terms: NonZeroUsize,
) -> f64 {
    clausen_with_options_checked(order, theta, nodes, direct_terms).unwrap_or(f64::NAN)
}

/// [`clausen_with_options`], reporting a numerical failure instead of
/// returning NaN.
///
/// # Errors
///
/// As for [`clausen_checked`].
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::{clausen_with_options_checked, ClausenOrder, TailNodes};
/// use std::num::NonZeroUsize;
///
/// let terms = NonZeroUsize::new(20).expect("20 is nonzero");
/// let v = clausen_with_options_checked(ClausenOrder::Three, 0.4, TailNodes::Ten, terms);
/// assert!(v.is_ok());
/// ```
pub fn clausen_with_options_checked(
    order: ClausenOrder,
    theta: f64,
    nodes: TailNodes,
    direct_terms: NonZeroUsize,
) -> Result<f64, SpecialError> {
    clausen_impl(order, theta, nodes, direct_terms)
}

// Unit tests for private internals — public API is tested in tests/clausen.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::with_iteration_cap;

    fn e1(z: Complex<f64>) -> Complex<f64> {
        expint_e1(z).unwrap_or(COMPLEX_NAN)
    }

    #[test]
    fn expint_series_branch() {
        // E₁(1) via convergent series (|z| < 2)
        let e = e1(Complex::new(1.0, 0.0));
        assert!((e.re - 0.21938393439552029).abs() < 1e-12);
        assert!(e.im.abs() < 1e-15);
    }

    #[test]
    fn expint_cf_branch() {
        // E₁(2) via modified Lentz CF (|z| ≥ 2)
        let e = e1(Complex::new(2.0, 0.0));
        assert!((e.re - 0.04890051070806112).abs() < 1e-10);
    }

    #[test]
    fn ci_real_axis() {
        // Ci(1) ≈ 0.3374039229009681
        let c = ci_complex(Complex::new(1.0, 0.0));
        assert!((c.re - 0.3374039229009681).abs() < 1e-10);
    }

    #[test]
    fn nan_and_infinite_arguments_are_values() {
        assert!(ci_complex_checked(Complex::new(f64::NAN, 1.0)).is_ok_and(|v| v.is_nan()));
        for theta in [f64::NAN, f64::INFINITY] {
            assert!(clausen_checked(ClausenOrder::Two, theta).is_ok_and(f64::is_nan));
        }
    }

    #[test]
    fn iteration_cap_reports_no_convergence() {
        let expected = Some(SpecialError::no_convergence(FAMILY, 1));
        with_iteration_cap(1, || {
            // Ci(z) evaluates E₁(±iz): z = 1 takes the series and z = 5 the
            // continued fraction.
            for z in [Complex::new(1.0, 0.0), Complex::new(5.0, 0.0)] {
                assert_eq!(ci_complex_checked(z).err(), expected, "z = {z}");
                assert!(ci_complex(z).is_nan(), "z = {z}");
            }
            // Cl_1 has a closed form; the other orders evaluate E₁ in the tail.
            for order in ClausenOrder::ALL.into_iter().skip(1) {
                assert_eq!(clausen_checked(order, 1.0).err(), expected, "{order:?}");
                assert!(clausen(order, 1.0).is_nan(), "{order:?}");
            }
        });
    }
}
