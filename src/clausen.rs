use num_complex::Complex;
use std::f64::consts::PI;

const EULER_GAMMA: f64 = 0.5772156649015328606_f64;

// Gauss-Laguerre nodes (xi) and weights (a) for N = 10
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

// Gauss-Laguerre nodes and weights for N = 20 (extended precision)
const XI_20: [f64; 20] = [
    7.8332741180217739859429592034215121e-2,
    7.99480557580280868888504305965764455e-1,
    2.633559843707676053063024600681851e0,
    6.067125354800522713778706639849939e0,
    1.1578014308014972867250241260973853e1,
    1.967574219682007399786022325652155e1,
    3.09284092373144262920338413642651e1,
    4.5988905145672958078531429470368746e1,
    6.5602547968040040431325251181403039e1,
    9.0652159037294131940072043099536872e1,
    1.2218773346692348159267034032123833e2,
    1.6148763659034474884235758941294436e2,
    2.1014062487874825104404326118075493e2,
    2.7017538454329128248632034154940478e2,
    3.4427146853197803740611415515669254e2,
    4.3612864159248546858458013080006752e2,
    5.5117670639883226012194621182330183e2,
    6.9813208096151573112006881397004676e2,
    8.9318894895282773667447559458923804e2,
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
    5.67246161661686603385118458797347e-9,
    9.740052786086906643974014704731242e-11,
    1.2068824524541185034179793734503826e-12,
    1.0549929160747159130056637444912904e-14,
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

/// Exponential integral E₁(z) for complex z, z ≠ 0, |arg(z)| < π.
///
/// Uses a convergent Taylor series for |z| < 2, and the modified Lentz
/// continued fraction algorithm for |z| ≥ 2.
///
/// CF representation (A&S 5.1.22):
///   E₁(z) = exp(-z) · 1/(z+ 1/(1+ 1/(z+ 2/(1+ 2/(z+ 3/...)))))
///
/// CF coefficients: j=1 → a=1, b=z;
///   j=2k (even) → a=k, b=1;
///   j=2k+1 (odd, k≥1) → a=k, b=z.
fn expint_e1(z: Complex<f64>) -> Complex<f64> {
    // 1e-150 keeps TINY² = 1e-300 > 0 in f64 (avoids norm_sqr underflow in division)
    const TINY: Complex<f64> = Complex {
        re: 1e-150,
        im: 0.0,
    };

    if z.norm() < 2.0 {
        // Series: E₁(z) = -γ - ln z + Σ_{k=1}^∞ (-1)^{k+1} z^k / (k · k!)
        let mut sum = Complex::new(0.0, 0.0);
        let mut pow_z = z;
        let mut factorial = 1.0_f64;
        for k in 1_usize..=80 {
            factorial *= k as f64;
            let sign = if !k.is_multiple_of(2) { 1.0 } else { -1.0 };
            let term = pow_z * (sign / (k as f64 * factorial));
            sum += term;
            pow_z *= z;
            if term.norm() < 1e-17 * (1.0 + sum.norm()) {
                break;
            }
        }
        Complex::new(-EULER_GAMMA, 0.0) - z.ln() + sum
    } else {
        // Modified Lentz algorithm.  b_0 = 0, so start f at TINY.
        let mut f = TINY;
        let mut c = f;
        let mut d = Complex::new(0.0, 0.0);

        for j in 1_usize..=300 {
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
                break;
            }
        }

        (-z).exp() * f
    }
}

/// Complex cosine integral Ci(z) = -½ (E₁(iz) + E₁(-iz)), with branch
/// correction +πi when Re(z) < 0.
fn ci_complex(z: Complex<f64>) -> Complex<f64> {
    if z == Complex::new(0.0, 0.0) {
        return Complex::new(f64::NAN, f64::NAN);
    }
    if z.im == 0.0 && z.re.is_infinite() {
        return if z.re > 0.0 {
            Complex::new(0.0, 0.0)
        } else {
            Complex::new(0.0, PI)
        };
    }
    if z.is_infinite() {
        return Complex::new(f64::NAN, f64::NAN);
    }

    let i = Complex::new(0.0, 1.0);
    let mut v = -(expint_e1(i * z) + expint_e1(-i * z)) * 0.5;
    if z.re < 0.0 {
        v += Complex::new(0.0, PI);
    }
    v
}

/// Euler-Maclaurin tail primitive F_n(z, θ) — returns Re(F_n(z, θ)).
fn f_clausen(n: usize, z: Complex<f64>, theta: f64) -> f64 {
    let tz = z * theta; // θz ∈ ℂ
    let ci = ci_complex(tz);
    let s = tz.sin();
    let c = tz.cos();

    (match n {
        1 => ci,
        2 => (tz * ci - s) / z,
        3 => {
            let z2 = z * z;
            -(z2 * 2.0).inv() * (tz * tz * ci + c - tz * s)
        }
        4 => {
            let z3 = z * z * z;
            -(z3 * 6.0).inv()
                * (tz * tz * tz * ci + (Complex::new(2.0, 0.0) - tz * tz) * s + tz * c)
        }
        5 => {
            let z4 = z * z * z * z;
            (z4 * 24.0).inv()
                * (tz * tz * tz * tz * ci
                    + tz * (Complex::new(2.0, 0.0) - tz * tz) * s
                    + (tz * tz - Complex::new(6.0, 0.0)) * c)
        }
        6 => {
            let z5 = z * z * z * z * z;
            (z5 * 120.0).inv()
                * (tz * tz * tz * tz * tz * ci + tz * (tz * tz - Complex::new(6.0, 0.0)) * c
                    - (tz * tz * tz * tz - tz * tz * 2.0 + Complex::new(24.0, 0.0)) * s)
        }
        _ => panic!("n must be 1..=6"),
    })
    .re
}

fn sum_term(n: usize, k: usize, theta: f64) -> f64 {
    let kt = k as f64 * theta;
    if n.is_multiple_of(2) {
        kt.sin() / (k as f64).powi(n as i32)
    } else {
        kt.cos() / (k as f64).powi(n as i32)
    }
}

fn clausen_impl(n: usize, theta: f64, xi: &[f64], weights: &[f64]) -> f64 {
    assert!((1..=6).contains(&n), "n must be 1..=6, got {n}");

    if theta == 0.0 {
        return match n {
            1 => f64::INFINITY,
            2 | 4 | 6 => 0.0,
            3 => ZETA_3,
            5 => ZETA_5,
            _ => unreachable!(),
        };
    }

    let theta_mod = theta.rem_euclid(2.0 * PI);

    let (phi, sign) = if theta_mod <= PI {
        (theta_mod, 1.0_f64)
    } else {
        (2.0 * PI - theta_mod, (-1.0_f64).powi((n as i32) + 1))
    };

    // phi = 0 after folding means theta is a multiple of 2π
    if phi == 0.0 {
        return match n {
            1 => f64::INFINITY,
            2 | 4 | 6 => 0.0,
            3 => ZETA_3,
            5 => ZETA_5,
            _ => unreachable!(),
        };
    }

    if n == 1 {
        if phi.abs() < 1e-14 {
            return f64::INFINITY;
        }
        return sign * (-(2.0 * (phi / 2.0).sin()).abs().ln());
    }

    let m = 20_usize;

    let s1: f64 = (1..m).map(|k| sum_term(n, k, phi)).sum();

    let s2: f64 = xi
        .iter()
        .zip(weights.iter())
        .map(|(&xi_v, &w)| {
            let z = Complex::new(m as f64 - 0.5, 0.5 * xi_v.sqrt());
            w * f_clausen(n, z, phi)
        })
        .sum();

    sign * (s1 - PI / 4.0 * s2)
}

/// Clausen function Cl_n(θ) for orders n = 1..6.
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
/// # Panics
/// Panics if `n` is not in 1..=6.
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::clausen;
///
/// // Cl_2(π/2) = Catalan's constant G ≈ 0.9159655941
/// let g = clausen(2, std::f64::consts::FRAC_PI_2);
/// assert!((g - 0.9159655941).abs() < 1e-9);
/// ```
pub fn clausen(n: usize, theta: f64) -> f64 {
    clausen_impl(n, theta, &XI_10, &A_10)
}

/// Clausen function Cl_n(θ) using N = 20 quadrature nodes for extended precision.
///
/// Same algorithm as [`clausen`] but with more nodes, giving roughly 2× more
/// significant digits. Use when you need accuracy below ~10⁻¹⁴.
///
/// # Examples
///
/// ```
/// use few_special_functions::clausen::clausen_n20;
///
/// // Cl_2(π/3) ≈ 1.01494 (Clausen's original value)
/// let v = clausen_n20(2, std::f64::consts::PI / 3.0);
/// assert!((v - 1.01494).abs() < 1e-5);
/// ```
pub fn clausen_n20(n: usize, theta: f64) -> f64 {
    clausen_impl(n, theta, &XI_20, &A_20)
}

// Unit tests for private internals — public API is tested in tests/clausen.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expint_series_branch() {
        // E₁(1) via convergent series (|z| < 2)
        let e = expint_e1(Complex::new(1.0, 0.0));
        assert!((e.re - 0.21938393439552029).abs() < 1e-12);
        assert!(e.im.abs() < 1e-15);
    }

    #[test]
    fn expint_cf_branch() {
        // E₁(2) via modified Lentz CF (|z| ≥ 2)
        let e = expint_e1(Complex::new(2.0, 0.0));
        assert!((e.re - 0.04890051070806112).abs() < 1e-10);
    }

    #[test]
    fn ci_real_axis() {
        // Ci(1) ≈ 0.3374039229009681
        let c = ci_complex(Complex::new(1.0, 0.0));
        assert!((c.re - 0.3374039229009681).abs() < 1e-10);
    }
}
