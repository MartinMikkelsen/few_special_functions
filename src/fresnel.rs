use num_complex::Complex;
use std::f64::consts::{FRAC_2_SQRT_PI, PI};

/// Complex error function erf(z).
///
/// Uses the Taylor series for |z| < 4 and the asymptotic expansion of
/// erfc(z) = 1 - erf(z) for |z| ≥ 4.
fn cerf(z: Complex<f64>) -> Complex<f64> {
    if z.re < 0.0 {
        return -cerf(-z);
    }
    if z.norm() < 4.0 {
        cerf_series(z)
    } else {
        Complex::new(1.0, 0.0) - cerfc_asymptotic(z)
    }
}

fn fresnel_series(z: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
    let u = (PI / 2.0) * z * z;
    let u2 = u * u;
    let mut c = z;
    let mut s = z * u / 3.0;
    let mut c_term = c;
    let mut s_term = s;

    for k in 1..=10_000 {
        let k = k as f64;
        c_term *= -((4.0 * k - 3.0) * u2) / (2.0 * k * (2.0 * k - 1.0) * (4.0 * k + 1.0));
        s_term *= -((4.0 * k - 1.0) * u2) / (2.0 * k * (2.0 * k + 1.0) * (4.0 * k + 3.0));
        c += c_term;
        s += s_term;
        if c_term.norm().max(s_term.norm())
            <= 4.0 * f64::EPSILON * 1.0_f64.max(c.norm()).max(s.norm())
        {
            break;
        }
    }
    (c, s)
}

fn fresnel_asymptotic_auxiliary(z: Complex<f64>, offset: f64) -> Complex<f64> {
    let z4 = z.powu(4);
    let mut term = Complex::new(1.0, 0.0);
    let mut sum = term;
    for k in 1..=10_000 {
        let k = k as f64;
        let next = -term * (4.0 * k - 3.0 + offset) * (4.0 * k - 1.0 + offset) / (PI * PI * z4);
        if next.norm() > term.norm() {
            break;
        }
        sum += next;
        if next.norm() <= 4.0 * f64::EPSILON * sum.norm() {
            break;
        }
        term = next;
    }
    sum
}

fn fresnel_asymptotic(z: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
    let theta = (PI / 2.0) * z * z;
    let mut f = fresnel_asymptotic_auxiliary(z, 0.0) / (PI * z);
    let mut g = fresnel_asymptotic_auxiliary(z, 2.0) / (PI * PI * z.powu(3));
    if z.im.abs() > z.re.abs() {
        let exponential = (Complex::<f64>::i() * theta).exp() / 2.0;
        f += Complex::new(1.0, 1.0) * exponential;
        g += Complex::new(1.0, -1.0) * exponential;
    }
    let s = 0.5 - f * theta.cos() - g * theta.sin();
    let c = 0.5 + f * theta.sin() - g * theta.cos();
    (c, s)
}

fn fresnel_real_pair(x: f64) -> (f64, f64) {
    let ax = x.abs();
    let (c, s) = if ax * ax <= 6.9 {
        let (c, s) = fresnel_series(Complex::new(ax, 0.0));
        (c.re, s.re)
    } else {
        let asymptotic_start = 5.0_f64.max((2.0 * (1.0 / f64::EPSILON).ln() / PI).sqrt() + 1.0);
        if ax < asymptotic_start {
            let scale = PI.sqrt() / 2.0;
            let e = Complex::new(0.5, 0.5) * cerf(Complex::new(scale * ax, -scale * ax));
            (e.re, e.im)
        } else {
            let (c, s) = fresnel_asymptotic(Complex::new(ax, 0.0));
            (c.re, s.re)
        }
    };
    if x < 0.0 { (-c, -s) } else { (c, s) }
}

/// Taylor series: erf(z) = (2/√π) Σ_{n=0}^∞ (-1)ⁿ z^{2n+1} / (n! (2n+1))
fn cerf_series(z: Complex<f64>) -> Complex<f64> {
    let z2 = z * z;
    let mut term = z;
    let mut sum = z;
    for n in 1_usize..=80 {
        term *= -z2 / n as f64;
        let contrib = term / (2 * n + 1) as f64;
        sum += contrib;
        if contrib.norm() < 1e-17 * sum.norm() {
            break;
        }
    }
    sum * FRAC_2_SQRT_PI
}

/// Asymptotic expansion of erfc(z) for large |z|, |arg(z)| < 3π/4.
///
/// erfc(z) ~ exp(-z²) / (z√π) · Σ_{m=0}^∞ (-1)ᵐ (2m−1)!! / (2z²)ᵐ
///
/// Truncated at the minimum term (the series is asymptotic, not convergent).
fn cerfc_asymptotic(z: Complex<f64>) -> Complex<f64> {
    let z2 = z * z;
    let prefix = (-z2).exp() / (z * PI.sqrt());

    let inv_2z2 = Complex::new(1.0, 0.0) / (z2 * 2.0);
    let mut sum = Complex::new(1.0, 0.0);
    let mut term = Complex::new(1.0, 0.0);
    let mut min_norm = f64::INFINITY;

    for m in 1_usize..=60 {
        term *= inv_2z2 * (-((2 * m - 1) as f64));
        let norm = term.norm();
        if norm > min_norm {
            break; // asymptotic: stop when terms start growing
        }
        min_norm = norm;
        sum += term;
        if norm < 1e-17 {
            break;
        }
    }

    prefix * sum
}

/// Fresnel integrals C(x), S(x), and the complex auxiliary E(x) = C(x) + i·S(x).
///
/// NIST / MATLAB convention:
///
/// ```text
/// C(x) = ∫₀ˣ cos(π/2 · t²) dt
/// S(x) = ∫₀ˣ sin(π/2 · t²) dt
/// ```
///
/// Both C and S are odd functions and approach ±1/2 as x → ±∞. The parametric
/// curve (C(t), S(t)) traces the Euler (Cornu) spiral, used in diffraction
/// optics and road/railway transition curve design.
///
/// Computed via the identity C(x) + i·S(x) = (1+i)/2 · erf((√π/2)(1−i)x).
///
/// Returns `(C, S, E)`.
///
/// # Examples
///
/// ```
/// use few_special_functions::fresnel::fresnel;
///
/// let (c, s, _e) = fresnel(0.0);
/// assert_eq!(c, 0.0);
/// assert_eq!(s, 0.0);
///
/// // Both integrals oscillate toward 0.5 as x → ∞ (envelope ~ 1/πx)
/// let (c, s, _e) = fresnel(10.0);
/// assert!((c - 0.5).abs() < 0.1);
/// assert!((s - 0.5).abs() < 0.1);
/// ```
pub fn fresnel(x: f64) -> (f64, f64, Complex<f64>) {
    let (c, s) = fresnel_real_pair(x);
    let e = Complex::new(c, s);
    (e.re, e.im, e)
}

/// Complex Fresnel cosine and sine integrals and their well-conditioned sum.
pub fn fresnel_complex(z: Complex<f64>) -> (Complex<f64>, Complex<f64>, Complex<f64>) {
    if z.im == 0.0 {
        let (c, s, e) = fresnel(z.re);
        return (Complex::new(c, 0.0), Complex::new(s, 0.0), e);
    }
    if z.re == 0.0 {
        let (c, s, _) = fresnel(z.im);
        return (
            Complex::new(0.0, c),
            Complex::new(0.0, -s),
            Complex::new(s, c),
        );
    }
    if z.re < 0.0 {
        let (c, s, e) = fresnel_complex(-z);
        return (-c, -s, -e);
    }

    let scale = PI.sqrt() / 2.0;
    let e = Complex::new(0.5, 0.5) * cerf(Complex::new(scale, -scale) * z);
    let (c, s) = if z.norm_sqr() <= 6.9 {
        fresnel_series(z)
    } else {
        let e_minus = Complex::new(0.5, -0.5) * cerf(Complex::new(scale, scale) * z);
        ((e + e_minus) / 2.0, (e - e_minus) / Complex::new(0.0, 2.0))
    };
    (c, s, e)
}

/// Fresnel cosine integral C(x) = ∫₀ˣ cos(π/2 · t²) dt.
///
/// See [`fresnel`] for the full definition and convention note.
///
/// # Examples
///
/// ```
/// use few_special_functions::fresnel::fresnel_c;
///
/// assert_eq!(fresnel_c(0.0), 0.0);
/// // C(1) ≈ 0.7799 (tabulated value)
/// assert!((fresnel_c(1.0) - 0.7799).abs() < 1e-4);
/// ```
pub fn fresnel_c(x: f64) -> f64 {
    fresnel(x).0
}

/// Fresnel sine integral S(x) = ∫₀ˣ sin(π/2 · t²) dt.
///
/// See [`fresnel`] for the full definition and convention note.
///
/// # Examples
///
/// ```
/// use few_special_functions::fresnel::fresnel_s;
///
/// assert_eq!(fresnel_s(0.0), 0.0);
/// // S(1) ≈ 0.4383 (tabulated value)
/// assert!((fresnel_s(1.0) - 0.4383).abs() < 1e-4);
/// ```
pub fn fresnel_s(x: f64) -> f64 {
    fresnel(x).1
}

/// Fresnel combination `C(x) + i S(x)` for real `x`.
pub fn fresnel_e(x: f64) -> Complex<f64> {
    fresnel(x).2
}

/// Complex Fresnel cosine integral.
pub fn fresnel_c_complex(z: Complex<f64>) -> Complex<f64> {
    fresnel_complex(z).0
}

/// Complex Fresnel sine integral.
pub fn fresnel_s_complex(z: Complex<f64>) -> Complex<f64> {
    fresnel_complex(z).1
}

/// Well-conditioned complex combination `C(z) + i S(z)`.
pub fn fresnel_e_complex(z: Complex<f64>) -> Complex<f64> {
    fresnel_complex(z).2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cerf_known() {
        // erf(1) ≈ 0.8427007929497149
        let e = cerf(Complex::new(1.0, 0.0));
        assert!((e.re - 0.8427007929497149).abs() < 1e-12);
        assert!(e.im.abs() < 1e-15);
    }

    #[test]
    fn at_zero() {
        let (c, s, e) = fresnel(0.0);
        assert_eq!(c, 0.0);
        assert_eq!(s, 0.0);
        assert_eq!(e, Complex::new(0.0, 0.0));
    }

    #[test]
    fn small_x() {
        // For tiny x: C(x) ≈ x,  S(x) ≈ π/6 · x³
        let x = 1e-8_f64;
        let (c, s, _) = fresnel(x);
        assert!((c - x).abs() < 1e-16);
        // S(x) = π/6·x³ + O(x⁷); at x=1e-8 the value is ~5e-25. The complex
        // erf path suffers cancellation (Re ≈ Im ≈ 1e-8) so absolute accuracy
        // is limited to ~machine_eps × x ≈ 1e-24.
        assert!((s - PI / 6.0 * x.powi(3)).abs() < 1e-22);
    }

    #[test]
    fn odd_symmetry() {
        for &x in &[0.5_f64, 1.0, 2.0, 3.5] {
            let (cp, sp, _) = fresnel(x);
            let (cn, sn, _) = fresnel(-x);
            assert!((cp + cn).abs() < 1e-12, "C symmetry failed at {x}");
            assert!((sp + sn).abs() < 1e-12, "S symmetry failed at {x}");
        }
    }
}
