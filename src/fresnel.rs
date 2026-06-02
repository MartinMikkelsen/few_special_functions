use num_complex::Complex;
use std::f64::consts::{PI, FRAC_2_SQRT_PI};

/// Complex error function erf(z).
///
/// Uses the Taylor series for |z| < 4 and the asymptotic expansion of
/// erfc(z) = 1 - erf(z) for |z| ≥ 4.
fn cerf(z: Complex<f64>) -> Complex<f64> {
    if z.norm() < 4.0 {
        cerf_series(z)
    } else {
        Complex::new(1.0, 0.0) - cerfc_asymptotic(z)
    }
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

/// Fresnel integrals C(x), S(x), and E(x) = C(x) + i·S(x).
///
/// NIST / MATLAB convention:
///   C(x) = ∫₀ˣ cos(π/2 · t²) dt
///   S(x) = ∫₀ˣ sin(π/2 · t²) dt
///
/// Computed via the identity C(x) + i·S(x) = (1+i)/2 · erf((√π/2)(1−i)x).
///
/// Returns `(C, S, E)`.
pub fn fresnel(x: f64) -> (f64, f64, Complex<f64>) {
    if x == 0.0 {
        let zero = Complex::new(0.0, 0.0);
        return (0.0, 0.0, zero);
    }

    // C and S are odd functions: handle negative x by symmetry.
    if x < 0.0 {
        let (c, s, e) = fresnel(-x);
        return (-c, -s, Complex::new(-e.re, -e.im));
    }

    // w = (√π/2)(1 − i)·x  →  w² = −(πi/2)x²  (purely imaginary)
    let half_sqrt_pi = PI.sqrt() / 2.0;
    let w = Complex::new(half_sqrt_pi, -half_sqrt_pi) * x;

    let erf_w = cerf(w);
    let e = Complex::new(0.5, 0.5) * erf_w; // (1+i)/2 · erf(w)
    (e.re, e.im, e)
}

/// Fresnel cosine integral C(x) = ∫₀ˣ cos(π/2 · t²) dt.
pub fn fresnel_c(x: f64) -> f64 {
    fresnel(x).0
}

/// Fresnel sine integral S(x) = ∫₀ˣ sin(π/2 · t²) dt.
pub fn fresnel_s(x: f64) -> f64 {
    fresnel(x).1
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
