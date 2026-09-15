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

pub(crate) fn kummer_m(a: Complex64, b: Complex64, z: Complex64) -> Complex64 {
    if z.re < 0.0 {
        return z.exp() * kummer_m_series(b - a, b, -z);
    }
    kummer_m_series(a, b, z)
}

fn kummer_m_series(a: Complex64, b: Complex64, z: Complex64) -> Complex64 {
    let mut sum = Complex64::new(1.0, 0.0);
    let mut compensation = Complex64::new(0.0, 0.0);
    let mut term = Complex64::new(1.0, 0.0);

    for index in 1..=100_000 {
        let previous = (index - 1) as f64;
        term *= (a + previous) * z / ((b + previous) * index as f64);

        let corrected = term - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
        if term.norm() <= 4.0 * f64::EPSILON * sum.norm().max(1.0) {
            return sum;
        }
    }
    panic!("Kummer M series did not converge")
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

pub(crate) fn kummer_m_regularized(a: Complex64, b: Complex64, z: Complex64) -> Complex64 {
    if z.re < 0.0 {
        return z.exp() * kummer_m_regularized(b - a, b, -z);
    }

    let mut coefficient = Complex64::new(1.0, 0.0);
    let mut sum = reciprocal_gamma(b);
    let mut compensation = Complex64::new(0.0, 0.0);
    for index in 1..=100_000 {
        let previous = (index - 1) as f64;
        coefficient *= (a + previous) * z / index as f64;
        let term = coefficient * reciprocal_gamma(b + index as f64);
        let corrected = term - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
        if index >= 8 && term.norm() <= 4.0 * f64::EPSILON * sum.norm().max(1.0) {
            return sum;
        }
    }
    panic!("regularized Kummer M series did not converge")
}

pub(crate) fn tricomi_u(a: Complex64, b: Complex64, z: Complex64) -> Complex64 {
    let sin_pi_b = (Complex64::new(PI, 0.0) * b).sin();
    if sin_pi_b.norm() < 1e-7 {
        let step = Complex64::new(1e-5, 0.0);
        return 0.5 * (tricomi_u(a, b - step, z) + tricomi_u(a, b + step, z));
    }

    Complex64::new(PI, 0.0) / sin_pi_b
        * (kummer_m(a, b, z)
            * reciprocal_gamma(Complex64::new(1.0, 0.0) + a - b)
            * reciprocal_gamma(b)
            - z.powc(Complex64::new(1.0, 0.0) - b)
                * kummer_m(
                    Complex64::new(1.0, 0.0) + a - b,
                    Complex64::new(2.0, 0.0) - b,
                    z,
                )
                * reciprocal_gamma(a)
                * reciprocal_gamma(Complex64::new(2.0, 0.0) - b))
}

pub(crate) fn zeta_real(s: f64) -> f64 {
    assert!(s != 1.0, "zeta has a pole at s = 1");
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

pub(crate) fn integrate(
    function: impl Fn(f64) -> f64,
    lower: f64,
    upper: f64,
    tolerance: f64,
    max_intervals: usize,
) -> Result<f64, &'static str> {
    if lower == upper {
        return Ok(0.0);
    }
    if upper < lower {
        return integrate(function, upper, lower, tolerance, max_intervals).map(|value| -value);
    }

    let mut total = 0.0;
    let mut intervals = vec![(lower, upper, tolerance)];
    let mut visited = 0;
    while let Some((left, right, local_tolerance)) = intervals.pop() {
        visited += 1;
        if visited > max_intervals {
            return Err("adaptive quadrature exceeded its interval limit");
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
        let m = kummer_m(a, b, z);
        assert!((m - Complex64::new(1.3922407178086509, 1.6474537115932955)).norm() < 2e-13);
        let u = tricomi_u(a, b, z);
        assert!((u - Complex64::new(0.36857118978677744, -0.22472938298119105)).norm() < 2e-12);
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
        let polynomial = integrate(|x| x * x, 0.0, 1.0, 1e-14, 1000).unwrap();
        assert!((polynomial - 1.0 / 3.0).abs() < 2e-14);

        let peaked = integrate(|x| x.sqrt(), 0.0, 1.0, 1e-13, 4000).unwrap();
        assert!((peaked - 2.0 / 3.0).abs() < 2e-13);
    }
}
