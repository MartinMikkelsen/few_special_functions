#![allow(clippy::excessive_precision)]
use few_special_functions::clausen::{
    ci_complex, clausen, clausen_n20, clausen_with_options, f_clausen, f_n,
};
use num_complex::Complex;
use std::f64::consts::PI;

fn parse_data(s: &str) -> Vec<(f64, f64)> {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut cols = l.split_whitespace();
            let x: f64 = cols.next().unwrap().parse().unwrap();
            let y: f64 = cols.next().unwrap().parse().unwrap();
            (x, y)
        })
        .collect()
}

fn check(n: usize, theta: f64, got: f64, expected: f64, atol: f64) {
    assert!(
        (got - expected).abs() <= atol,
        "Cl_{n}({theta:.6}): got {got:.15}, expected {expected:.15}, diff = {:.2e}",
        (got - expected).abs()
    );
}

// Scale-aware: Cl_n has zeros inside (0, 2π), so a pure relative check is
// meaningless there — floor the scale instead of skipping those points.
fn check_rel(n: usize, theta: f64, got: f64, expected: f64, rtol: f64) {
    let scale = expected.abs().max(1e-2);
    assert!(
        (got - expected).abs() <= rtol * scale,
        "Cl_{n}({theta:.6}): got {got:.15}, expected {expected:.15}, rel = {:.2e}",
        (got - expected).abs() / scale
    );
}

const DATA: [(usize, &str); 5] = [
    (2, include_str!("test_data/Cl2.txt")),
    (3, include_str!("test_data/Cl3.txt")),
    (4, include_str!("test_data/Cl4.txt")),
    (5, include_str!("test_data/Cl5.txt")),
    (6, include_str!("test_data/Cl6.txt")),
];

// --- n = 1 (exact formula -ln|2 sin(θ/2)|) ---

#[test]
fn cl1_data() {
    let data = parse_data(include_str!("test_data/Cl1.txt"));
    for (theta, expected) in data {
        if expected.is_infinite() {
            continue;
        }
        check(1, theta, clausen(1, theta), expected, 1e-9);
    }
}

#[test]
fn cl1_singularity() {
    assert!(clausen(1, 0.0).is_infinite());
    assert!(clausen(1, 1e-16).is_infinite());
}

#[test]
fn cl1_symmetry() {
    // Even symmetry: Cl₁(-θ) = Cl₁(θ); periodicity: Cl₁(θ + 2π) = Cl₁(θ)
    for &t in &[0.5_f64, 1.0, 2.0, PI / 3.0] {
        let v = clausen(1, t);
        assert!((clausen(1, -t) - v).abs() < 1e-12, "symmetry failed at {t}");
        assert!(
            (clausen(1, t + 2.0 * PI) - v).abs() < 1e-12,
            "periodicity failed at {t}"
        );
    }
}

// --- n = 2 ---

#[test]
fn cl2_data() {
    let data = parse_data(include_str!("test_data/Cl2.txt"));
    for (theta, expected) in data {
        check(2, theta, clausen(2, theta), expected, 1e-12);
    }
}

#[test]
fn cl2_special() {
    // Cl₂(π/2) = Catalan's constant G
    assert!((clausen(2, PI / 2.0) - 0.9159655941772190).abs() < 1e-10);
    // Cl₂(π) = 0
    assert!(clausen(2, PI).abs() < 1e-10);
}

// --- n = 3 ---

#[test]
fn cl3_data() {
    let data = parse_data(include_str!("test_data/Cl3.txt"));
    for (theta, expected) in data {
        check(3, theta, clausen(3, theta), expected, 1e-12);
    }
}

#[test]
fn cl3_at_zero() {
    // Cl₃(0) = ζ(3) (Apéry's constant)
    assert!((clausen(3, 0.0) - 1.2020569031595942854).abs() < 1e-10);
}

// --- n = 4 ---

#[test]
fn cl4_data() {
    let data = parse_data(include_str!("test_data/Cl4.txt"));
    for (theta, expected) in data {
        check(4, theta, clausen(4, theta), expected, 1e-12);
    }
}

// --- n = 5 ---

#[test]
fn cl5_data() {
    let data = parse_data(include_str!("test_data/Cl5.txt"));
    for (theta, expected) in data {
        check(5, theta, clausen(5, theta), expected, 1e-12);
    }
}

#[test]
fn cl5_at_zero() {
    // Cl₅(0) = ζ(5)
    assert!((clausen(5, 0.0) - 1.0369277551433699341).abs() < 1e-10);
}

// --- n = 6 ---

#[test]
fn cl6_data() {
    let data = parse_data(include_str!("test_data/Cl6.txt"));
    for (theta, expected) in data {
        check(6, theta, clausen(6, theta), expected, 1e-12);
    }
}

// --- N = 20 quadrature table ---

// The N = 20 nodes/weights are used for every order, but only n = 2 was covered,
// and an absolute 1e-8 tolerance on values of order 1e-3..1 hid a relative error
// six orders of magnitude larger than the docstring's ~1e-14 promise.
#[test]
fn cl_data_n20() {
    for (n, raw) in DATA {
        for (theta, expected) in parse_data(raw) {
            check_rel(n, theta, clausen_n20(n, theta), expected, 1e-11);
        }
    }
}

// Reference-free: both entry points evaluate the same Euler-Maclaurin formula, so
// a single corrupt node or weight in either table shows up as a disagreement.
#[test]
fn n10_and_n20_agree() {
    for n in 2..=6 {
        for k in 1..=1000 {
            let theta = k as f64 * 2.0 * PI / 1001.0;
            let (a, b) = (clausen(n, theta), clausen_n20(n, theta));
            assert!(
                (a - b).abs() <= 1e-12 * a.abs().max(1e-2),
                "Cl_{n}({theta:.6}): N=10 gives {a:.15e}, N=20 gives {b:.15e}"
            );
        }
    }
}

#[test]
fn julia_public_helpers_are_available() {
    let ci = ci_complex(Complex::new(1.0, 0.0));
    assert!((ci.re - 0.3374039229009681).abs() < 1e-10);
    assert!(ci.im.abs() < 1e-14);

    assert!((f_n(2, 2, 0.7) - 1.4_f64.sin() / 4.0).abs() < 1e-15);
    assert!((f_n(3, 2, 0.7) - 1.4_f64.cos() / 8.0).abs() < 1e-15);

    let z = Complex::new(2.0, 0.5);
    assert!((f_clausen(1, z, 0.7) - ci_complex(z * 0.7)).norm() < 1e-14);
}

#[test]
fn julia_clausen_options_are_available() {
    let theta = std::f64::consts::PI / 3.0;
    assert_eq!(clausen_with_options(2, theta, 10, 20), clausen(2, theta));
    assert_eq!(
        clausen_with_options(2, theta, 20, 20),
        clausen_n20(2, theta)
    );
}
