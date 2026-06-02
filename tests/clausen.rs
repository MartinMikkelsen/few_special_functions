use few_special_functions::clausen::{clausen, clausen_n20};
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
        assert!((clausen(1, t + 2.0 * PI) - v).abs() < 1e-12, "periodicity failed at {t}");
    }
}

// --- n = 2 ---

#[test]
fn cl2_data() {
    let data = parse_data(include_str!("test_data/Cl2.txt"));
    for (theta, expected) in data {
        check(2, theta, clausen(2, theta), expected, 1e-8);
    }
}

#[test]
fn cl2_data_n20() {
    let data = parse_data(include_str!("test_data/Cl2.txt"));
    for (theta, expected) in data {
        check(2, theta, clausen_n20(2, theta), expected, 1e-8);
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
        check(3, theta, clausen(3, theta), expected, 1e-8);
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
        check(4, theta, clausen(4, theta), expected, 1e-8);
    }
}

// --- n = 5 ---

#[test]
fn cl5_data() {
    let data = parse_data(include_str!("test_data/Cl5.txt"));
    for (theta, expected) in data {
        check(5, theta, clausen(5, theta), expected, 1e-8);
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
        check(6, theta, clausen(6, theta), expected, 1e-8);
    }
}
