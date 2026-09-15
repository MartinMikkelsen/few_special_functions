#![allow(clippy::excessive_precision)]

use few_special_functions::bose_einstein::{bose_einstein_integral, bose_einstein_integral_norm};

fn check(k: f64, eta: f64, expected: f64, relative_tolerance: f64) {
    let actual = bose_einstein_integral_norm(k, eta);
    let error = (actual - expected).abs() / expected.abs();
    assert!(
        error <= relative_tolerance,
        "B_{k}({eta}): expected {expected:.16e}, got {actual:.16e}, relative error {error:.3e}"
    );
}

#[test]
fn values_match_julia() {
    check(0.5, -1.0, 0.4284407345998379, 8e-15);
    check(-4.5, -0.75, 42.454973112334443, 2e-13);
    check(1.5, -1.0, 0.3957280103803376, 8e-15);
    check(20.0, -1.0, 0.367879505709098, 8e-15);

    let unnormalized = bose_einstein_integral(1.5, -1.0);
    assert!((unnormalized - libm::tgamma(2.5) * 0.3957280103803376).abs() < 2e-15);
}

#[test]
fn elementary_orders_and_endpoints() {
    assert!((bose_einstein_integral_norm(0.0, -1e-20) + 1e-20_f64.ln()).abs() < 4e-15);
    assert!((bose_einstein_integral_norm(-1.0, -1e-20) / 1e20 - 1.0).abs() < 4e-15);
    assert_eq!(bose_einstein_integral_norm(1.0, f64::NEG_INFINITY), 0.0);
    assert_eq!(bose_einstein_integral_norm(-0.5, 0.0), f64::INFINITY);
    assert_eq!(bose_einstein_integral(0.0, 0.0), f64::INFINITY);
    assert!(
        (bose_einstein_integral_norm(1.0, 0.0) - std::f64::consts::PI.powi(2) / 6.0).abs() < 3e-15
    );
}

#[test]
#[should_panic]
fn rejects_non_half_integer_order() {
    bose_einstein_integral_norm(0.25, -1.0);
}

#[test]
#[should_panic]
fn rejects_positive_eta() {
    bose_einstein_integral_norm(0.5, 0.1);
}

#[test]
#[should_panic]
fn unnormalized_rejects_continued_orders() {
    bose_einstein_integral(-1.0, -1.0);
}
