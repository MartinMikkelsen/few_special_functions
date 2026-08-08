// Ported from the Julia test suite (test/test_Voigt.jl) of
// FewSpecialFunctions.jl. The Julia suite builds its reference values on the fly
// with a 256-bit QuadGK evaluation of the transformed integral; Rust has no
// QuadGK, so those same references are embedded here as constants computed once
// from the Julia code (setprecision(BigFloat, 256), rtol = 1e-60 on [0, 30]).

use few_special_functions::voigt::voigt;
use std::f64::consts::PI;

fn approx(got: f64, expected: f64, rtol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= rtol * expected.abs(),
        "got {got:.17}, expected {expected:.17}, rel diff = {:.2e}",
        diff / expected.abs()
    );
}

#[test]
fn exact_special_values() {
    assert_eq!(voigt(0.0, 0.0), 1.0);
    assert_eq!(voigt(2.0, 0.0), (-4.0_f64).exp());
    // K is even in x.
    assert_eq!(voigt(2.0, 0.5), voigt(-2.0, 0.5));
}

#[test]
#[should_panic(expected = "y ≥ 0")]
fn negative_y_is_a_domain_error() {
    voigt(1.0, -0.1);
}

#[test]
fn quadgk_reference_points() {
    // (x, y, K(x, y)) — 256-bit QuadGK references from the Julia suite.
    let reference = [
        (0.0, 0.1, 0.8964569799691267),
        (1.0, 0.1, 0.37317014831126744),
        (3.0, 0.01, 0.0009088307067415805),
        (6.0, 4.0, 0.04414092342364238),
    ];
    for (x, y, expected) in reference {
        approx(voigt(x, y), expected, 2.0e-8);
    }
}

#[test]
fn narrow_y_reference_points() {
    let reference = [
        (1e-12, 1e-10, 0.999999999887162),
        (PI / 12.0 + 1e-12, 1e-10, 0.933757117982429),
    ];
    for (x, y, expected) in reference {
        approx(voigt(x, y), expected, 2.0e-12);
    }
}

#[test]
fn large_y_asymptote() {
    // For y → ∞, K(x, y) → 1 / (√π · y).
    approx(voigt(1.0, 1e200), 1.0 / (PI.sqrt() * 1e200), 2.0e-12);
}

#[test]
fn nan_and_inf_propagation() {
    assert!(voigt(f64::NAN, 0.5).is_nan());
    assert!(voigt(0.5, f64::NAN).is_nan());
    assert_eq!(voigt(f64::INFINITY, 0.5), 0.0);
    assert_eq!(voigt(0.5, f64::INFINITY), 0.0);
}
