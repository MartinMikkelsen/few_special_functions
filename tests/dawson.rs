// Ported from the Julia test suite (test/test_Dawson.jl) of
// FewSpecialFunctions.jl. Type-genericity cases (Float32/BigFloat/Rational)
// are omitted because the Rust port is f64-only; the region tests that poke the
// private `_dawson_*` helpers live in the unit tests inside src/dawson.rs.

use few_special_functions::dawson::dawson;

fn approx(got: f64, expected: f64, rtol: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff <= rtol * expected.abs().max(f64::MIN_POSITIVE),
        "got {got:.17}, expected {expected:.17}, rel diff = {:.2e}",
        diff / expected.abs()
    );
}

#[test]
fn reference_values() {
    let reference = [
        (0.0, 0.0),
        (0.1, 0.09933599239785286),
        (0.5, 0.4244363835020223),
        (1.0, 0.5380795069127684),
        (2.0, 0.30134038892379197),
        (5.0, 0.10213407442427684),
        (10.0, 0.05025384718759853),
        (100.0, 0.005000250037509379),
    ];
    for (x, expected) in reference {
        if expected == 0.0 {
            assert_eq!(dawson(x), 0.0);
        } else {
            approx(dawson(x), expected, 2.0e-14);
        }
    }
}

#[test]
fn special_values() {
    assert_eq!(dawson(0.0), 0.0);
    assert!(dawson(-0.0).is_sign_negative());
    assert_eq!(dawson(-2.0), -dawson(2.0));
    assert!(dawson(f64::NAN).is_nan());
    assert_eq!(dawson(f64::INFINITY), 0.0);
}

#[test]
fn numerical_regions() {
    approx(dawson(1.0e-10), 1.0e-10, 2.0e-14);
    approx(dawson(20.0), 0.02503136792640367, 2.0e-14);
    assert_eq!(dawson(1.0e12), 5.0e-13);
}

#[test]
fn satisfies_defining_ode() {
    // D'(x) = 1 - 2x·D(x); check against a central difference.
    for &x in &[0.1_f64, 1.0, 10.0] {
        let h = 1.0e-6;
        let derivative = (dawson(x + h) - dawson(x - h)) / (2.0 * h);
        let expected = (-2.0 * x).mul_add(dawson(x), 1.0);
        approx(derivative, expected, 1.0e-7);
    }
}
