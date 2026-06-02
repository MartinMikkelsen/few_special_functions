use few_special_functions::fermi_dirac::{fermi_dirac_integral, fermi_dirac_integral_norm};
use std::f64::consts::E;

fn check(j: f64, x: f64, got: f64, expected: f64, atol: f64) {
    assert!(
        (got - expected).abs() <= atol,
        "F_{j}({x}): got {got:.15}, expected {expected:.15}, diff = {:.2e}",
        (got - expected).abs()
    );
}

// j = 0 is exact: F_0(x) = ln(1 + exp(x))
#[test]
fn j0_exact() {
    check(
        0.0,
        0.0,
        fermi_dirac_integral(0.0, 0.0),
        (2.0_f64).ln(),
        1e-15,
    );
    check(
        0.0,
        2.0,
        fermi_dirac_integral(0.0, 2.0),
        (1.0 + E * E).ln(),
        1e-14,
    );
    check(
        0.0,
        -3.0,
        fermi_dirac_integral(0.0, -3.0),
        (1.0 + (-3.0_f64).exp()).ln(),
        1e-15,
    );
}

// Reference values from https://npplus.readthedocs.io/en/latest/fermi.html
// and https://github.com/scott-maddox/fdint
#[test]
fn j_neg_half() {
    check(
        -0.5,
        0.0,
        fermi_dirac_integral(-0.5, 0.0),
        1.0721549299400754,
        1e-12,
    );
    check(
        -0.5,
        1.0,
        fermi_dirac_integral(-0.5, 1.0),
        1.8204113571471041,
        1e-12,
    );
    check(
        -0.5,
        1.2,
        fermi_dirac_integral(-0.5, 1.2),
        1.9785617633438695,
        1e-12,
    );
    check(
        -0.5,
        4.1,
        fermi_dirac_integral(-0.5, 4.1),
        3.928454737099184,
        1e-10,
    );
    check(
        -0.5,
        5.2,
        fermi_dirac_integral(-0.5, 5.2),
        4.477432715418454,
        1e-10,
    );
    check(
        -0.5,
        6.6,
        fermi_dirac_integral(-0.5, 6.6),
        5.082787981164429,
        1e-10,
    );
}

#[test]
fn j_half() {
    check(
        0.5,
        0.0,
        fermi_dirac_integral(0.5, 0.0),
        0.6780938951530457,
        1e-12,
    );
    check(
        0.5,
        1.0,
        fermi_dirac_integral(0.5, 1.0),
        1.3963752806666279,
        1e-12,
    );
    check(
        0.5,
        1.2,
        fermi_dirac_integral(0.5, 1.2),
        1.5863233997463857,
        1e-12,
    );
    check(
        0.5,
        4.1,
        fermi_dirac_integral(0.5, 4.1),
        5.965800008889902,
        1e-4,
    );
    check(
        0.5,
        5.2,
        fermi_dirac_integral(0.5, 5.2),
        8.28102917922544,
        1e-4,
    );
    check(
        0.5,
        6.6,
        fermi_dirac_integral(0.5, 6.6),
        11.632113406633252,
        1e-4,
    );
}

#[test]
fn j_three_half() {
    check(
        1.5,
        0.0,
        fermi_dirac_integral(1.5, 0.0),
        1.15280383708879,
        1e-12,
    );
    check(
        1.5,
        1.0,
        fermi_dirac_integral(1.5, 1.0),
        2.6616826247307124,
        1e-10,
    );
    check(
        1.5,
        1.2,
        fermi_dirac_integral(1.5, 1.2),
        3.10869199517456,
        1e-10,
    );
    check(
        1.5,
        -2.5,
        fermi_dirac_integral(1.5, -2.5),
        0.1075808743944384,
        1e-12,
    );
    check(
        1.5,
        -2.0,
        fermi_dirac_integral(1.5, -2.0),
        0.175800988853926,
        1e-12,
    );
}

#[test]
fn j_five_half() {
    check(
        2.5,
        0.0,
        fermi_dirac_integral(2.5, 0.0),
        3.0825860828379246,
        1e-12,
    );
    check(
        2.5,
        1.0,
        fermi_dirac_integral(2.5, 1.0),
        7.626535355004442,
        1e-10,
    );
    check(
        2.5,
        1.2,
        fermi_dirac_integral(2.5, 1.2),
        9.066754659807005,
        1e-12,
    );
    check(
        2.5,
        -2.5,
        fermi_dirac_integral(2.5, -2.5),
        0.27085618652992494,
        1e-12,
    );
    check(
        2.5,
        -2.0,
        fermi_dirac_integral(2.5, -2.0),
        0.4445544534586879,
        1e-12,
    );
}

#[test]
fn normalised() {
    // F̃_j(x) = F_j(x) / Γ(j+1) — spot check a few values
    let v = fermi_dirac_integral_norm(0.5, 0.0);
    // F_{1/2}(0) / Γ(3/2) = 0.6781 / (√π/2) ≈ 0.7652
    assert!((v - 0.765147).abs() < 1e-5, "got {v}");
}

#[test]
#[should_panic]
fn invalid_order() {
    fermi_dirac_integral(-1.0, 0.0);
}
