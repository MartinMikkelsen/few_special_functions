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

// High branch (x >= 2), which the checks above barely touch. Antia (1993) promises
// relative error < 1e-12 here for every half-integer index. Reference values are
// F_j(x) = -Γ(j+1)·Li_{j+1}(-eˣ) evaluated with mpmath (dps=50), cross-checked to
// agree with direct quadrature of the defining integral to < 1e-20.
#[test]
fn high_branch_matches_reference() {
    let cases = [
        (-0.5, 2.0, 2.5953945832884786),
        (-0.5, 3.0, 3.2852167828877117),
        (-0.5, 5.0, 4.383256434711571),
        (-0.5, 10.0, 6.297137244533848),
        (-0.5, 20.0, 8.93497266616697),
        (0.5, 2.0, 2.50245782600714),
        (0.5, 3.0, 3.9769853540479776),
        (0.5, 5.0, 7.837976057293097),
        (0.5, 10.0, 21.344471492355183),
        (0.5, 20.0, 59.812795370358025),
        (1.5, 2.0, 5.537253675008345),
        (1.5, 3.0, 10.353714864761452),
        (1.5, 5.0, 27.80244621574838),
        (1.5, 10.0, 134.27015996313986),
        (1.5, 20.0, 726.5682839651753),
        (2.5, 2.0, 17.529419187384264),
        (2.5, 3.0, 36.932065413293266),
        (2.5, 5.0, 127.48954491322323),
        (2.5, 10.0, 1034.6842541815338),
        (2.5, 20.0, 10590.639176614386),
    ];
    for (j, x, expected) in cases {
        let got = fermi_dirac_integral(j, x);
        let rel = (got - expected).abs() / expected.abs();
        assert!(
            rel < 1e-9,
            "F_{j}({x}): got {got:.15e}, expected {expected:.15e}, rel = {rel:.2e}"
        );
    }
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
