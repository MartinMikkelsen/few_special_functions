mod helpers;

use few_special_functions::marcum_q::{dq_db, marcum_q};
use helpers::check;

// --- Spot checks from published tables ---

#[test]
fn acta_univ_sapientiae() {
    check(
        "Q(1,0.2,0.6)",
        marcum_q(1.0, 0.2, 0.6),
        0.838249985438908,
        1e-9,
    );
    check(
        "Q(5,0.2,0.6)",
        marcum_q(5.0, 0.2, 0.6),
        0.999998670306184,
        1e-9,
    );
    check(
        "Q(7.7,0.2,0.6)",
        marcum_q(7.7, 0.2, 0.6),
        0.999999999927717,
        1e-9,
    );
    check(
        "Q(1,1.2,1.6)",
        marcum_q(1.0, 1.2, 1.6),
        0.501536568390858,
        1e-9,
    );
    check(
        "Q(5,1.2,1.6)",
        marcum_q(5.0, 1.2, 1.6),
        0.994346394491553,
        1e-9,
    );
    check(
        "Q(7.7,1.2,1.6)",
        marcum_q(7.7, 1.2, 1.6),
        0.99994493722354,
        1e-9,
    );
    check(
        "Q(1,2.2,2.6)",
        marcum_q(1.0, 2.2, 2.6),
        0.426794627821735,
        1e-9,
    );
    check(
        "Q(5,2.2,2.6)",
        marcum_q(5.0, 2.2, 2.6),
        0.929671935077756,
        1e-9,
    );
    check(
        "Q(7.7,2.2,2.6)",
        marcum_q(7.7, 2.2, 2.6),
        0.993735633182201,
        1e-9,
    );
}

#[test]
fn matlab_reference() {
    check(
        "Q(5,0.47,4.85)",
        marcum_q(5.0, 0.47, 4.85),
        0.0106766402997493,
        1e-8,
    );
    check(
        "Q(5,1.46,4.0)",
        marcum_q(5.0, 1.46, 4.0),
        0.211798804811782,
        1e-8,
    );
    check(
        "Q(1,1.27,4.58)",
        marcum_q(1.0, 1.27, 4.58),
        0.000931257801666407,
        1e-8,
    );
    check(
        "Q(4,2.88,3.28)",
        marcum_q(4.0, 2.88, 3.28),
        0.773155207859263,
        1e-8,
    );
    check(
        "Q(1,2.55,4.67)",
        marcum_q(1.0, 2.55, 4.67),
        0.024112315799424,
        1e-8,
    );
    check(
        "Q(4,2.27,3.72)",
        marcum_q(4.0, 2.27, 3.72),
        0.400088995953665,
        1e-8,
    );
    check(
        "Q(2,1.97,0.86)",
        marcum_q(2.0, 1.97, 0.86),
        0.990345203236692,
        1e-8,
    );
    check(
        "Q(4,0.1,1.38)",
        marcum_q(4.0, 0.1, 1.38),
        0.983869651076909,
        1e-8,
    );
    check(
        "Q(1,0.29,4.12)",
        marcum_q(1.0, 0.29, 4.12),
        0.00028475422733874,
        1e-8,
    );
    check(
        "Q(4,0.95,4.75)",
        marcum_q(4.0, 0.95, 4.75),
        0.00899422673877906,
        1e-8,
    );
}

// --- Full 1500-point reference file (path relative to this file) ---

#[test]
fn data_file() {
    let data = helpers::parse_reference(include_str!("../test_data/marcumq_test.txt"));
    let mut max_err = 0.0_f64;
    for (m, a, b, expected) in data {
        let got = marcum_q(m, a, b);
        let err = (got - expected).abs() / expected.abs().max(1e-30);
        max_err = max_err.max(err);
        assert!(
            err <= 1e-9,
            "Q({m},{a},{b}): got {got:.15}, expected {expected:.15}, rtol={err:.2e}"
        );
    }
    println!("max relative error across 1500 points: {max_err:.2e}");
}

// --- dQdb ---

#[test]
fn dq_db_sign_and_range() {
    for &b in &[0.5_f64, 1.0, 2.0, 3.0] {
        let d = dq_db(1, 1.0, b);
        assert!(d < 0.0, "dQ/db should be negative at b={b}, got {d}");
    }
}

#[test]
#[should_panic]
fn invalid_mu() {
    marcum_q(0.3, 1.0, 1.0);
}

#[test]
#[should_panic]
fn invalid_a() {
    marcum_q(1.0, -1.0, 1.0);
}

#[test]
#[should_panic]
fn invalid_b() {
    marcum_q(1.0, 1.0, -1.0);
}
