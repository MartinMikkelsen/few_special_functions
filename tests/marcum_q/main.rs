mod helpers;

use few_special_functions::marcum_q::{dq_db, dq_db_order_one, marcum_q, marcum_q_order_one};
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

#[test]
fn order_one_convenience_functions_match_general_api() {
    assert_eq!(marcum_q_order_one(1.2, 1.6), marcum_q(1.0, 1.2, 1.6));
    assert_eq!(dq_db_order_one(1.2, 1.6), dq_db(1, 1.2, 1.6));
}

#[test]
fn large_order_matches_centered_poisson_gamma_reference() {
    check("Q(150,10,10)", marcum_q(150.0, 10.0, 10.0), 1.0, 2e-14);
    check(
        "Q(150,5,20)",
        marcum_q(150.0, 5.0, 20.0),
        0.0037985009089966685,
        2e-12,
    );
}

// --- Identities that need no reference table ---

#[test]
fn rayleigh_tail_closed_form() {
    // Q_1(0, b) = exp(−b²/2) exactly, for every b.
    for i in 1..=60 {
        let b = i as f64 * 0.5;
        let want = (-b * b / 2.0).exp();
        if want == 0.0 {
            continue;
        }
        check(&format!("Q(1,0,{b})"), marcum_q(1.0, 0.0, b), want, 1e-12);
    }
    // Q_μ(0, b) = Q(μ, b²/2) is likewise a pure incomplete gamma; spot-check μ = ½.
    #[allow(clippy::excessive_precision)]
    let q_half = 1.52397060483210521e-23;
    check("Q(0.5,0,10)", marcum_q(0.5, 0.0, 10.0), q_half, 1e-12);
}

#[test]
fn recurrence_against_dq_db() {
    // Q_{M+1}(a,b) − Q_M(a,b) = (b/a)^M e^{−(a²+b²)/2} I_M(ab) = −dQ/db(M+1,a,b) / b.
    // Reference-free: it ties marcum_q to the crate's own Bessel evaluation. The gap is
    // formed by subtraction, so it is checked against the scale of Q, not of the gap.
    let mut worst = 0.0_f64;
    for &m in &[1u32, 2, 3, 5, 9, 20, 60, 120] {
        for &(a, b) in &[
            (0.5, 1.5),
            (2.0, 3.0),
            (5.0, 4.0),
            (8.0, 10.0),
            (9.0, 12.0),
            (25.0, 26.0),
            (80.0, 83.5),
        ] {
            let q = marcum_q(m as f64, a, b);
            let gap = marcum_q(m as f64 + 1.0, a, b) - q;
            let want = -dq_db(m + 1, a, b) / b;
            if q < 1e-300 {
                continue;
            }
            let err = (gap - want).abs() / q;
            worst = worst.max(err);
            assert!(
                err < 1e-6,
                "recurrence M={m} a={a} b={b}: {gap:.15e} vs {want:.15e}, {err:.2e} of Q"
            );
        }
    }
    println!("worst recurrence residual, as a fraction of Q: {worst:.2e}");
}

#[test]
fn nonincreasing_in_b() {
    // Q_μ(a, ·) is a survival function: it can never rise as b grows.
    for &mu in &[0.5_f64, 1.0, 2.0, 8.0, 35.0, 120.0] {
        for &a in &[0.0_f64, 1.0, 5.0, 8.0, 14.0, 45.0, 80.0] {
            let mut prev = f64::INFINITY;
            for i in 0..=200 {
                let b = (a + 40.0) * i as f64 / 200.0;
                let q = marcum_q(mu, a, b);
                assert!(
                    q <= prev + 1e-15 && (0.0..=1.0).contains(&q),
                    "Q({mu},{a},{b}) = {q} rose above the previous {prev}"
                );
                prev = q;
            }
        }
    }
}

// --- Branch coverage: the reference file above only ever reaches the x < 30 series ---

#[test]
fn all_dispatch_branches_against_reference_values() {
    #[allow(clippy::excessive_precision)]
    let refs = [
        // series, x < 30
        (0.5, 7.0, 8.660254037844387, 0.0484316789098623483),
        (1.0, 7.0, 10.0, 0.00163884620575779567),
        (1.0, 3.0, 12.0, 2.27526516094407434e-19),
        (3.0, 5.0, 14.0, 1.46972826392732899e-18),
        (10.0, 2.0, 15.0, 3.42141032486021123e-31),
        (2.0, 7.7, 11.0, 0.000851685854546712275),
        // asymptotic expansion for large ξ
        (1.0, 8.0, 8.12403840463596, 0.475337993674657251),
        (1.5, 8.0, 8.0, 0.549867785050179085),
        (2.0, 10.0, 10.0995, 0.519940453477264467),
        (20.0, 14.0, 14.9733, 0.645614534463318241),
        (8.0, 12.0, 12.5, 0.544943543388039862),
        (1.0, 100.0, 101.0, 0.159862112904856357),
        (50.0, 45.0, 46.0, 0.535059171291818644),
        // three-term recurrence
        (120.0, 80.0, 81.4862, 0.497542818691657467),
        (134.0, 60.0, 62.1932, 0.496793792196779259),
        (20.0, 8.0, 10.198, 0.479955097571850733),
        // quadrature
        (60.0, 25.0, 60.0, 1.52193388607305412e-246),
        (400.0, 80.0, 100.0, 8.58261368910364303e-55),
        (150.0, 80.0, 100.0, 2.28661517078968935e-75),
        (500.0, 50.0, 60.6218, 0.0565239946319431525),
    ];
    for (mu, a, b, want) in refs {
        check(&format!("Q({mu},{a},{b})"), marcum_q(mu, a, b), want, 1e-6);
    }
}
