#![allow(clippy::excessive_precision)]
use few_special_functions::debye::debye_function;

fn check(n: f64, beta: f64, x: f64, got: f64, expected: f64, atol: f64) {
    assert!(
        (got - expected).abs() <= atol,
        "D_{n}(β={beta}, x={x}): got {got:.15}, expected {expected:.15}, diff = {:.2e}",
        (got - expected).abs()
    );
}

// --- Hand-verified spot checks from the Julia test suite ---

#[test]
fn spot_checks() {
    check(
        2.0,
        1.0,
        5.0,
        debye_function(2.0, 1.0, 5.0),
        0.172329034857624782145,
        1e-6,
    );
    check(
        5.0,
        1.0,
        4.5,
        debye_function(5.0, 1.0, 4.5),
        0.10164118339698890968,
        1e-12,
    );
    check(
        7.0,
        1.0,
        0.8,
        debye_function(7.0, 1.0, 0.8),
        0.69112406526865230673,
        1e-10,
    );
    check(
        9.0,
        1.0,
        3.4,
        debye_function(9.0, 1.0, 3.4),
        0.15413773867789254146,
        1e-12,
    );
    check(
        12.0,
        1.0,
        5.4,
        debye_function(12.0, 1.0, 5.4),
        0.03618849233828133,
        1e-10,
    );
    check(
        15.0,
        1.0,
        2.4,
        debye_function(15.0, 1.0, 2.4),
        0.2661409156647294951955,
        1e-10,
    );
    check(
        20.0,
        1.0,
        1.24,
        debye_function(20.0, 1.0, 1.24),
        0.523361585088859680745,
        1e-9,
    );
    check(
        25.0,
        1.0,
        4.2,
        debye_function(25.0, 1.0, 4.2),
        0.07296496706218587,
        1e-10,
    );
    check(
        30.0,
        1.0,
        3.42,
        debye_function(30.0, 1.0, 3.42),
        0.1258426106590655660781,
        1e-10,
    );
}

#[test]
fn boundary() {
    assert_eq!(debye_function(1.0, 1.0, 0.0), 1.0);
    assert_eq!(debye_function(3.0, 2.5, 0.0), 1.0);
}

#[test]
#[should_panic]
fn invalid_n() {
    debye_function(-1.0, 1.0, 1.0);
}

#[test]
#[should_panic]
fn invalid_beta() {
    debye_function(1.0, -1.0, 1.0);
}

#[test]
#[should_panic]
fn invalid_x() {
    debye_function(1.0, 1.0, -1.0);
}

// --- Full data file from DebyeFunctions.jl comparison ---

#[test]
fn data_file() {
    let data = include_str!("test_data/debye_test.txt");
    for line in data.lines().filter(|l| !l.trim().is_empty()) {
        let mut cols = line.split_whitespace();
        let x: f64 = cols.next().unwrap().parse().unwrap();
        let n: f64 = cols.next().unwrap().parse().unwrap();
        let expected: f64 = cols.next().unwrap().parse().unwrap();
        let got = debye_function(n, 1.0, x);
        check(n, 1.0, x, got, expected, 1e-4);
    }
}
