#![allow(clippy::excessive_precision)]

/// Assert relative error ≤ rtol, printing a descriptive failure message.
pub fn check(label: &str, got: f64, expected: f64, rtol: f64) {
    let err = (got - expected).abs() / expected.abs().max(1e-30);
    assert!(
        err <= rtol,
        "{label}: got {got:.15}, expected {expected:.15}, rtol = {err:.2e}"
    );
}

/// Parse the four-column reference data file  (M  a  b  Q_ref).
pub fn parse_reference(content: &str) -> Vec<(f64, f64, f64, f64)> {
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut c = l.split_whitespace();
            let m: f64 = c.next().unwrap().parse().unwrap();
            let a: f64 = c.next().unwrap().parse().unwrap();
            let b: f64 = c.next().unwrap().parse().unwrap();
            let q: f64 = c.next().unwrap().parse().unwrap();
            (m, a, b, q)
        })
        .collect()
}
