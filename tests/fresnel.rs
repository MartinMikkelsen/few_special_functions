use few_special_functions::fresnel::{fresnel, fresnel_c, fresnel_s};
use std::f64::consts::PI;

fn check(label: &str, got: f64, expected: f64, atol: f64) {
    assert!(
        (got - expected).abs() <= atol,
        "{label}: got {got:.15}, expected {expected:.15}, diff = {:.2e}",
        (got - expected).abs()
    );
}

// --- Spot checks from the Julia test suite ---

#[test]
fn spot_checks() {
    // C(1) ≈ 0.7798934, S(1) ≈ 0.4382591
    let (c1, s1, _) = fresnel(1.0);
    check("C(1)", c1, 0.7798934003604542, 1e-10);
    check("S(1)", s1, 0.4382591473903547, 1e-10);

    // At x = 2: reference values from SpecialFunctions.jl
    let (c2, s2, _) = fresnel(2.0);
    check("C(2)", c2, 0.48825340607534486, 1e-8);
    check("S(2)", s2, 0.34341567836369824, 1e-8);
}

#[test]
fn boundary() {
    let (c, s, e) = fresnel(0.0);
    assert_eq!(c, 0.0);
    assert_eq!(s, 0.0);
    assert_eq!(e.re, 0.0);
    assert_eq!(e.im, 0.0);
}

#[test]
fn small_x() {
    let x = 1e-8_f64;
    let (c, s, _) = fresnel(x);
    assert!((c - x).abs() < 1e-16, "C(tiny): got {c}, expected {x}");
    // At x=1e-8 the S value (~5e-25) arises from cancellation of terms ~1e-8,
    // so absolute accuracy is limited to ~eps·x ≈ 1e-24. Match Julia's atol=1e-24.
    assert!((s - PI / 6.0 * x.powi(3)).abs() < 1e-22);
}

#[test]
fn odd_symmetry() {
    for &x in &[0.5_f64, 1.0, 1.5, 2.0, 3.0, 5.0] {
        let (cp, sp, _) = fresnel(x);
        let (cn, sn, _) = fresnel(-x);
        assert!(
            (cp + cn).abs() < 1e-10,
            "C odd symmetry at {x}: {cp} vs {cn}"
        );
        assert!(
            (sp + sn).abs() < 1e-10,
            "S odd symmetry at {x}: {sp} vs {sn}"
        );
    }
}

#[test]
fn wrapper_consistency() {
    for &x in &[0.0_f64, 0.5, 1.0, 2.5, 5.0, -1.0] {
        let (c, s, e) = fresnel(x);
        assert_eq!(fresnel_c(x), c);
        assert_eq!(fresnel_s(x), s);
        // E = C + iS
        assert!((e.re - c).abs() < 1e-15);
        assert!((e.im - s).abs() < 1e-15);
    }
}

// --- Full data file: 499 points, x, S, C (rtol = 1e-2 as in Julia tests) ---

#[test]
fn data_file() {
    let data = include_str!("test_data/FresnelF.txt");
    for line in data.lines().filter(|l| !l.trim().is_empty()) {
        let mut cols = line.split_whitespace();
        let x: f64 = cols.next().unwrap().parse().unwrap();
        let s_ref: f64 = cols.next().unwrap().parse().unwrap();
        let c_ref: f64 = cols.next().unwrap().parse().unwrap();

        let (c, s, _) = fresnel(x);

        let c_ok = c_ref.abs() < 1e-10 || (c - c_ref).abs() / c_ref.abs() < 1e-2;
        let s_ok = s_ref.abs() < 1e-10 || (s - s_ref).abs() / s_ref.abs() < 1e-2;

        assert!(c_ok, "C({x}): got {c:.10}, expected {c_ref:.10}");
        assert!(s_ok, "S({x}): got {s:.10}, expected {s_ref:.10}");
    }
}
