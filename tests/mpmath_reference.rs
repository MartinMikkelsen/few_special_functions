//! Accuracy against the mpmath tables in `tests/reference/`.
//!
//! Every check asserts a stated tolerance on the relative error (see
//! `reference::error`). Run with `--nocapture` to print the measured maximum
//! and median error of each check, and SciPy's error on the same points where
//! SciPy has the function:
//!
//! ```text
//! cargo test --release --test mpmath_reference -- --nocapture --test-threads=1
//! ```
//!
//! Tests marked `#[ignore]` are known accuracy gaps: they assert
//! [`GAP_TARGET`] on regions where the implementation is currently less
//! accurate, and the ignore reason gives the measured error. Run them with
//! `-- --ignored`; they fail until the gap is closed.

mod reference;

use few_special_functions::{
    bose_einstein, clausen, coulomb, dawson, debye, fermi_dirac, fresnel, marcum_q,
    parabolic_cylinder as pcf, voigt, whittaker,
};
use reference::{Check, Report, Row, Table};

/// A function of two or three real arguments, as listed in the check tables below.
type Real2 = fn(f64, f64) -> f64;
type Real3 = fn(f64, f64, f64) -> f64;

/// Tolerance asserted by the known-gap tests.
const GAP_TARGET: f64 = 1e-12;

macro_rules! table {
    ($name:literal) => {
        Table::parse($name, include_str!(concat!("reference/", $name, ".tsv")))
    };
}

#[test]
fn clausen_functions() {
    let table = table!("clausen");
    let mut report = Report::default();
    Check::new(&table, "Cl_n(theta)", &["n", "theta"], "value", 5e-14).run(&mut report, |row| {
        clausen::clausen(row.get("n") as usize, row.get("theta"))
    });
    report.finish();
}

#[test]
fn fermi_dirac_integrals() {
    let table = table!("fermi_dirac");
    let inputs = &["j", "x"];
    let mut report = Report::default();
    Check::new(&table, "F_j(x)", inputs, "value", 2e-14).run(&mut report, |row| {
        fermi_dirac::fermi_dirac_integral(row.get("j"), row.get("x"))
    });
    Check::new(&table, "F_j(x)/Gamma(j+1)", inputs, "norm", 2e-14).run(&mut report, |row| {
        fermi_dirac::fermi_dirac_integral_norm(row.get("j"), row.get("x"))
    });
    report.finish();
}

#[test]
fn bose_einstein_integrals() {
    let table = table!("bose_einstein");
    let inputs = &["k", "eta"];
    let mut report = Report::default();
    Check::new(&table, "Gamma(k+1) Li_{k+1}(e^eta)", inputs, "value", 5e-15)
        .run(&mut report, |row| {
            bose_einstein::bose_einstein_integral(row.get("k"), row.get("eta"))
        });
    Check::new(&table, "Li_{k+1}(e^eta)", inputs, "norm", 5e-15).run(&mut report, |row| {
        bose_einstein::bose_einstein_integral_norm(row.get("k"), row.get("eta"))
    });
    report.finish();
}

#[test]
fn debye_functions() {
    let table = table!("debye");
    let mut report = Report::default();
    Check::new(&table, "D_n(beta, x)", &["n", "beta", "x"], "value", 2e-15)
        .run(&mut report, |row| {
            debye::debye_function(row.get("n"), row.get("beta"), row.get("x"))
        });
    report.finish();
}

#[test]
fn dawson_integral() {
    let table = table!("dawson");
    let mut report = Report::default();
    Check::new(&table, "D(x)", &["x"], "value", 5e-15)
        .parity("scipy")
        .run(&mut report, |row| dawson::dawson(row.get("x")));
    report.finish();
}

/// Everywhere except the far wing at small width, `|x| >= 5` with `y <= 0.01`.
fn voigt_core(row: &Row) -> bool {
    row.get("x").abs() < 5.0 || row.get("y") > 0.01
}

#[test]
fn voigt_function() {
    let table = table!("voigt");
    let mut report = Report::default();
    Check::new(&table, "K(x, y)", &["x", "y"], "value", 5e-13)
        .domain("|x| < 5 or y > 0.01", voigt_core)
        .parity("scipy")
        .run(&mut report, |row| voigt::voigt(row.get("x"), row.get("y")));
    report.finish();
}

#[test]
#[ignore = "known accuracy gap: relative error up to 1.5e-6 in the far wing (x = 30, y = 1e-8)"]
fn voigt_function_far_wing() {
    let table = table!("voigt");
    let mut report = Report::default();
    Check::new(&table, "K(x, y)", &["x", "y"], "value", GAP_TARGET)
        .domain("|x| >= 5 and y <= 0.01", |row| !voigt_core(row))
        .parity("scipy")
        .run(&mut report, |row| voigt::voigt(row.get("x"), row.get("y")));
    report.finish();
}

#[test]
fn fresnel_integrals() {
    let table = table!("fresnel");
    let inputs = &["x"];
    let mut report = Report::default();
    Check::new(&table, "C(x)", inputs, "c", 1e-11)
        .parity("scipy_c")
        .run(&mut report, |row| fresnel::fresnel_c(row.get("x")));
    Check::new(&table, "S(x)", inputs, "s", 5e-13)
        .parity("scipy_s")
        .run(&mut report, |row| fresnel::fresnel_s(row.get("x")));

    let table = table!("fresnel_complex");
    let inputs = &["z_re", "z_im"];
    Check::new(&table, "C(z)", inputs, "c", 1e-11).run_complex(&mut report, |row| {
        fresnel::fresnel_c_complex(row.complex("z"))
    });
    Check::new(&table, "S(z)", inputs, "s", 1e-11).run_complex(&mut report, |row| {
        fresnel::fresnel_s_complex(row.complex("z"))
    });
    report.finish();
}

#[test]
fn marcum_q_function() {
    let table = table!("marcum_q");
    let mut report = Report::default();
    Check::new(&table, "Q_mu(a, b)", &["mu", "a", "b"], "value", 1e-13)
        .parity("scipy")
        .run(&mut report, |row| {
            marcum_q::marcum_q(row.get("mu"), row.get("a"), row.get("b"))
        });

    let table = table!("marcum_q_db");
    Check::new(&table, "dQ_M/db", &["m", "a", "b"], "value", 1e-13).run(&mut report, |row| {
        marcum_q::dq_db(row.get("m") as u32, row.get("a"), row.get("b"))
    });
    report.finish();
}

#[test]
fn parabolic_cylinder_functions() {
    let mut report = Report::default();
    let table = table!("parabolic_cylinder_uv");
    let inputs = &["a", "x"];
    let cases: [(&str, &str, Real2); 4] = [
        ("U(a, x)", "u", pcf::u),
        ("dU/dx", "du", pcf::du),
        ("V(a, x)", "v", pcf::v),
        ("dV/dx", "dv", pcf::dv),
    ];
    for (quantity, expected, function) in cases {
        Check::new(&table, quantity, inputs, expected, 5e-13)
            .run(&mut report, |row| function(row.get("a"), row.get("x")));
    }

    let table = table!("parabolic_cylinder_w");
    let cases: [(&str, &str, f64, Real2); 2] = [
        ("W(a, x)", "w", 2e-10, pcf::w),
        ("dW/dx", "dw", 5e-11, pcf::dw),
    ];
    for (quantity, expected, tolerance, function) in cases {
        Check::new(&table, quantity, inputs, expected, tolerance)
            .run(&mut report, |row| function(row.get("a"), row.get("x")));
    }

    let table = table!("parabolic_cylinder_d");
    let inputs = &["nu", "x"];
    Check::new(&table, "D_nu(x)", inputs, "d", 5e-13)
        .parity("scipy")
        .run(&mut report, |row| {
            pcf::parabolic_cylinder_d(row.get("nu"), row.get("x"))
        });
    Check::new(&table, "dD_nu/dx", inputs, "dd", 5e-13).run(&mut report, |row| {
        pcf::d_parabolic_cylinder_d(row.get("nu"), row.get("x"))
    });
    report.finish();
}

#[test]
fn whittaker_functions() {
    let table = table!("whittaker");
    let inputs = &["kappa", "mu", "z"];
    let mut report = Report::default();
    let cases: [(&str, &str, f64, Real3); 2] = [
        ("M(kappa, mu, z)", "m", 1e-14, whittaker::whittaker_m),
        ("dM/dz", "dm", 2e-14, whittaker::d_whittaker_m),
    ];
    for (quantity, expected, tolerance, function) in cases {
        Check::new(&table, quantity, inputs, expected, tolerance).run(&mut report, |row| {
            function(row.get("kappa"), row.get("mu"), row.get("z"))
        });
    }
    let cases: [(&str, &str, f64, Real3); 2] = [
        ("W(kappa, mu, z)", "w", 1e-11, whittaker::whittaker_w),
        ("dW/dz", "dw", 5e-11, whittaker::d_whittaker_w),
    ];
    for (quantity, expected, tolerance, function) in cases {
        Check::new(&table, quantity, inputs, expected, tolerance)
            .domain("regular", whittaker_w_regular)
            .run(&mut report, |row| {
                function(row.get("kappa"), row.get("mu"), row.get("z"))
            });
    }

    let table = table!("whittaker_complex");
    let inputs = &["kappa_re", "kappa_im", "mu_re", "mu_im", "z_re", "z_im"];
    Check::new(&table, "M (complex)", inputs, "m", 1e-14).run_complex(&mut report, |row| {
        whittaker::whittaker_m_complex(row.complex("kappa"), row.complex("mu"), row.complex("z"))
    });
    Check::new(&table, "W (complex)", inputs, "w", 2e-13).run_complex(&mut report, |row| {
        whittaker::whittaker_w_complex(row.complex("kappa"), row.complex("mu"), row.complex("z"))
    });
    report.finish();
}

/// Excludes two regions: the decayed tail `kappa < 0, z >= 30`, where
/// `|W| < 1e-7`, and the degenerate parameters where `1/2 + mu - kappa` is a
/// non-positive integer.
fn whittaker_w_regular(row: &Row) -> bool {
    let (kappa, mu, z) = (row.get("kappa"), row.get("mu"), row.get("z"));
    let c = 0.5 + mu - kappa;
    let degenerate = c <= 0.0 && c == c.round();
    let decayed = kappa < 0.0 && z >= 30.0;
    !(degenerate || decayed)
}

#[test]
#[ignore = "known accuracy gap: W and dW/dz have relative error up to 3.4e-7 where 1/2 + mu - kappa is a non-positive integer (kappa = 3, mu = 1.5, z = 0.01) and up to 5.3e-8 in the decayed tail (kappa = -2, z = 30)"]
fn whittaker_w_degenerate_and_tail() {
    let table = table!("whittaker");
    let mut report = Report::default();
    let cases: [(&str, &str, Real3); 2] = [
        ("W(kappa, mu, z)", "w", whittaker::whittaker_w),
        ("dW/dz", "dw", whittaker::d_whittaker_w),
    ];
    for (quantity, expected, function) in cases {
        Check::new(
            &table,
            quantity,
            &["kappa", "mu", "z"],
            expected,
            GAP_TARGET,
        )
        .domain("degenerate or tail", |row| !whittaker_w_regular(row))
        .run(&mut report, |row| {
            function(row.get("kappa"), row.get("mu"), row.get("z"))
        });
    }
    report.finish();
}

#[test]
fn coulomb_normalization_and_auxiliary() {
    let table = table!("coulomb_normalization");
    let mut report = Report::default();
    Check::new(&table, "C_l(eta)", &["ell", "eta"], "value", 2e-13).run(&mut report, |row| {
        coulomb::normalization(row.get("ell"), row.get("eta"))
    });

    let table = table!("coulomb_auxiliary");
    let inputs = &["ell", "eta", "rho"];
    Check::new(&table, "phase", inputs, "phase", 1e-13).run(&mut report, |row| {
        coulomb::phase(row.get("ell"), row.get("eta"), row.get("rho"))
    });
    Check::new(&table, "aux_g", inputs, "aux_g", 1e-13).run(&mut report, |row| {
        coulomb::aux_g(row.get("ell"), row.get("eta"))
    });
    // psi and i take a central difference in the order with the default step
    // eps^(1/3); rounding in phi divided by that step limits them to about 1e-8.
    Check::new(&table, "psi", inputs, "psi", 2e-8).run_complex(&mut report, |row| {
        coulomb::psi(row.get("ell"), row.get("eta"), row.get("rho"), None)
    });
    Check::new(&table, "i", inputs, "i", 2e-8).run_complex(&mut report, |row| {
        coulomb::i(row.get("ell"), row.get("eta"), row.get("rho"), None)
    });
    report.finish();
}

#[test]
fn coulomb_wave_functions() {
    let table = table!("coulomb");
    let inputs = &["ell", "eta", "rho"];
    let mut report = Report::default();
    let cases: [(&str, &str, Real3); 4] = [
        ("F_l(eta, rho)", "f", coulomb::f),
        ("G_l(eta, rho)", "g", coulomb::g),
        ("Re H+ = G", "g", |l, e, r| coulomb::h_plus(l, e, r).re),
        ("Im H+ = F", "f", |l, e, r| coulomb::h_plus(l, e, r).im),
    ];
    for (quantity, expected, function) in cases {
        Check::new(&table, quantity, inputs, expected, 1e-12).run(&mut report, |row| {
            function(row.get("ell"), row.get("eta"), row.get("rho"))
        });
    }
    report.finish();
}
