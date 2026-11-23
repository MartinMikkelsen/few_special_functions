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

mod reference;

use few_special_functions::{
    bose_einstein, clausen, coulomb, dawson, debye, fermi_dirac, fresnel, marcum_q,
    parabolic_cylinder as pcf, voigt, whittaker,
};
use reference::{Check, Report, Table};

/// A function of two or three real arguments, as listed in the check tables below.
type Real2 = fn(f64, f64) -> f64;
type Real3 = fn(f64, f64, f64) -> f64;

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
        clausen::clausen(
            clausen::ClausenOrder::new(row.get("n") as usize).expect("order in 1..=6"),
            row.get("theta"),
        )
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

#[test]
fn voigt_function() {
    let table = table!("voigt");
    let mut report = Report::default();
    Check::new(&table, "K(x, y)", &["x", "y"], "value", 5e-15)
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
        let order = std::num::NonZeroU32::new(row.get("m") as u32);
        order.map_or(f64::NAN, |m| marcum_q::dq_db(m, row.get("a"), row.get("b")))
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
    let cases: [(&str, &str, f64, Real3); 4] = [
        ("M(kappa, mu, z)", "m", 2e-13, whittaker::whittaker_m),
        ("dM/dz", "dm", 5e-13, whittaker::d_whittaker_m),
        ("W(kappa, mu, z)", "w", 2e-13, whittaker::whittaker_w),
        ("dW/dz", "dw", 5e-13, whittaker::d_whittaker_w),
    ];
    for (quantity, expected, tolerance, function) in cases {
        Check::new(&table, quantity, inputs, expected, tolerance).run(&mut report, |row| {
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
    // The Kummer series of phi cancels for larger rho: at rho = 5 phi itself is
    // good to about 6e-11, and phi_dot, psi and i inherit that.
    Check::new(&table, "phi_dot", inputs, "phi_dot", 2e-10).run_complex(&mut report, |row| {
        coulomb::phi_dot(row.get("ell"), row.get("eta"), row.get("rho"))
    });
    Check::new(&table, "psi", inputs, "psi", 5e-10).run_complex(&mut report, |row| {
        coulomb::psi(row.get("ell"), row.get("eta"), row.get("rho"))
    });
    Check::new(&table, "i", inputs, "i", 5e-10).run_complex(&mut report, |row| {
        coulomb::i(row.get("ell"), row.get("eta"), row.get("rho"))
    });

    let table = table!("coulomb_auxiliary_complex");
    let inputs = &["ell", "eta_re", "eta_im", "rho_re", "rho_im"];
    Check::new(&table, "psi (complex)", inputs, "psi", 1e-13).run_complex(&mut report, |row| {
        coulomb::psi_complex(row.get("ell"), row.complex("eta"), row.complex("rho"))
    });
    Check::new(&table, "i (complex)", inputs, "i", 1e-13).run_complex(&mut report, |row| {
        coulomb::i_complex(row.get("ell"), row.complex("eta"), row.complex("rho"))
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
